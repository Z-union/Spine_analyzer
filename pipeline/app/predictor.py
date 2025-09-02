import logging
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Union, List, Tuple, Optional, Dict
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# Use unified logger from main
logger = logging.getLogger("dicom-pipeline")


def compute_gaussian(
        patch_size: Union[Tuple[int, ...], List[int]],
        sigma_scale: float = 1 / 8,
        value_scaling_factor: float = 1.0,
        dtype=np.float16
) -> np.ndarray:
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = tuple(s // 2 for s in patch_size)
    tmp[center] = 1.0
    sigmas = [s * sigma_scale for s in patch_size]
    gaussian_map = gaussian_filter(tmp, sigma=sigmas, mode="constant", cval=0.0).astype(dtype)
    gaussian_map /= np.max(gaussian_map) / value_scaling_factor
    gaussian_map[gaussian_map == 0] = np.min(gaussian_map[gaussian_map != 0])
    return gaussian_map


def compute_sliding_steps(image_size: Tuple[int, ...], patch_size: Tuple[int, ...], step_fraction: float = 0.5) -> List[
    List[int]]:
    steps = []
    for img_dim, patch_dim in zip(image_size, patch_size):
        max_step = img_dim - patch_dim
        if max_step <= 0:
            steps.append([0])
            continue
        num_steps = int(np.ceil(max_step / (patch_dim * step_fraction))) + 1
        actual_step = max_step / max(num_steps - 1, 1)
        steps.append([int(round(i * actual_step)) for i in range(num_steps)])
    return steps


def get_sliding_window_slicers(
        image_size: Tuple[int, ...],
        patch_size: Tuple[int, ...] = (128, 96, 96),
        step_fraction: float = 0.5
) -> List[Tuple[slice, ...]]:
    slicers = []
    steps = compute_sliding_steps(image_size, patch_size, step_fraction)
    if len(patch_size) < len(image_size):
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        tuple([slice(None), d, slice(sx, sx + patch_size[0]), slice(sy, sy + patch_size[1])]))
    else:
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(tuple([slice(None), slice(sx, sx + patch_size[0]),
                                          slice(sy, sy + patch_size[1]), slice(sz, sz + patch_size[2])]))
    return slicers


@torch.inference_mode()
def local_inference(batch: torch.Tensor, network: torch.nn.Module) -> torch.Tensor:
    network.eval()
    return network(batch)


def triton_inference(
    batch_np: np.ndarray,
    triton_client: grpcclient.InferenceServerClient,
    model_name: str,
    input_name: str,
    output_names: Union[List[str], str],
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Send batch to Triton Inference Server via gRPC.
    
    Args:
        batch_np: Input batch CxDxHxW or BxCxDxHxW
        triton_client: Triton gRPC client
        model_name: Name of the model in Triton
        input_name: Name of the input tensor
        output_names: Output name (str) or list of output names
        
    Returns:
        np.ndarray if single output, dict {output_name: np.ndarray} otherwise
    """
    logger.debug(f"Triton inference request - model: {model_name}, input shape: {batch_np.shape}")
    
    try:
        if batch_np.ndim == 4:
            batch_np = batch_np[None]  # Add batch dimension
            logger.debug(f"Added batch dimension, new shape: {batch_np.shape}")

        batch_np = batch_np.astype(np.float32)

        # Convert output_names to list
        if isinstance(output_names, str):
            output_names = [output_names]

        # Prepare input
        inputs = [grpcclient.InferInput(input_name, batch_np.shape, "FP32")]
        inputs[0].set_data_from_numpy(batch_np)

        # Prepare outputs
        outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

        logger.debug(f"Sending inference request to Triton for model {model_name}")
        response = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        preds = {name: response.as_numpy(name) for name in output_names}
        logger.debug(f"Inference successful, output shapes: {[{k: v.shape for k, v in preds.items()}]}")
        
        # Return np.ndarray if single output
        if len(preds) == 1:
            return next(iter(preds.values()))
        return preds
        
    except InferenceServerException as e:
        logger.error(f"Triton inference failed for model {model_name}: {str(e)}")
        logger.debug(f"Failed inference details - input_shape: {batch_np.shape}, model: {model_name}")
        raise RuntimeError(f"Triton inference failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during Triton inference: {str(e)}")
        raise


def sliding_window_inference(
        data: np.ndarray,
        slicers: List[tuple],
        patch_size: Tuple[int, ...],
        batch_size: int = 4,
        num_heads: int = 9,
        use_gaussian: bool = True,
        mode: str = "3d",
        triton_client: grpcclient.InferenceServerClient = None,
        triton_model_name: Optional[str] = None,
        triton_input_name: Optional[str] = None,
        triton_output_name: Optional[str] = None
) -> np.ndarray:
    """
    Sliding window inference with Gaussian fusion.
    Works with Triton Inference Server via gRPC.
    
    Args:
        data: Input data array
        slicers: List of slice tuples for patches
        patch_size: Size of each patch
        batch_size: Batch size for inference
        num_heads: Number of output heads/classes
        use_gaussian: Whether to use Gaussian weighting
        mode: "2d" or "3d" mode
        triton_client: Triton gRPC client
        triton_model_name: Name of model in Triton
        triton_input_name: Input tensor name
        triton_output_name: Output tensor name
        
    Returns:
        Predicted logits array
    """
    logger.info(f"Starting sliding window inference - model: {triton_model_name}, "
               f"data shape: {data.shape}, patch_size: {patch_size}, "
               f"num_slicers: {len(slicers)}, batch_size: {batch_size}")
    
    try:
        predicted_logits = np.zeros((num_heads, *data.shape[1:]), dtype=np.float16)
        n_predictions = np.zeros(data.shape[1:], dtype=np.float16)
        gaussian_map = compute_gaussian(patch_size) if use_gaussian else 1.0

        batch_patches, batch_slicers = [], []
        total_batches = (len(slicers) + batch_size - 1) // batch_size
        processed_batches = 0

        for i, sl in enumerate(slicers):
            patch = data[sl][None]
            if mode == "2d" and patch.ndim == 5:
                patch = patch.squeeze(2)
            batch_patches.append(patch)
            batch_slicers.append(sl)

            if len(batch_patches) == batch_size or sl == slicers[-1]:
                batch_tensor = np.concatenate(batch_patches, axis=0)
                
                logger.debug(f"Processing batch {processed_batches + 1}/{total_batches}, "
                           f"batch shape: {batch_tensor.shape}")
                
                try:
                    batch_pred = triton_inference(
                        batch_tensor,
                        triton_client,
                        triton_model_name,
                        triton_input_name,
                        triton_output_name
                    )
                except Exception as e:
                    logger.error(f"Inference failed at batch {processed_batches + 1}/{total_batches}: {str(e)}")
                    raise

                for b, sl_b in enumerate(batch_slicers):
                    pred_patch = batch_pred[b]
                    if use_gaussian:
                        pred_patch = pred_patch * gaussian_map
                    predicted_logits[sl_b] += pred_patch
                    n_predictions[sl_b[1:]] += gaussian_map

                batch_patches, batch_slicers = [], []
                processed_batches += 1
                
                if processed_batches % 10 == 0:
                    logger.debug(f"Processed {processed_batches}/{total_batches} batches")

        predicted_logits /= n_predictions
        
        logger.info(f"Sliding window inference completed successfully - "
                   f"output shape: {predicted_logits.shape}")
        
        return predicted_logits
        
    except Exception as e:
        logger.error(f"Sliding window inference failed: {str(e)}")
        raise


# -------------------------------
# Пример локального и Triton инференса
# -------------------------------
if __name__ == "__main__":
    # Dummy volume
    C, D, H, W = 1, 128, 128, 128
    dummy_volume = torch.randn((C, D, H, W), dtype=torch.float32)
    patch_size = (64, 64, 64)
    slicers = get_sliding_window_slicers(dummy_volume.shape, patch_size)


    # --- Локальная модель ---
    class DummyNet(torch.nn.Module):
        def forward(self, x):
            return torch.randn((x.shape[0], 9, *x.shape[2:]), device=x.device, dtype=torch.half)


    net = DummyNet()
    logits_local = sliding_window_inference(dummy_volume, slicers, net, patch_size, batch_size=2, device="cpu")
    print("Local output shape:", logits_local.shape)

    # --- Triton gRPC (пример) ---
    # triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
    # logits_triton = sliding_window_inference(
    #     dummy_volume, slicers, triton_client, patch_size, batch_size=2,
    #     device="cpu", triton_mode=True, triton_model_name="my_model",
    #     triton_input_name="input", triton_output_name="output"
    # )
    # print("Triton output shape:", logits_triton.shape)
