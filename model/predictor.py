import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Union, List, Tuple, Optional
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def compute_gaussian(
        patch_size: Union[Tuple[int, ...], List[int]],
        sigma_scale: float = 1 / 8,
        value_scaling_factor: float = 1.0,
        device: str = "cpu",
        dtype=torch.float16
) -> torch.Tensor:
    tmp = np.zeros(patch_size, dtype=np.float32)
    center = tuple(s // 2 for s in patch_size)
    tmp[center] = 1.0
    sigmas = [s * sigma_scale for s in patch_size]
    gaussian_map = gaussian_filter(tmp, sigma=sigmas, mode="constant", cval=0.0)
    gaussian_map = torch.from_numpy(gaussian_map).to(device=device, dtype=dtype)
    gaussian_map /= torch.max(gaussian_map) / value_scaling_factor
    gaussian_map[gaussian_map == 0] = torch.min(gaussian_map[gaussian_map != 0])
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


def triton_inference(batch_tensor: torch.Tensor, triton_client: grpcclient.InferenceServerClient, model_name: str,
                     input_name: str, output_name: str) -> torch.Tensor:
    """
    Отправка батча на Triton ONNX Server через gRPC.
    batch_tensor: CxDxHxW или BxCxDxHxW
    """
    batch_np = batch_tensor.cpu().numpy().astype(np.float32)
    if batch_np.ndim == 4:
        batch_np = batch_np[None]  # добавляем batch dim

    inputs = [grpcclient.InferInput(input_name, batch_np.shape, "FP32")]
    inputs[0].set_data_from_numpy(batch_np)

    outputs = [grpcclient.InferRequestedOutput(output_name)]

    try:
        response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        pred = response.as_numpy(output_name)
        pred_tensor = torch.from_numpy(pred).to(batch_tensor.device).half()
        return pred_tensor
    except InferenceServerException as e:
        raise RuntimeError(f"Triton inference failed: {e}")


def sliding_window_inference(
        data: torch.Tensor,
        slicers: List[tuple],
        model_or_triton,
        patch_size: Tuple[int, ...],
        batch_size: int = 4,
        num_heads: int = 9,
        use_gaussian: bool = True,
        device: str = "cpu",
        mode: str = "3d",
        triton_mode: bool = False,
        triton_model_name: Optional[str] = None,
        triton_input_name: Optional[str] = None,
        triton_output_name: Optional[str] = None
) -> torch.Tensor:
    """
    Скользящее окно с Gaussian fusion, работает локально или через Triton ONNX Server.
    """
    predicted_logits = torch.zeros((num_heads, *data.shape[1:]), dtype=torch.half, device=device)
    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=device)
    gaussian_map = compute_gaussian(patch_size, device=device) if use_gaussian else 1.0

    batch_patches, batch_slicers = [], []

    for sl in slicers:
        patch = data[sl][None]
        if mode == "2d" and patch.ndim == 5:
            patch = patch.squeeze(2)
        batch_patches.append(patch)
        batch_slicers.append(sl)

        if len(batch_patches) == batch_size or sl == slicers[-1]:
            batch_tensor = np.concatenate(batch_patches, axis=0)

            if triton_mode:
                batch_pred = triton_inference(
                    batch_tensor,
                    model_or_triton,
                    triton_model_name,
                    triton_input_name,
                    triton_output_name
                )
            else:
                batch_pred = local_inference(torch.tensor(batch_tensor).to(device), model_or_triton)

            for b, sl_b in enumerate(batch_slicers):
                pred_patch = batch_pred[b]
                if use_gaussian:
                    pred_patch = pred_patch * gaussian_map
                predicted_logits[sl_b] += pred_patch
                n_predictions[sl_b[1:]] += gaussian_map

            batch_patches, batch_slicers = [], []

    predicted_logits /= n_predictions
    return predicted_logits


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
