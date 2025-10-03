"""
Module for creating and uploading DICOM SR (Structured Report) and SC (Secondary Capture) to Orthanc
"""
from fastapi import HTTPException
import json
import logging
import io
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import requests
from pydicom import dcmread

from .config import settings

# Используем единый логгер из main
logger = logging.getLogger("dicom-pipeline")

class ResourceNotFoundError(HTTPException):
    def __init__(
        self,
        status_code: int = 404,
        detail: str = "Requested resource doesn't exist",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(status_code, detail, headers)
        self.status_code = status_code
        self.detail = detail


def flatten_to_scalar(val):
    """Recursively descend into nested lists or numpy arrays until a scalar is found.
    Always returns native Python str, int, or float.
    Converts all other types (including np.str_, np.int64, bool, None, dict) to str.
    """
    while isinstance(val, (list, np.ndarray)):
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                val = val.item() 
                break
            if val.size == 0:
                return ""
            val = val[0]  
        else:  # list
            if len(val) == 0:
                return ""
            val = val[0]

    if isinstance(val, np.number):
        val = val.item()  
    elif isinstance(val, np.str_):
        val = str(val)   
    elif isinstance(val, np.bool_):
        val = bool(val)  

    if isinstance(val, (int, float)):
        return val
    else:
        return str(val)


class DICOMReportGenerator:
    """Generator for DICOM SR and SC reports"""
    
    def __init__(self, study_info: Optional[Dict] = None):
        """
        Initialize report generator
        
        Args:
            study_info: Dictionary with study metadata (StudyInstanceUID, PatientID, etc.)
        """
        self.study_info = study_info or {}
        self.orthanc_url = settings.ORTHANC_URL
        self.orthanc_auth = (settings.ORTHANC_USER, settings.ORTHANC_PASSWORD)
    
    def _create_concept_name_sequence(self, code_value: str, scheme: str, meaning: str) -> List[Dataset]:
        """Helper to create ConceptNameCodeSequence"""
        concept = Dataset()
        concept.CodeValue = code_value
        concept.CodingSchemeDesignator = scheme
        concept.CodeMeaning = meaning
        return [concept]
    
    def _create_measurement_units_sequence(self, code_value: str, scheme: str, meaning: str) -> List[Dataset]:
        """Helper to create MeasurementUnitsCodeSequence"""
        units = Dataset()
        units.CodeValue = code_value
        units.CodingSchemeDesignator = scheme
        units.CodeMeaning = meaning
        return [units]
    
    def _create_text_item(self, code_value: str, meaning: str, text_value: str) -> Dataset:
        """Helper to create TEXT content item"""
        item = Dataset()
        item.ValueType = 'TEXT'
        item.ConceptNameCodeSequence = self._create_concept_name_sequence(code_value, 'DCM', meaning)
        item.TextValue = text_value
        return item
    
    def _create_num_item(self, code_value: str, meaning: str, numeric_value: float, 
                        unit_code: str, unit_scheme: str, unit_meaning: str) -> Dataset:
        """Helper to create NUM content item"""
        item = Dataset()
        item.ValueType = 'NUM'
        item.ConceptNameCodeSequence = self._create_concept_name_sequence(code_value, 'DCM', meaning)
        
        # Create MeasuredValueSequence
        measured_value = Dataset()
        measured_value.NumericValue = str(numeric_value)
        measured_value.MeasurementUnitsCodeSequence = self._create_measurement_units_sequence(
            unit_code, unit_scheme, unit_meaning
        )
        item.MeasuredValueSequence = [measured_value]
        
        return item
    
    def _get_resource_uid(self, resourse: str, resource_id: str) -> str:
        data = {
            "Level": resourse,  # Study
            "Query": {"StudyInstanceUID": resource_id},
        }
        uid = self.find(data)
        if uid:
            return uid[0]
        else:
            raise ResourceNotFoundError(detail="Requested resource doesn't exist")

    def find(
        self, search_params: dict, headers: dict[str, str] | None = None,
    ) -> list[str | dict]:
        if not headers:
            headers = {"Content-Type": "application/json"}
        # url = f"{self.protocol}://{self.host}:{self.port}/tools/find"
            
            response = requests.post(
                f"{self.orthanc_url}/tools/find",
                auth=self.orthanc_auth,
                data=json.dumps(search_params).encode(),
                headers={'Content-Type': 'application/dicom'}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully get study id from orthanc: {result}")

        return result



    def create_structured_report(self, 
                                grading_results: Dict,
                                pathology_measurements: Dict,
                                study_id: str) -> bytes:
        """
        Create DICOM Structured Report with analysis results
        
        Args:
            grading_results: Dictionary with grading analysis results
            pathology_measurements: Dictionary with pathology measurements
            study_id: Orthanc study ID
            
        Returns:
            DICOM SR file as bytes
        """
        # Create new SR dataset
        ds = Dataset()
        
        # Patient Module
        ds.PatientName = self.study_info.get('PatientName', 'Anonymous')
        ds.PatientID = self.study_info.get('PatientID', 'Unknown')
        ds.PatientBirthDate = self.study_info.get('PatientBirthDate', '')
        ds.PatientSex = self.study_info.get('PatientSex', '')
        
        # General Study Module
        ds.StudyInstanceUID = self.study_info.get('StudyInstanceUID', generate_uid())
        ds.StudyDate = self.study_info.get('StudyDate', datetime.now().strftime('%Y%m%d'))
        ds.StudyTime = self.study_info.get('StudyTime', datetime.now().strftime('%H%M%S'))
        ds.ReferringPhysicianName = self.study_info.get('ReferringPhysicianName', '')
        ds.StudyID = self.study_info.get('StudyID', study_id)
        ds.AccessionNumber = self.study_info.get('AccessionNumber', '')
        
        # SR Document Series Module
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = 999  # High number to appear last
        ds.SeriesDate = datetime.now().strftime('%Y%m%d')
        ds.SeriesTime = datetime.now().strftime('%H%M%S')
        ds.Modality = 'SR'
        ds.SeriesDescription = 'Spine Analysis Report'
        
        # General Equipment Module
        ds.Manufacturer = 'Spine Analyzer'
        ds.InstitutionName = self.study_info.get('InstitutionName', '')
        ds.StationName = 'AI Analysis Station'
        
        # SR Document General Module
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.11'  # Basic Text SR
        ds.SOPInstanceUID = generate_uid()
        ds.InstanceNumber = 1
        ds.CompletionFlag = 'COMPLETE'
        ds.VerificationFlag = 'UNVERIFIED'
        ds.ContentDate = datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.now().strftime('%H%M%S')
        
        # Create SR content
        content_items = []
        
        # Add title - create as Dataset
        title_item = Dataset()
        title_item.ValueType = 'CONTAINER'
        
        # Create ConceptNameCodeSequence as a sequence of Datasets
        concept_name = Dataset()
        concept_name.CodeValue = '121070'
        concept_name.CodingSchemeDesignator = 'DCM'
        concept_name.CodeMeaning = 'Findings'
        title_item.ConceptNameCodeSequence = [concept_name]
        
        title_item.ContinuityOfContent = 'SEPARATE'
        title_item.ContentSequence = []
        
        content_items.append(title_item)
        
        # Add grading results
        if grading_results:
            for disk_label, result in grading_results.items():
                if 'predictions' in result:
                    level_name = result.get('level_name', f'Disk_{disk_label}')
                    disk_content = []
                    
                    # Add disk level identifier - create as Dataset
                    disk_level_item = Dataset()
                    disk_level_item.ValueType = 'TEXT'
                    
                    # Create ConceptNameCodeSequence
                    concept = Dataset()
                    concept.CodeValue = '121071'
                    concept.CodingSchemeDesignator = 'DCM'
                    concept.CodeMeaning = 'Finding'
                    disk_level_item.ConceptNameCodeSequence = [concept]
                    
                    disk_level_item.TextValue = f'Disk Level: {level_name}'
                    disk_content.append(disk_level_item)
                    
                    # Add grading scores
                    for category, value in result['predictions'].items():
                        safe_value = flatten_to_scalar(value)
                        print(f"safe_value = {safe_value}, category = {category}, value = {value}, predictions = {result['predictions']}, \
                              type_value = {type(value)}")
                        score_item = self._create_num_item(
                            '121072', category, safe_value,
                            '1', 'UCUM', 'grade'
                        )
                        disk_content.append(score_item)

                    
                    # Add pathology measurements if available
                    if disk_label in pathology_measurements:
                        measurements = pathology_measurements[disk_label]
                        
                        # Herniation measurements
                        if measurements.get('herniation'):
                            hernia = measurements['herniation']
                            if hernia.get('detected'):
                                volume_item = self._create_num_item(
                                    '121073', 'Herniation Volume', 
                                    hernia.get('volume_mm3', 0),
                                    'mm3', 'UCUM', 'cubic millimeter'
                                )
                                disk_content.append(volume_item)
                                
                                protrusion_item = self._create_num_item(
                                    '121074', 'Herniation Protrusion',
                                    hernia.get('max_protrusion_mm', 0),
                                    'mm', 'UCUM', 'millimeter'
                                )
                                disk_content.append(protrusion_item)
                        
                        # Spondylolisthesis measurements
                        if measurements.get('spondylolisthesis'):
                            spondy = measurements['spondylolisthesis']
                            if spondy.get('detected'):
                                displacement_item = self._create_num_item(
                                    '121075', 'Spondylolisthesis Displacement',
                                    spondy.get('displacement_mm', 0),
                                    'mm', 'UCUM', 'millimeter'
                                )
                                disk_content.append(displacement_item)
                                
                                grade_item = self._create_text_item(
                                    '121076', 'Spondylolisthesis Grade',
                                    f"Grade {spondy.get('grade', 'Unknown')}"
                                )
                                disk_content.append(grade_item)
                    
                    # Add disk container to main content - create as Dataset
                    disk_container = Dataset()
                    disk_container.ValueType = 'CONTAINER'
                    disk_container.ConceptNameCodeSequence = self._create_concept_name_sequence(
                        '121077', 'DCM', f'Disk Analysis: {level_name}'
                    )
                    disk_container.ContinuityOfContent = 'SEPARATE'
                    disk_container.ContentSequence = disk_content
                    
                    # Add to title item's ContentSequence
                    title_item.ContentSequence.append(disk_container)
        
        ds.ContentSequence = content_items
        
        # Set transfer syntax
        ds.is_implicit_VR = False
        ds.is_little_endian = True
        
        # Create file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.ImplementationClassUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # Create FileDataset
        filename = f"SR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dcm"
        file_ds = FileDataset(filename, ds, file_meta=file_meta, preamble=b"\0" * 128)
        
        # Save to bytes
        with io.BytesIO() as buffer:
            file_ds.save_as(buffer, write_like_original=False)
            return buffer.getvalue()
    
    def create_enhanced_segmentation_object(self,
                                          segmentation_data: np.ndarray,
                                          reference_image_nifti,
                                          study_id: str,
                                          grading_results: Dict = None,
                                          series_description: str = "AI Spine Segmentation") -> bytes:
        """
        Create DICOM Segmentation Object with pathological areas included
        
        Args:
            segmentation_data: 3D numpy array with segmentation labels
            reference_image_nifti: Reference NIfTI image for spatial information
            study_id: Orthanc study ID
            grading_results: Grading results to identify pathological areas
            series_description: Description for the segmentation series
            
        Returns:
            DICOM Segmentation Object as bytes
        """
        try:
            # Create enhanced segmentation that includes pathological areas
            enhanced_seg_data = segmentation_data.copy()
            
            # Add pathological areas if grading results are provided
            if grading_results:
                from .config import settings
                
                logger.info(f"Processing grading results for pathological areas: {len(grading_results)} disks")
                
                # Find pathological disks
                pathological_disks = set()
                for disk_id, res in grading_results.items():
                    logger.debug(f"Checking disk {disk_id}: {res}")
                    preds = res.get('predictions') if isinstance(res, dict) else None
                    if not isinstance(preds, dict):
                        logger.debug(f"Disk {disk_id}: no predictions found")
                        continue
                        
                    # Check for herniation
                    hernia_val = preds.get('Disc herniation', 0)
                    if isinstance(hernia_val, list):
                        hernia_val = hernia_val[0] if len(hernia_val) > 0 else 0
                    logger.debug(f"Disk {disk_id}: herniation value = {hernia_val}")
                    if hernia_val > 0:
                        pathological_disks.add(int(disk_id))
                        logger.info(f"Disk {disk_id}: marked as pathological (herniation = {hernia_val})")
                        
                    # Check for bulging
                    bulge_val = preds.get('Disc bulging', 0)
                    if isinstance(bulge_val, list):
                        bulge_val = bulge_val[0] if len(bulge_val) > 0 else 0
                    logger.debug(f"Disk {disk_id}: bulging value = {bulge_val}")
                    if bulge_val > 0:
                        pathological_disks.add(int(disk_id))
                        logger.info(f"Disk {disk_id}: marked as pathological (bulging = {bulge_val})")
                
                logger.info(f"Found {len(pathological_disks)} pathological disks: {sorted(pathological_disks)}")
                
                # Compute pathological masks and add them to segmentation
                if pathological_disks:
                    canal_label = settings.CANAL_LABEL
                    pathology_label = 200  # Special label for pathological areas (fits in uint8)
                    
                    logger.info(f"Adding pathological areas for disks: {sorted(pathological_disks)}")
                    logger.info(f"Using canal label: {canal_label}")
                    
                    # Check if disks exist in segmentation
                    unique_labels = np.unique(enhanced_seg_data)
                    logger.info(f"Available labels in segmentation: {sorted(unique_labels.tolist())}")
                    
                    for disk_id in pathological_disks:
                        if disk_id not in unique_labels:
                            logger.warning(f"Disk {disk_id} not found in segmentation data")
                            continue
                            
                        logger.info(f"Computing protrusion mask for disk {disk_id}")
                        protrusion_mask = self._compute_protrusion_mask(enhanced_seg_data, disk_id, canal_label)
                        if np.any(protrusion_mask):
                            enhanced_seg_data[protrusion_mask] = pathology_label
                            logger.info(f"Added {np.sum(protrusion_mask)} pathological voxels for disk {disk_id}")
                        else:
                            logger.warning(f"No pathological voxels found for disk {disk_id}")
                else:
                    logger.info("No pathological disks found")
            else:
                logger.info("No grading results provided for pathological area detection")
            
            return self._create_segmentation_dicom(enhanced_seg_data, reference_image_nifti, study_id, series_description)
            
        except Exception as e:
            logger.error(f"Failed to create enhanced DICOM Segmentation Object: {e}")
            raise

    def _compute_protrusion_mask(self, seg3d: np.ndarray, disk_label: int, canal_label: int) -> np.ndarray:
        """
        Compute protrusion mask for a specific disk (copied from pipeline.py)
        """
        disk_mask = (seg3d == disk_label)
        if not np.any(disk_mask):
            return np.zeros_like(seg3d, dtype=bool)

        canal_mask = (seg3d == canal_label)
        if not np.any(canal_mask):
            logger.debug(f"No canal found for disk {disk_label}, using simple morphological approach")
            return self._simple_protrusion_mask(seg3d, disk_label)

        coords = np.argwhere(disk_mask)
        if coords.size == 0:
            return np.zeros_like(seg3d, dtype=bool)

        # Define region of interest around the disk
        zc = int(np.round(coords[:, 0].mean()))
        z_from = max(0, zc - 3)
        z_to = min(seg3d.shape[0], zc + 4)

        protrusion = np.zeros_like(seg3d, dtype=bool)
        
        for z in range(z_from, z_to):
            disk_slice = disk_mask[z].astype(np.uint8)
            canal_slice = canal_mask[z].astype(np.uint8)
            
            if disk_slice.sum() == 0:
                continue
                
            # Find canal-directed protrusions
            slice_protrusion = self._find_canal_directed_protrusions(disk_slice, canal_slice)
            protrusion[z] = slice_protrusion

        return protrusion

    def _simple_protrusion_mask(self, seg3d: np.ndarray, disk_label: int) -> np.ndarray:
        """
        Simple protrusion detection without canal information (fallback)
        """
        disk_mask = (seg3d == disk_label)
        coords = np.argwhere(disk_mask)
        if coords.size == 0:
            return np.zeros_like(seg3d, dtype=bool)

        zc = int(np.round(coords[:, 0].mean()))
        z_from = max(0, zc - 2)
        z_to = min(seg3d.shape[0], zc + 3)

        protrusion = np.zeros_like(seg3d, dtype=bool)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        for z in range(z_from, z_to):
            disk_slice = disk_mask[z].astype(np.uint8)
            if disk_slice.sum() == 0:
                continue
            normal_disk = cv2.morphologyEx(disk_slice, cv2.MORPH_OPEN, kernel)
            protruding = (disk_slice.astype(bool) & (~normal_disk.astype(bool)))
            
            # Remove small artifacts
            if np.sum(protruding) > 10:  # minimum protrusion size
                protrusion[z] = protruding

        return protrusion

    def _find_canal_directed_protrusions(self, disk_slice: np.ndarray, canal_slice: np.ndarray) -> np.ndarray:
        """
        Find disk parts that protrude toward the canal
        """
        if disk_slice.sum() == 0 or canal_slice.sum() == 0:
            return np.zeros_like(disk_slice, dtype=bool)
        
        # Find centers of mass for disk and canal
        disk_moments = cv2.moments(disk_slice)
        canal_moments = cv2.moments(canal_slice)
        
        if disk_moments['m00'] == 0 or canal_moments['m00'] == 0:
            return np.zeros_like(disk_slice, dtype=bool)
        
        disk_center = (int(disk_moments['m10'] / disk_moments['m00']), 
                       int(disk_moments['m01'] / disk_moments['m00']))
        canal_center = (int(canal_moments['m10'] / canal_moments['m00']), 
                        int(canal_moments['m01'] / canal_moments['m00']))
        
        # Calculate direction from disk to canal
        direction = np.array([canal_center[0] - disk_center[0], 
                             canal_center[1] - disk_center[1]])
        
        if np.linalg.norm(direction) == 0:
            return np.zeros_like(disk_slice, dtype=bool)
        
        direction = direction / np.linalg.norm(direction)
        
        # Create "normal" disk shape using erosion + dilation
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Erosion removes protrusions
        eroded = cv2.erode(disk_slice, kernel, iterations=1)
        # Dilation restores size but without sharp protrusions
        normal_disk = cv2.dilate(eroded, kernel, iterations=1)
        
        # Find difference - potential protrusions
        potential_protrusions = disk_slice.astype(bool) & (~normal_disk.astype(bool))
        
        if not np.any(potential_protrusions):
            return np.zeros_like(disk_slice, dtype=bool)
        
        # Filter protrusions by direction toward canal
        protrusion_coords = np.argwhere(potential_protrusions)
        valid_protrusions = np.zeros_like(disk_slice, dtype=bool)
        
        for coord in protrusion_coords:
            y, x = coord  # OpenCV uses (y, x)
            
            # Vector from disk center to protrusion point
            to_protrusion = np.array([x - disk_center[0], y - disk_center[1]])
            
            if np.linalg.norm(to_protrusion) == 0:
                continue
                
            to_protrusion = to_protrusion / np.linalg.norm(to_protrusion)
            
            # Check if protrusion is directed toward canal (cosine > 0.3)
            dot_product = np.dot(direction, to_protrusion)
            if dot_product > 0.3:  # angle less than ~70 degrees
                valid_protrusions[y, x] = True
        
        # Remove small artifacts and isolated pixels
        if np.sum(valid_protrusions) < 5:  # minimum protrusion size
            return np.zeros_like(disk_slice, dtype=bool)
        
        # Morphological closing to connect nearby areas
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        valid_protrusions = cv2.morphologyEx(valid_protrusions.astype(np.uint8), 
                                            cv2.MORPH_CLOSE, close_kernel).astype(bool)
        
        return valid_protrusions

    def create_segmentation_object(self,
                                  segmentation_data: np.ndarray,
                                  reference_image_nifti,
                                  study_id: str,
                                  series_description: str = "Spine Segmentation") -> bytes:
        """
        Create basic DICOM Segmentation Object from segmentation mask (legacy method)
        """
        return self._create_segmentation_dicom(segmentation_data, reference_image_nifti, study_id, series_description)

    def _create_segmentation_dicom(self,
                                  segmentation_data: np.ndarray,
                                  reference_image_nifti,
                                  study_id: str,
                                  series_description: str = "Spine Segmentation") -> bytes:
        """
        Create DICOM Segmentation Object from segmentation mask
        
        Args:
            segmentation_data: 3D numpy array with segmentation labels
            reference_image_nifti: Reference NIfTI image for spatial information
            study_id: Orthanc study ID
            series_description: Description for the segmentation series
            
        Returns:
            DICOM Segmentation Object as bytes
        """
        try:
            # Create dataset
            ds = Dataset()
            
            # Patient Module
            ds.PatientName = self.study_info.get('PatientName', 'Anonymous')
            ds.PatientID = self.study_info.get('PatientID', 'Unknown')
            ds.PatientBirthDate = self.study_info.get('PatientBirthDate', '')
            ds.PatientSex = self.study_info.get('PatientSex', '')
            
            # General Study Module
            ds.StudyInstanceUID = self.study_info.get('StudyInstanceUID', generate_uid())
            ds.StudyDate = self.study_info.get('StudyDate', datetime.now().strftime('%Y%m%d'))
            ds.StudyTime = self.study_info.get('StudyTime', datetime.now().strftime('%H%M%S'))
            ds.ReferringPhysicianName = self.study_info.get('ReferringPhysicianName', '')
            ds.StudyID = self.study_info.get('StudyID', study_id)
            ds.AccessionNumber = self.study_info.get('AccessionNumber', '')
            
            # General Series Module
            ds.SeriesInstanceUID = generate_uid()
            ds.SeriesNumber = 997  # High number for segmentation
            ds.SeriesDate = datetime.now().strftime('%Y%m%d')
            ds.SeriesTime = datetime.now().strftime('%H%M%S')
            ds.Modality = 'SEG'  # Segmentation
            ds.SeriesDescription = series_description
            
            # General Equipment Module
            ds.Manufacturer = 'Spine Analyzer'
            ds.InstitutionName = self.study_info.get('InstitutionName', '')
            ds.StationName = 'AI Analysis Station'
            ds.ManufacturerModelName = 'SpineAnalyzer v1.0'
            ds.SoftwareVersions = '1.0.0'
            
            # General Image Module
            ds.InstanceNumber = 1
            ds.ImageType = ['DERIVED', 'PRIMARY', 'SEGMENTATION']
            ds.ContentDate = datetime.now().strftime('%Y%m%d')
            ds.ContentTime = datetime.now().strftime('%H%M%S')
            
            # Segmentation Image Module
            ds.ImageType = ['DERIVED', 'PRIMARY']
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            
            # Get dimensions from segmentation data
            if len(segmentation_data.shape) == 3:
                depth, height, width = segmentation_data.shape
            else:
                raise ValueError("Segmentation data must be 3D")
            
            ds.Rows = height
            ds.Columns = width
            ds.NumberOfFrames = depth
            
            # Image Position and Orientation from reference image
            if hasattr(reference_image_nifti, 'affine'):
                affine = reference_image_nifti.affine
                # Extract position from affine matrix
                ds.ImagePositionPatient = [float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3])]
                # Extract orientation from affine matrix
                row_cosines = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
                col_cosines = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
                ds.ImageOrientationPatient = [float(x) for x in list(row_cosines) + list(col_cosines)]
                # Pixel spacing
                ds.PixelSpacing = [float(np.linalg.norm(affine[:3, 1])), float(np.linalg.norm(affine[:3, 0]))]
                ds.SliceThickness = float(np.linalg.norm(affine[:3, 2]))
            else:
                # Default values if no affine information
                ds.ImagePositionPatient = [0.0, 0.0, 0.0]
                ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                ds.PixelSpacing = [1.0, 1.0]
                ds.SliceThickness = 1.0
            
            # Segmentation-specific attributes
            ds.SegmentationType = 'BINARY'
            ds.SegmentationFractionalType = 'PROBABILITY'
            ds.MaximumFractionalValue = 255
            
            # Create Segment Sequence
            segment_sequence = []
            
            # Get unique labels (excluding background)
            unique_labels = np.unique(segmentation_data)
            unique_labels = unique_labels[unique_labels != 0]
            
            # Define segment colors and names
            segment_info = {
                # Vertebrae
                **{i: {'name': f'Vertebra_{i}', 'color': [0, 255, 0]} for i in range(11, 51)},
                # Discs
                63: {'name': 'Disc_Th12-L1', 'color': [255, 255, 0]},
                71: {'name': 'Disc_L1-L2', 'color': [255, 255, 0]},
                91: {'name': 'Disc_L4-L5', 'color': [255, 255, 0]},
                100: {'name': 'Disc_L5-S1', 'color': [255, 255, 0]},
                # Special structures
                1: {'name': 'Spinal_Cord', 'color': [0, 255, 255]},
                2: {'name': 'Spinal_Canal', 'color': [255, 0, 0]},
                # Pathological areas
                200: {'name': 'Disc_Pathology', 'color': [255, 0, 0]},  # Red for pathologies
            }
            
            for idx, label in enumerate(sorted(unique_labels)):
                segment = Dataset()
                segment.SegmentNumber = idx + 1
                segment.SegmentLabel = segment_info.get(label, {'name': f'Structure_{label}'})['name']
                segment.SegmentAlgorithmType = 'AUTOMATIC'
                segment.SegmentAlgorithmName = 'SpineAnalyzer AI'
                
                # Segment color
                color = segment_info.get(label, {'color': [128, 128, 128]})['color']
                segment.RecommendedDisplayCIELabValue = color
                
                # Anatomical codes (simplified)
                segment.SegmentedPropertyCategoryCodeSequence = [Dataset()]
                segment.SegmentedPropertyCategoryCodeSequence[0].CodeValue = 'T-D0050'
                segment.SegmentedPropertyCategoryCodeSequence[0].CodingSchemeDesignator = 'SRT'
                segment.SegmentedPropertyCategoryCodeSequence[0].CodeMeaning = 'Tissue'
                
                segment.SegmentedPropertyTypeCodeSequence = [Dataset()]
                segment.SegmentedPropertyTypeCodeSequence[0].CodeValue = 'T-D0050'
                segment.SegmentedPropertyTypeCodeSequence[0].CodingSchemeDesignator = 'SRT'
                segment.SegmentedPropertyTypeCodeSequence[0].CodeMeaning = 'Tissue'
                
                segment_sequence.append(segment)
            
            ds.SegmentSequence = segment_sequence
            
            # Convert segmentation data to binary masks and pack
            # For simplicity, we'll create a multi-frame image where each frame contains all segments
            # In a full implementation, you might want separate frames per segment
            
            # Normalize segmentation data to 0-255 range
            seg_normalized = np.zeros_like(segmentation_data, dtype=np.uint8)
            for i, label in enumerate(sorted(unique_labels)):
                seg_normalized[segmentation_data == label] = min(255, (i + 1) * (255 // len(unique_labels)))
            
            # Reshape for DICOM (frames, rows, columns)
            pixel_data = seg_normalized.astype(np.uint8)
            ds.PixelData = pixel_data.tobytes()
            
            # SOP Common Module
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.66.4'  # Segmentation Storage
            ds.SOPInstanceUID = generate_uid()
            
            # Set transfer syntax
            ds.is_implicit_VR = False
            ds.is_little_endian = True
            
            # Create file meta information
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.ImplementationClassUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            
            # Create FileDataset
            filename = f"SEG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dcm"
            file_ds = FileDataset(filename, ds, file_meta=file_meta, preamble=b"\0" * 128)
            
            # Save to bytes
            with io.BytesIO() as buffer:
                file_ds.save_as(buffer, write_like_original=False)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Failed to create DICOM Segmentation Object: {e}")
            raise

    def create_secondary_capture(self, 
                                images: List[np.ndarray],
                                study_id: str,
                                series_description: str = "Spine Segmentation Overlay") -> List[bytes]:
        """
        Create DICOM Secondary Capture images with segmentation overlays
        
        Args:
            images: List of numpy arrays with overlay images
            study_id: Orthanc study ID
            series_description: Description for the series
            
        Returns:
            List of DICOM SC files as bytes
        """
        dicom_files = []
        series_uid = generate_uid()
        
        for idx, image in enumerate(images):
            # Create dataset
            ds = Dataset()
            
            # Patient Module
            ds.PatientName = self.study_info.get('PatientName', 'Anonymous')
            ds.PatientID = self.study_info.get('PatientID', 'Unknown')
            ds.PatientBirthDate = self.study_info.get('PatientBirthDate', '')
            ds.PatientSex = self.study_info.get('PatientSex', '')
            
            # General Study Module
            ds.StudyInstanceUID = self.study_info.get('StudyInstanceUID', generate_uid())
            ds.StudyDate = self.study_info.get('StudyDate', datetime.now().strftime('%Y%m%d'))
            ds.StudyTime = self.study_info.get('StudyTime', datetime.now().strftime('%H%M%S'))
            ds.ReferringPhysicianName = self.study_info.get('ReferringPhysicianName', '')
            ds.StudyID = self.study_info.get('StudyID', study_id)
            ds.AccessionNumber = self.study_info.get('AccessionNumber', '')
            
            # General Series Module
            ds.SeriesInstanceUID = series_uid
            ds.SeriesNumber = 998  # High number to appear near the end
            ds.SeriesDate = datetime.now().strftime('%Y%m%d')
            ds.SeriesTime = datetime.now().strftime('%H%M%S')
            ds.Modality = 'OT'  # Other
            ds.SeriesDescription = series_description
            
            # General Equipment Module
            ds.Manufacturer = 'Spine Analyzer'
            ds.InstitutionName = self.study_info.get('InstitutionName', '')
            ds.StationName = 'AI Analysis Station'
            
            # SC Equipment Module
            ds.ConversionType = 'WSD'  # Workstation
            ds.SecondaryCaptureDeviceID = 'SpineAnalyzer'
            ds.SecondaryCaptureDeviceManufacturer = 'AI System'
            ds.SecondaryCaptureDeviceManufacturerModelName = 'v1.0'
            ds.SecondaryCaptureDeviceSoftwareVersions = '1.0.0'
            
            # General Image Module
            ds.InstanceNumber = idx + 1
            ds.ImageType = ['DERIVED', 'SECONDARY']
            ds.ContentDate = datetime.now().strftime('%Y%m%d')
            ds.ContentTime = datetime.now().strftime('%H%M%S')
            
            # Image Pixel Module
            if len(image.shape) == 2:
                # Grayscale image
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = 'MONOCHROME2'
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
                pixel_data = image.astype(np.uint8)
            else:
                # Color image (BGR to RGB conversion if needed)
                ds.SamplesPerPixel = 3
                ds.PhotometricInterpretation = 'RGB'
                ds.PlanarConfiguration = 0
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
                # Convert BGR to RGB if needed
                if image.shape[2] == 3:
                    pixel_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
                else:
                    pixel_data = image.astype(np.uint8)
            
            ds.Rows = image.shape[0]
            ds.Columns = image.shape[1]
            ds.PixelRepresentation = 0  # unsigned
            ds.PixelData = pixel_data.tobytes()
            
            # SOP Common Module
            ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture
            ds.SOPInstanceUID = generate_uid()
            
            # Set transfer syntax
            ds.is_implicit_VR = False
            ds.is_little_endian = True
            
            # Create file meta information
            file_meta = Dataset()
            file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
            file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            file_meta.ImplementationClassUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            
            # Create FileDataset
            filename = f"SC_{idx:04d}.dcm"
            file_ds = FileDataset(filename, ds, file_meta=file_meta, preamble=b"\0" * 128)
            
            # Save to bytes
            with io.BytesIO() as buffer:
                file_ds.save_as(buffer, write_like_original=False)
                dicom_files.append(buffer.getvalue())
        
        return dicom_files
    
    def upload_to_orthanc(self, dicom_data: bytes, study_id: str) -> Dict[str, Any]:
        """
        Upload DICOM file to Orthanc
        
        Args:
            dicom_data: DICOM file as bytes
            study_id: Target study ID in Orthanc
            
        Returns:
            Response from Orthanc
        """
        try:
            # Upload DICOM instance
            url = f"{self.orthanc_url}/instances"
            response = requests.post(
                url,
                data=dicom_data,
                auth=self.orthanc_auth,
                headers={'Content-Type': 'application/dicom'}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully uploaded DICOM to study {study_id}: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to upload DICOM to Orthanc: {e}")
            raise
    
    def get_study_metadata(self, study_id: str) -> Dict[str, Any]:
        """
        Fetch study metadata from Orthanc
        
        Args:
            study_id: Orthanc study ID
            
        Returns:
            Study metadata dictionary
        """
        try:
            resource_id = self._get_resource_uid("Study", study_id)
            url = f"{self.orthanc_url}/studies/{resource_id}"
            response = requests.get(url, auth=self.orthanc_auth)
            response.raise_for_status()
            
            study_data = response.json()
            
            # Extract main DICOM tags
            main_tags = study_data.get('MainDicomTags', {})
            patient_tags = study_data.get('PatientMainDicomTags', {})
            
            return {
                'StudyInstanceUID': main_tags.get('StudyInstanceUID'),
                'StudyDate': main_tags.get('StudyDate'),
                'StudyTime': main_tags.get('StudyTime'),
                'StudyID': main_tags.get('StudyID'),
                'AccessionNumber': main_tags.get('AccessionNumber'),
                'ReferringPhysicianName': main_tags.get('ReferringPhysicianName'),
                'InstitutionName': main_tags.get('InstitutionName'),
                'PatientName': patient_tags.get('PatientName'),
                'PatientID': patient_tags.get('PatientID'),
                'PatientBirthDate': patient_tags.get('PatientBirthDate'),
                'PatientSex': patient_tags.get('PatientSex')
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch study metadata from Orthanc for study {study_id}: {e}")
            return {}


def send_reports_to_orthanc(
    study_id: str,
    grading_results: Dict,
    pathology_measurements: Dict,
    segmentation_images: Any,
    segmentation_nifti=None,
    reference_image_nifti=None
) -> Dict[str, Any]:
    """
    Main function to create and send all reports to Orthanc
    
    Args:
        study_id: Orthanc study ID
        grading_results: Grading analysis results
        pathology_measurements: Pathology measurements
        segmentation_images: List of segmentation overlay images
        
    Returns:
        Dictionary with upload results
    """
    try:
        generator = DICOMReportGenerator()
        
        study_info = generator.get_study_metadata(study_id)
        logger.info(f"study_info: {study_info}")
        generator.study_info = study_info
        
        results = {
            'study_id': study_id,
            'sr_created': False,
            'sr_uploaded': False,
            'seg_created': False,
            'seg_uploaded': False,
            'sc_created': False,
            'sc_uploaded': False,
            'sc_count': 0,
            'errors': []
        }

        sr_data = None
        try:
            sr_data = generator.create_structured_report(
                grading_results,
                pathology_measurements,
                study_id
            )
            results['sr_created'] = True
            logger.info(f"SR created successfully for study {study_id}")
        except Exception as e:
            error_msg = f"SR creation failed: {str(e)}"
            logger.error(f"Failed to create SR for study {study_id}: {e}")
            results['errors'].append(error_msg)

        if sr_data:
            try:
                sr_result = generator.upload_to_orthanc(sr_data, study_id)
                results['sr_uploaded'] = True
                results['sr_instance'] = sr_result.get('ID')
                logger.info(f"SR uploaded successfully for study {study_id}")
            except Exception as e:
                error_msg = f"SR upload failed: {str(e)}"
                logger.error(f"Failed to upload SR for study {study_id}: {e}")
                results['errors'].append(error_msg)

        # Create and upload DICOM Segmentation Object if segmentation data is provided
        if segmentation_nifti is not None and reference_image_nifti is not None:
            try:
                # Extract segmentation data
                seg_data = np.asanyarray(segmentation_nifti.dataobj).astype(np.uint8)
                
                seg_dicom_data = generator.create_enhanced_segmentation_object(
                    seg_data,
                    reference_image_nifti,
                    study_id,
                    grading_results,
                    "AI Spine Segmentation"
                )
                results['seg_created'] = True
                logger.info(f"DICOM Segmentation Object created successfully for study {study_id}")
                
                try:
                    seg_result = generator.upload_to_orthanc(seg_dicom_data, study_id)
                    results['seg_uploaded'] = True
                    results['seg_instance'] = seg_result.get('ID')
                    logger.info(f"DICOM Segmentation Object uploaded successfully for study {study_id}")
                except Exception as e:
                    error_msg = f"Segmentation upload failed: {str(e)}"
                    logger.error(f"Failed to upload Segmentation Object for study {study_id}: {e}")
                    results['errors'].append(error_msg)
                    
            except Exception as e:
                error_msg = f"Segmentation creation failed: {str(e)}"
                logger.error(f"Failed to create Segmentation Object for study {study_id}: {e}")
                results['errors'].append(error_msg)

        # Handle one or multiple overlay variants
        total_uploaded = 0
        if segmentation_images:
            try:
                variant_map: List[Tuple[str, List[np.ndarray]]] = []

                if isinstance(segmentation_images, dict):
                    # Варианты переданы словарём
                    for key in ['variant_a', 'variant_b', 'variant_c']:
                        if key in segmentation_images and segmentation_images[key]:
                            desc = {
                                'variant_a': 'Spine Overlay A - Contours (Verts/Disks/Pathologies)',
                                'variant_b': 'Spine Overlay B - Filled (Verts/Disks/Pathologies)',
                                'variant_c': 'Spine Overlay C - Focused (Modic/Spondy + Pathologies)'
                            }.get(key, f'Spine Overlay {key}')
                            variant_map.append((desc, segmentation_images[key]))

                elif isinstance(segmentation_images, list) and segmentation_images and isinstance(
                        segmentation_images[0], list):
                    # Варианты переданы списком списков [ [A], [B], [C] ]
                    variant_labels = [
                        "Spine Overlay A - Contours (Verts/Disks/Pathologies)",
                        "Spine Overlay B - Filled (Verts/Disks/Pathologies)",
                        "Spine Overlay C - Focused (Modic/Spondy + Pathologies)",
                    ]
                    for idx, imgs in enumerate(segmentation_images):
                        if imgs:
                            desc = variant_labels[idx] if idx < len(
                                variant_labels) else f"Spine Overlay Variant {idx + 1}"
                            variant_map.append((desc, imgs))

                else:
                    # Старый случай — просто список картинок
                    variant_map.append(("Spine Segmentation with Pathology Overlay", segmentation_images))

                for desc, imgs in variant_map:
                    if not imgs:
                        continue
                    sc_files = generator.create_secondary_capture(imgs, study_id, desc)
                    results.setdefault('sc_series', []).append({'description': desc, 'count': len(sc_files)})

                    uploaded_count = 0
                    for i, sc_data in enumerate(sc_files):
                        try:
                            generator.upload_to_orthanc(sc_data, study_id)
                            uploaded_count += 1
                        except Exception as e:
                            error_msg = f"SC upload failed ({desc}) for image {i+1}: {str(e)}"
                            logger.error(f"Failed to upload SC image {i+1} for study {study_id} [{desc}]: {e}")
                            results['errors'].append(error_msg)

                    total_uploaded += uploaded_count

                results['sc_count'] = total_uploaded
                results['sc_created'] = total_uploaded > 0
                results['sc_uploaded'] = total_uploaded > 0
                if total_uploaded > 0:
                    logger.info(f"Uploaded {total_uploaded} SC images across {len(variant_map)} series for study {study_id}")
            except Exception as e:
                error_msg = f"SC creation/upload failed: {str(e)}"
                logger.error(f"Failed to create/upload SC for study {study_id}: {e}")
                results['errors'].append(error_msg)

        return results

    except Exception as e:
        logger.error(f"Unexpected error in send_reports_to_orthanc for study {study_id}: {e}")
        return {
            'study_id': study_id,
            'sr_created': False,
            'sr_uploaded': False,
            'seg_created': False,
            'seg_uploaded': False,
            'sc_created': False,
            'sc_uploaded': False,
            'errors': [f"Unexpected error: {str(e)}"]
        }