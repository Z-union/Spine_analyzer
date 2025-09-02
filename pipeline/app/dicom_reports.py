"""
Module for creating and uploading DICOM SR (Structured Report) and SC (Secondary Capture) to Orthanc
"""
import logging
import io
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import cv2
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import requests

from .config import settings

# Используем единый логгер из main
logger = logging.getLogger("dicom-pipeline")


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
        
        # Add title
        content_items.append({
            'ValueType': 'CONTAINER',
            'ConceptNameCodeSequence': [{
                'CodeValue': '121070',
                'CodingSchemeDesignator': 'DCM',
                'CodeMeaning': 'Findings'
            }],
            'ContinuityOfContent': 'SEPARATE',
            'ContentSequence': []
        })
        
        # Add grading results
        if grading_results:
            for disk_label, result in grading_results.items():
                if 'predictions' in result:
                    level_name = result.get('level_name', f'Disk_{disk_label}')
                    disk_content = []
                    
                    # Add disk level identifier
                    disk_content.append({
                        'ValueType': 'TEXT',
                        'ConceptNameCodeSequence': [{
                            'CodeValue': '121071',
                            'CodingSchemeDesignator': 'DCM',
                            'CodeMeaning': 'Finding'
                        }],
                        'TextValue': f'Disk Level: {level_name}'
                    })
                    
                    # Add grading scores
                    for category, value in result['predictions'].items():
                        disk_content.append({
                            'ValueType': 'NUM',
                            'ConceptNameCodeSequence': [{
                                'CodeValue': '121072',
                                'CodingSchemeDesignator': 'DCM',
                                'CodeMeaning': category
                            }],
                            'MeasuredValueSequence': [{
                                'NumericValue': str(value),
                                'MeasurementUnitsCodeSequence': [{
                                    'CodeValue': '1',
                                    'CodingSchemeDesignator': 'UCUM',
                                    'CodeMeaning': 'grade'
                                }]
                            }]
                        })
                    
                    # Add pathology measurements if available
                    if disk_label in pathology_measurements:
                        measurements = pathology_measurements[disk_label]
                        
                        # Herniation measurements
                        if measurements.get('herniation'):
                            hernia = measurements['herniation']
                            if hernia.get('detected'):
                                disk_content.append({
                                    'ValueType': 'NUM',
                                    'ConceptNameCodeSequence': [{
                                        'CodeValue': '121073',
                                        'CodingSchemeDesignator': 'DCM',
                                        'CodeMeaning': 'Herniation Volume'
                                    }],
                                    'MeasuredValueSequence': [{
                                        'NumericValue': str(hernia.get('volume_mm3', 0)),
                                        'MeasurementUnitsCodeSequence': [{
                                            'CodeValue': 'mm3',
                                            'CodingSchemeDesignator': 'UCUM',
                                            'CodeMeaning': 'cubic millimeter'
                                        }]
                                    }]
                                })
                                
                                disk_content.append({
                                    'ValueType': 'NUM',
                                    'ConceptNameCodeSequence': [{
                                        'CodeValue': '121074',
                                        'CodingSchemeDesignator': 'DCM',
                                        'CodeMeaning': 'Herniation Protrusion'
                                    }],
                                    'MeasuredValueSequence': [{
                                        'NumericValue': str(hernia.get('max_protrusion_mm', 0)),
                                        'MeasurementUnitsCodeSequence': [{
                                            'CodeValue': 'mm',
                                            'CodingSchemeDesignator': 'UCUM',
                                            'CodeMeaning': 'millimeter'
                                        }]
                                    }]
                                })
                        
                        # Spondylolisthesis measurements
                        if measurements.get('spondylolisthesis'):
                            spondy = measurements['spondylolisthesis']
                            if spondy.get('detected'):
                                disk_content.append({
                                    'ValueType': 'NUM',
                                    'ConceptNameCodeSequence': [{
                                        'CodeValue': '121075',
                                        'CodingSchemeDesignator': 'DCM',
                                        'CodeMeaning': 'Spondylolisthesis Displacement'
                                    }],
                                    'MeasuredValueSequence': [{
                                        'NumericValue': str(spondy.get('displacement_mm', 0)),
                                        'MeasurementUnitsCodeSequence': [{
                                            'CodeValue': 'mm',
                                            'CodingSchemeDesignator': 'UCUM',
                                            'CodeMeaning': 'millimeter'
                                        }]
                                    }]
                                })
                                
                                disk_content.append({
                                    'ValueType': 'TEXT',
                                    'ConceptNameCodeSequence': [{
                                        'CodeValue': '121076',
                                        'CodingSchemeDesignator': 'DCM',
                                        'CodeMeaning': 'Spondylolisthesis Grade'
                                    }],
                                    'TextValue': f"Grade {spondy.get('grade', 'Unknown')}"
                                })
                    
                    # Add disk container to main content
                    content_items[0]['ContentSequence'].append({
                        'ValueType': 'CONTAINER',
                        'ConceptNameCodeSequence': [{
                            'CodeValue': '121077',
                            'CodingSchemeDesignator': 'DCM',
                            'CodeMeaning': f'Disk Analysis: {level_name}'
                        }],
                        'ContinuityOfContent': 'SEPARATE',
                        'ContentSequence': disk_content
                    })
        
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
            url = f"{self.orthanc_url}/studies/{study_id}"
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
            logger.error(f"Failed to fetch study metadata from Orthanc: {e}")
            return {}


def send_reports_to_orthanc(study_id: str,
                           grading_results: Dict,
                           pathology_measurements: Dict,
                           segmentation_images: List[np.ndarray]) -> Dict[str, Any]:
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
        
        # Get study metadata from Orthanc
        study_info = generator.get_study_metadata(study_id)
        generator.study_info = study_info
        
        results = {
            'study_id': study_id,
            'sr_uploaded': False,
            'sc_uploaded': False,
            'sc_count': 0,
            'errors': []
        }
        
        # Create and upload Structured Report
        try:
            sr_data = generator.create_structured_report(
                grading_results, 
                pathology_measurements,
                study_id
            )
            sr_result = generator.upload_to_orthanc(sr_data, study_id)
            results['sr_uploaded'] = True
            results['sr_instance'] = sr_result.get('ID')
            logger.info(f"SR uploaded successfully for study {study_id}")
        except Exception as e:
            logger.error(f"Failed to create/upload SR for study {study_id}: {e}")
            results['errors'].append(f"SR upload failed: {str(e)}")
        
        # Create and upload Secondary Captures
        if segmentation_images:
            try:
                sc_files = generator.create_secondary_capture(
                    segmentation_images,
                    study_id,
                    "Spine Segmentation with Pathology Overlay"
                )
                
                for sc_data in sc_files:
                    sc_result = generator.upload_to_orthanc(sc_data, study_id)
                    results['sc_count'] += 1
                
                results['sc_uploaded'] = True
                logger.info(f"Uploaded {results['sc_count']} SC images for study {study_id}")
                
            except Exception as e:
                logger.error(f"Failed to create/upload SC for study {study_id}: {e}")
                results['errors'].append(f"SC upload failed: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to send reports to Orthanc for study {study_id}: {e}")
        return {
            'study_id': study_id,
            'sr_uploaded': False,
            'sc_uploaded': False,
            'errors': [str(e)]
        }