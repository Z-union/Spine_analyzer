from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def create_disk_report(disk_data, patient_info, seg_sagittal=None, seg_axial=None, output_filename=None):
    """
    Generates a DOCX report with a table from a list of disk data dictionaries,
    followed by two segmentation images (sagittal and axial views).
    Margins set to 0.5 cm, font: Times New Roman, 12 pt.

    :param disk_data: List of dicts with disk information
    :param patient_info: Dict with keys: 'patient_name', 'patient_age', 'study_date', 'modality', 'series_description'
    :param seg_sagittal: NumPy array for sagittal segmentation image (2D or 3D)
    :param seg_axial: NumPy array for axial segmentation image (2D or 3D)
    :param output_filename: Output .docx filename (without path)
    :return: Full path to saved file
    """
    if not output_filename:
        output_filename = "mri_disk_report.docx"

    output_path = os.path.join(REPORTS_DIR, output_filename)

    doc = Document()
    
    sections = doc.sections
    for section in sections:
        section.top_margin = Cm(0.5)
        section.bottom_margin = Cm(0.5)
        section.left_margin = Cm(0.5)
        section.right_margin = Cm(0.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('Заключение по результатам МРТ поясничного отдела позвоночника')
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.name = 'Times New Roman'
    run.font.color.rgb = RGBColor(0, 0, 0)

    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)
    font_element = style._element.rPr
    font_element.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    # Add patient info block
    patient_lines = [
        f"Пациент: {patient_info.get('patient_name', 'N/A')}",
        f"Возраст: {patient_info.get('patient_age', 'N/A')}",
        f"Дата исследования: {patient_info.get('study_date', 'N/A')}",
        f"Методика: {patient_info.get('modality', 'MRI')}",
        f"Уровни сканирования: {patient_info.get('series_description', 'Lumbar Spine')}"
    ]

    for line in patient_lines:
        p = doc.add_paragraph(line)
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            run.font.color.rgb = RGBColor(0, 0, 0)

    # Add spacing
    doc.add_paragraph()


    # Add spacing after title
    doc.add_paragraph()

    if not disk_data:
        doc.add_paragraph("No data available.")
        doc.save(output_path)
        return output_path

    headers = [
        'Диск', 'Modic', 'Верхняя замыкательная пластина', 'Нижняя замыкательная пластина', 'Спондилолизтез',
        'Грыжа диска', 'Сужение диска', 'Протрузия диска', 'Степень Pfirrmann',
        'Грыжа', 'Объем грыжи (mm³)', 'Hernia Max Protrusion (mm)',
        'Spondy Detected', 'Spondy Displacement (mm)', 'Spondy Displacement (%)',
        'Spondy Grade'
    ]

    key_mapping = [
        'disk_label', 'Modic', 'UP endplate', 'LOW endplate', 'Spondylolisthesis',
        'Disc herniation', 'Disc narrowing', 'Disc bulging', 'Pfirrmann grade',
        'hernia_detected', 'hernia_volume_mm3', 'hernia_max_protrusion_mm',
        'spondy_detected', 'spondy_displacement_mm', 'spondy_displacement_percentage',
        'spondy_grade'
    ]

    disk_mapping = {
        63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
        71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5',
        76: 'T5-T6', 77: 'T6-T7', 78: 'T7-T8', 79: 'T8-T9', 80: 'T9-T10',
        81: 'T10-T11', 82: 'T11-T12', 91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3',
        94: 'L3-L4', 95: 'L4-L5', 96: 'L5-S1', 100: 'L5-S1'
    }

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'

    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        hdr_cells[i].text = header
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(0, 0, 0)
                run.font.bold = True

    for item in disk_data:
        row_cells = table.add_row().cells
        for i, key in enumerate(key_mapping):
            raw_value = item.get(key, None)
            # Only map disk_label using disk_mapping if key matches and value exists in mapping
            if key == 'disk_label' and raw_value in disk_mapping:
                value = disk_mapping[raw_value]
            else:
                value = raw_value

            cell = row_cells[i]
            # Format value to string, handle lists
            if value is None:
                text = ""
            elif isinstance(value, list):
                text = str(value).replace("[", "").replace("]", "")
            else:
                text = str(value)

            cell.text = text
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                    run.font.color.rgb = RGBColor(0, 0, 0)

    # Add spacing after table
    doc.add_paragraph()

    # === Add Sagittal Segmentation Image (Рис. 27) ===
    if seg_sagittal is not None:
        # Handle channel-first format (3, H, W) → transpose to (H, W, 3)
        if seg_sagittal.ndim == 3 and seg_sagittal.shape[0] == 3:
            seg_sagittal = np.transpose(seg_sagittal, (1, 2, 0))
        # Rotate 90 degrees clockwise
        seg_sagittal = np.rot90(seg_sagittal, k=-1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.figure(figsize=(6, 4))
            plt.imshow(seg_sagittal)
            plt.axis('off')
            plt.title("Срез 27", fontsize=12, fontname='Times New Roman', fontweight='bold')
            plt.tight_layout()
            plt.savefig(tmpfile.name, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

            doc.add_paragraph()  # Spacing
            doc.add_picture(tmpfile.name, width=Inches(6))
            # Optional: Add caption below image
            caption = doc.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption.add_run("Срез 27")
            caption_run.font.name = 'Times New Roman'
            caption_run.font.size = Pt(12)
            os.unlink(tmpfile.name)  # Clean up

    # === Add Axial Segmentation Image (Рис. 28) ===
    if seg_axial is not None:
        # Handle channel-first format (3, H, W) → transpose to (H, W, 3)
        if seg_axial.ndim == 3 and seg_axial.shape[0] == 3:
            seg_axial = np.transpose(seg_axial, (1, 2, 0))
        # Rotate 90 degrees clockwise
        seg_axial = np.rot90(seg_axial, k=-1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.figure(figsize=(6, 4))
            plt.imshow(seg_axial)
            plt.axis('off')
            plt.title("Срез 28", fontsize=12, fontname='Times New Roman', fontweight='bold')
            plt.tight_layout()
            plt.savefig(tmpfile.name, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

            doc.add_paragraph()  # Spacing
            doc.add_picture(tmpfile.name, width=Inches(6))
            # Optional: Add caption below image
            caption = doc.add_paragraph()
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            caption_run = caption.add_run("Срез 28")
            caption_run.font.name = 'Times New Roman'
            caption_run.font.size = Pt(12)
            os.unlink(tmpfile.name)  # Clean up

    doc.save(output_path)
    return output_path