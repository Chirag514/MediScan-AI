"""
report/pdf_export.py
Generate a downloadable PDF from the structured report.
"""

from fpdf import FPDF
from PIL import Image
import io
import tempfile
import os
from datetime import datetime


def sanitize(text: str) -> str:
    """
    Replace all Unicode characters unsupported by fpdf2's built-in
    Helvetica font with ASCII equivalents. Catches both hardcoded
    strings and LLM-generated output.
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2014": "-",    # em dash —
        "\u2013": "-",    # en dash –
        "\u2011": "-",    # non-breaking hyphen ‑
        "\u2010": "-",    # hyphen ‐
        "\u2012": "-",    # figure dash ‒
        "\u2015": "-",    # horizontal bar ―
        "\u2019": "'",    # right single quote '
        "\u2018": "'",    # left single quote '
        "\u201c": '"',    # left double quote "
        "\u201d": '"',    # right double quote "
        "\u2022": "*",    # bullet •
        "\u2023": "*",    # triangular bullet ‣
        "\u25cf": "*",    # black circle ●
        "\u2026": "...",  # ellipsis …
        "\u00b0": " deg", # degree °
        "\u00b1": "+/-",  # plus-minus ±
        "\u00d7": "x",    # multiplication ×
        "\u2264": "<=",   # less than or equal ≤
        "\u2265": ">=",   # greater than or equal ≥
        "\u00e9": "e",    # é
        "\u00e8": "e",    # è
        "\u00ea": "e",    # ê
        "\u00fc": "u",    # ü
        "\u00e4": "a",    # ä
        "\u00f6": "o",    # ö
        "\u2032": "'",    # prime ′
        "\u00b2": "2",    # superscript 2 ²
        "\u00b3": "3",    # superscript 3 ³
        "\u03b1": "alpha",# α
        "\u03b2": "beta", # β
        "\u03b3": "gamma",# γ
        "\u2248": "~",    # approximately equal ≈
        "\u00a0": " ",    # non-breaking space
        "\u2212": "-",    # minus sign −
        "\u00ad": "-",    # soft hyphen ­
        "\u2039": "<",    # single left angle «
        "\u203a": ">",    # single right angle »
        "\u00ab": "<<",   # left double angle «
        "\u00bb": ">>",   # right double angle »
        "\u2588": "|",    # full block █
        "\u26a0": "!",    # warning sign ⚠
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # catch-all: drop anything still outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


class MedicalReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 80, 160)
        self.cell(0, 10, "AI Medical Image Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, sanitize(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  AI-GENERATED - NOT A CLINICAL DIAGNOSIS"),
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_draw_color(30, 80, 160)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, sanitize(
            f"Page {self.page_no()} | AI Preliminary Report"), align="C")


def generate_pdf(
    report: dict,
    stats: dict,
    original_image: Image.Image,
    overlay_image: Image.Image,
    scan_type: str,
) -> bytes:
    """
    Build a PDF report and return as bytes for Streamlit download.
    """
    pdf = MedicalReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Scan Info ────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 8, "Scan Information", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    scan_labels = {
        "skin_lesion": "Dermatoscopy / Skin Lesion",
        "chest_xray": "Chest X-Ray",
        "ultrasound": "Breast Ultrasound",
    }
    pdf.cell(0, 6, f"Scan Type: {scan_labels.get(scan_type, 'Medical Scan')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Image Dimensions: {stats['image_size'][0]} x {stats['image_size'][1]} px", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Images side by side ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 8, "Imaging", new_x="LMARGIN", new_y="NEXT")

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = os.path.join(tmpdir, "original.jpg")
        overlay_path = os.path.join(tmpdir, "overlay.jpg")
        original_image.convert("RGB").save(orig_path, "JPEG")
        overlay_image.convert("RGB").save(overlay_path, "JPEG")

        img_y = pdf.get_y()

        # Left image + caption
        pdf.image(orig_path, x=10, y=img_y, w=90, h=70)
        pdf.set_xy(10, img_y + 72)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(90, 5, "Original Image", align="C")

        # Right image + caption
        pdf.image(overlay_path, x=110, y=img_y, w=90, h=70)
        pdf.set_xy(110, img_y + 72)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(90, 5, "Segmentation Overlay", align="C")

        # Move cursor below both images
        pdf.set_xy(10, img_y + 80)
        pdf.ln(4)

    # ── Quantitative Stats ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 8, "Quantitative Measurements", new_x="LMARGIN", new_y="NEXT")

    roi_label = {
    "skin_lesion": "Lesion Area",
    "chest_xray":  "Lung Field Area",
    "ultrasound":  "Mass Area",
    }.get(scan_type, "ROI Area")
    
    rows = [
        (roi_label,            f"{stats['area_pct']}% of image"),
        ("Location", stats['location']),
        ("Mean Intensity", f"{stats['mean_intensity']} / 255"),
        ("Contrast Ratio", str(stats['contrast_ratio'])),
        ("Shape Irregularity", str(stats['irregularity'])),
        ("Solidity", str(stats['solidity'])),
        ("Bbox Aspect Ratio", str(stats['bbox_wh_ratio'])),
    ]

    pdf.set_font("Helvetica", "", 10)
    pdf.set_fill_color(240, 245, 255)
    for i, (label, value) in enumerate(rows):
        pdf.set_text_color(40, 40, 40)
        fill = (i % 2 == 0)
        pdf.cell(80, 7, label, border=0, fill=fill)
        pdf.cell(100, 7, value, border=0, fill=fill, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Report Sections ───────────────────────────────────────────────────────
    sections = [
        ("Findings",        sanitize(report.get("findings", "N/A"))),
        ("Impression",      sanitize(report.get("impression", "N/A"))),
        ("Recommendations", sanitize(report.get("recommendations", "N/A"))),
    ]

    for title, content in sections:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 80, 160)
        pdf.cell(0, 8, sanitize(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, content)
        pdf.ln(3)

    # Differential
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 8, "Differential Considerations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    for item in report.get("differential", []):
        pdf.cell(0, 6, sanitize(f"  - {item}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Confidence
    confidence = sanitize(report.get("confidence", "N/A"))
    color_map = {"High": (0, 150, 0), "Moderate": (200, 130, 0), "Low": (200, 0, 0)}
    r, g, b = color_map.get(confidence, (80, 80, 80))
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(pdf.get_string_width("AI Confidence: "), 8, "AI Confidence: ")
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 8, confidence, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Disclaimer box
    pdf.set_fill_color(255, 245, 220)
    pdf.set_draw_color(200, 150, 0)
    pdf.set_line_width(0.3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(150, 100, 0)
    pdf.multi_cell(0, 6, sanitize(f"! DISCLAIMER: {report.get('disclaimer', '')}"), border=1, fill=True)

    return bytes(pdf.output())