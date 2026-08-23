"""
report/generator.py
Structured clinical report generation via Groq (GPT-OSS-120B).
Takes segmentation stats → returns structured report dict.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a medical imaging AI assistant that generates structured preliminary 
reports based on image segmentation analysis data. You are NOT a doctor and your output is 
NOT a clinical diagnosis. Always include appropriate disclaimers.

You must respond ONLY with a valid JSON object — no preamble, no markdown, no explanation.
The JSON must have exactly these keys:
{
  "findings": "string — detailed description of the segmented region",
  "impression": "string — overall preliminary impression (1-2 sentences)",
  "confidence": "string — one of: Low / Moderate / High",
  "differential": ["string", "string"],  // 2-3 possible conditions to consider
  "recommendations": "string — suggested follow-up actions",
  "disclaimer": "string — standard medical disclaimer"
}"""


def build_prompt(stats: dict, interpretations: dict, scan_type: str) -> str:
    scan_labels = {
        "skin_lesion": "Dermatoscopy / Skin Lesion Image",
        "chest_xray":  "Chest X-Ray",
        "ultrasound":  "Breast Ultrasound",
    }
    # scan-specific ROI terminology
    roi_labels = {
        "skin_lesion": "Lesion",
        "chest_xray":  "Lung field",
        "ultrasound":  "Mass/Nodule",
    }
    scan_label = scan_labels.get(scan_type, "Medical Scan")
    roi_label  = roi_labels.get(scan_type, "Region of interest")

    return f"""Generate a structured preliminary medical report for the following segmentation analysis.

SCAN TYPE: {scan_label}

QUANTITATIVE MEASUREMENTS:
- {roi_label} area: {stats['area_pct']}% of total image area ({interpretations['size']} {roi_label.lower()})
- Location: {stats['location']} region of image
- Mean pixel intensity: {stats['mean_intensity']} / 255 ({interpretations['intensity']})
- Intensity standard deviation: {stats['std_intensity']} (texture variation)
- Contrast ratio ({roi_label.lower()} vs background): {stats['contrast_ratio']}
- Shape irregularity index: {stats['irregularity']} ({interpretations['shape']})
- Solidity (convexity): {stats['solidity']} ({interpretations['border']})
- Bounding box aspect ratio (W/H): {stats['bbox_wh_ratio']}
- Image dimensions: {stats['image_size'][0]}x{stats['image_size'][1]} px

INTERPRETATION SUMMARY:
- Size: {interpretations['size']}
- Shape: {interpretations['shape']}
- Intensity: {interpretations['intensity']}
- Border: {interpretations['border']}

Based on this analysis, generate the structured JSON report.
Use terminology appropriate for {scan_label} — avoid using 'lesion' 
for chest X-ray (use 'lung field', 'opacity', 'consolidation') 
and for ultrasound use 'mass', 'nodule', or 'lesion' only if appropriate."""


def generate_report(stats: dict, interpretations: dict, scan_type: str) -> dict:
    """
    Call Groq API and return parsed report dict.
    Falls back gracefully on API/parse errors.
    """
    prompt = build_prompt(stats, interpretations, scan_type)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,      # low temp for consistent structured output
            max_tokens=800,
            response_format={"type": "json_object"},  # enforce JSON output
        )

        raw = response.choices[0].message.content
        report = json.loads(raw)

        # validate required keys exist
        required = ["findings", "impression", "confidence",
                    "differential", "recommendations", "disclaimer"]
        for key in required:
            if key not in report:
                report[key] = "Not available"

        return {"success": True, "report": report}

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"JSON parse error: {e}",
            "report": _fallback_report(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "report": _fallback_report(),
        }


def _fallback_report() -> dict:
    return {
        "findings": "Unable to generate findings due to an error.",
        "impression": "Report generation failed. Please try again.",
        "confidence": "Low",
        "differential": ["Analysis unavailable"],
        "recommendations": "Please consult a qualified medical professional.",
        "disclaimer": "This is an AI-generated preliminary analysis and is NOT a clinical diagnosis. Always consult a licensed physician.",
    }
