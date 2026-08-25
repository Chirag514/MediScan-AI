"""
report/generator.py
Structured clinical report generation via Groq with Gemini fallback.
Takes segmentation stats → returns structured report dict.
"""

import json
import os
from functools import lru_cache
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None

from json_repair import repair_json

LIST_KEYS = {"differential"}

def _parse_report(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        report = json.loads(repair_json(raw))
    if not isinstance(report, dict):
        raise ValueError("Model response was not a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in report]
    for key in REQUIRED_KEYS:
        if key not in report:
            report[key] = [] if key in LIST_KEYS else "Not available"
        elif key in LIST_KEYS and not isinstance(report[key], list):
            report[key] = [str(report[key])]
    if len(missing) > 2:
        raise ValueError(f"Response too incomplete — missing keys: {missing}")
    return report


def _model_list(variable: str, defaults: list[str]) -> list[str]:
    configured = os.getenv(variable)
    return [model.strip() for model in configured.split(",") if model.strip()] if configured else defaults


GROQ_MODELS = _model_list(
    "GROQ_MODELS",
    ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
)
GEMINI_MODELS = _model_list(
    "GEMINI_MODELS",
    ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
)

LOCAL_LLM_MODEL = os.getenv(
    "LOCAL_LLM_MODEL",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)

# Quality-ranked route with providers interleaved. The lighter models preserve
# request and token quota when a stronger model is unavailable or rate-limited.
MODEL_ROUTE = []
for index in range(max(len(GROQ_MODELS), len(GEMINI_MODELS))):
    if index < len(GEMINI_MODELS):
        MODEL_ROUTE.append(("Gemini", GEMINI_MODELS[index]))
    if index < len(GROQ_MODELS):
        MODEL_ROUTE.append(("Groq", GROQ_MODELS[index]))

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

STANDARD_DISCLAIMER = (
    "This is an AI-generated preliminary analysis and is NOT a clinical "
    "diagnosis. Always consult a licensed physician."
)

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


REQUIRED_KEYS = [
    "findings", "impression", "confidence", "differential",
    "recommendations", "disclaimer",
]


@lru_cache(maxsize=1)
def _load_local_llm(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        # SmolLM2 doesn't define a pad token by default — without this,
        # generate() can fail deep inside transformers with a bare
        # AssertionError that has no message.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def _generate_with_local_llm(prompt: str, model_name: str) -> dict:
    if os.getenv("LOCAL_LLM_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("LOCAL_LLM_ENABLED is disabled")

    import torch

    tokenizer, model = _load_local_llm(model_name)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
    else:
        input_ids = tokenizer(
            f"{SYSTEM_PROMPT}\n\n{prompt}",
            return_tensors="pt",
        ).input_ids

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if not text.strip():
        raise RuntimeError("Local model produced an empty response")
    report = _parse_report(text)
    report["disclaimer"] = STANDARD_DISCLAIMER
    return report

def _generate_with_groq(prompt: str, model: str) -> dict:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY is not configured")

    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
        response_format={"type": "json_object"},
    )
    return _parse_report(response.choices[0].message.content)


def _generate_with_gemini(prompt: str, model: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    try:
        import google.genai as genai
        from google.genai import types
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Gemini SDK is unavailable in the active environment; install google-genai."
        ) from error

    gemini_client = genai.Client(api_key=api_key)
    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=2048,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return _parse_report(response.text)

def generate_report(stats: dict, interpretations: dict, scan_type: str) -> dict:
    """Try the quality-ranked multi-provider route, then use a local fallback."""
    prompt = build_prompt(stats, interpretations, scan_type)
    errors = []

    generators = {"Groq": _generate_with_groq, "Gemini": _generate_with_gemini}
    attempts = [(provider, model, generators[provider]) for provider, model in MODEL_ROUTE]
    attempts.append(("Local", LOCAL_LLM_MODEL, _generate_with_local_llm))

    for provider, model, generator in attempts:
        try:
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "report": generator(prompt, model),
            }
        except Exception as error:
            errors.append(f"{provider} ({model}): {error}")

    return {
        "success": False,
        "error": "; ".join(errors),
        "report": _fallback_report(),
    }


def _fallback_report() -> dict:
    return {
        "findings": "Unable to generate findings due to an error.",
        "impression": "Report generation failed. Please try again.",
        "confidence": "Low",
        "differential": ["Analysis unavailable"],
        "recommendations": "Please consult a qualified medical professional.",
        "disclaimer": STANDARD_DISCLAIMER,
        }
