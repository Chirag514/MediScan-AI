# 🩺 MediScan AI — Medical Image Segmentation + Report Generation

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-MediScan%20AI-FF4B4B?style=for-the-badge)](https://mediscanai-app.streamlit.app/)

AI-powered pipeline that segments medical scans using SegFormer-b2 (fine-tuned) and generates structured clinical reports through a multi-provider LLM fallback chain (Gemini, Groq, and Local LLM).

## Architecture

```text
User uploads image
       ↓
SegFormer-b2 (ONNX, CPU)  →  segmentation mask + region stats
       ↓
Report Generator          →  Multi-Provider LLM Fallback Chain (Gemini, Groq → Local)
       ↓
Streamlit UI              →  overlay image + structured JSON report + PDF download
```

## 🧠 LLM Fallback Chain
To prevent failure from API rate limits or network downtime, MediScan AI uses an interleaved, quality-ranked routing system that steps down through models until a valid report is generated.

### Fallback Execution Order:
1. **Gemini** — `gemini-3.7-flash` *(Primary Cloud)*
2. **Groq** — `openai/gpt-oss-120b`
3. **Gemini** — `gemini-3.5-flash`
4. **Groq** — `openai/gpt-oss-20b`
5. **Gemini** — `gemini-3.5-flash-lite`
6. **Groq** — `qwen/qwen3.6-27b`
7. **Local LLM** — `HuggingFaceTB/SmolLM2-360M-Instruct` *(Offline fallback when cloud APIs fail)*

## Supported Scan Types
- 🔬 **Skin Lesions** — ISIC 2018 dataset (Val Dice: 0.9166, IoU: 0.8474)
- 🫁 **Chest X-Rays** — Montgomery + Shenzhen dataset (Val Dice: 0.9643, IoU: 0.9313)
- 🔊 **Breast Ultrasound** — BUSI dataset (Val Dice: 0.8176, IoU: 0.7065)

## Supported Formats
JPG, PNG, TIFF, BMP, WEBP, DICOM (.dcm)

## Local Setup

```bash
# 1. Clone repository and install dependencies
pip install -r requirements.txt

# 2. Set provider API keys (Optional: defaults to local LLM if API keys are missing)
echo GROQ_API_KEY=your_groq_key_here > .env
echo GEMINI_API_KEY=your_gemini_key_here >> .env

# 3. Launch Streamlit app
streamlit run app.py
```

## Environment Variables

| Variable | Status | Description | Default / Fallback Value |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Recommended | API key for Groq API models | `None` |
| `GEMINI_API_KEY` | Recommended | Primary API key for Google Gemini models | `None` |
| `GOOGLE_API_KEY` | Optional | Alternative alias API key for Google Gemini models | `None` |
| `GROQ_MODELS` | Optional | Comma-separated model fallback order for Groq | `openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.6-27b` |
| `GEMINI_MODELS` | Optional | Comma-separated model fallback order for Gemini | `gemini-3.7-flash, gemini-3.5-flash, gemini-3.5-flash-lite` |
| `LOCAL_LLM_MODEL` | Optional | HuggingFace repository ID for local fallback model | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| `LOCAL_LLM_ENABLED` | Optional | Enable or disable local LLM fallback execution (`true`/`false`) | `true` |

## Project Structure
```text
MediScan_AI/
├── app.py                    # Streamlit UI
├── requirements.txt
├── models/
│   ├── segformer_isic.onnx   # Skin lesion model
│   ├── segformer_xray.onnx   # Chest X-ray model
│   └── segformer_busi.onnx   # Breast ultrasound model
├── segmentation/
│   ├── model.py              # SegFormer ONNX inference engine
│   └── postprocess.py        # Mask overlay & stats extraction
├── report/
│   ├── generator.py          # LLM multi-provider fallback router
│   └── pdf_export.py         # Report export to PDF
└── evaluation/
    └── metrics.py            # Dice & IoU calculation functions
```

## ⚠️ Disclaimer
This is NOT a substitute for professional medical diagnosis. Always consult a licensed physician.