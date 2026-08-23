# 🩺 MediScan AI — Medical Image Segmentation + Report Generation

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-MediScan%20AI-FF4B4B?style=for-the-badge)](https://mediscanai-app.streamlit.app/)

AI-powered pipeline that segments medical scans using SegFormer-b2 (fine-tuned) and generates structured clinical reports via Groq LLM. Fully CPU-deployable.

## Architecture

```
User uploads image
       ↓
SegFormer-b2 (ONNX, CPU)  →  segmentation mask + region stats
       ↓
Groq API                   →  structured clinical report (JSON)
       ↓
Streamlit UI               →  overlay image + report + PDF download
```

## Supported Scan Types
- 🔬 **Skin Lesions** — ISIC 2018 dataset (Val Dice: 0.9166, IoU: 0.8474)
- 🫁 **Chest X-Rays** — Montgomery + Shenzhen dataset (Val Dice: 0.9643, IoU: 0.9313)
- 🔊 **Breast Ultrasound** — BUSI dataset (Val Dice: 0.8176, IoU: 0.7065)

## Supported Formats
JPG, PNG, TIFF, BMP, WEBP, DICOM (.dcm)

## Local Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your Groq API key
echo GROQ_API_KEY=your_key_here > .env

# 3. Run
streamlit run app.py
```

## Project Structure
```
MediScan_AI/
├── app.py                    # Streamlit UI
├── requirements.txt
├── models/
│   ├── segformer_isic.onnx   # Skin lesion model
│   ├── segformer_xray.onnx   # Chest X-ray model
│   └── segformer_busi.onnx   # Breast ultrasound model
├── segmentation/
│   ├── model.py              # SegFormer ONNX inference
│   └── postprocess.py        # Overlay + stats extraction
├── report/
│   ├── generator.py          # Groq report generation
│   └── pdf_export.py         # PDF download
└── evaluation/
    └── metrics.py            # Dice + IoU
```

## ⚠️ Disclaimer
This is NOT a substitute for professional medical diagnosis. Always consult a licensed physician.
