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
- 🔬 **Skin Lesions** — ISIC 2018 dataset (Val Dice: 0.9016, IoU: 0.8364)
- 🫁 **Chest X-Rays** — Chest X-Ray Masks and Labels dataset (Montgomery + Shenzhen) (Val Dice: 0.9650, IoU: 0.9334)
- 🔊 **Breast Ultrasound** — BUSI dataset (Val Dice: 0.8429, IoU: 0.7673)

## Why SegFormer-b2?

SegFormer-b2 was selected after benchmarking 10 segmentation architectures — SegFormer-b0/b1/b2, and five CNN decoder architectures (U-Net, U-Net++, DeepLabV3+, FPN, MAnet) with ResNet34 encoders, plus MobileNetV2 and EfficientNet-B0 encoder variants on U-Net and DeepLabV3+ respectively — across all three datasets. Full results are in [`evaluation/model_comparision.csv`](evaluation/model_comparision.csv).

| Dataset | Best alternative | Alt. Dice | SegFormer-b2 Dice | Margin |
| --- | --- | --- | --- | --- |
| ISIC | MAnet (ResNet34) | 0.9002 | **0.9016** | +0.0014 |
| BUSI | MAnet (ResNet34) | 0.8317 | **0.8429** | +0.0112 |
| X-Ray | U-Net++ (ResNet34) | 0.9662 | 0.9650 | −0.0012 |

SegFormer-b2 is the top performer on ISIC and BUSI. On X-ray the field is essentially saturated (all 10 models land within ~0.004 Dice of each other), so the choice there comes down to consistency across all three scan types rather than a single-dataset win.

## Dataset Download
The datasets used for training are publicly available and are not included in this repository.

### 1. ISIC 2018 — Skin Lesion Segmentation
```bash
pip install kaggle
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation
unzip isic2018-challenge-task1-data-segmentation.zip
```
Dataset: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation

### 2. Chest X-Rays — Masks and Labels (Montgomery + Shenzhen)
```bash
kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels
unzip chest-xray-masks-and-labels.zip -d XRay_with_masks/
```
Dataset: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

### 3. BUSI — Breast Ultrasound Images
```bash
kaggle datasets download -d sabahesaraki/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip
```
Dataset: https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset

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

## Training
Training is performed using the provided `train.py` script.

For GPU-based training, install the required dependencies using:
```bash
bash setup_training.sh
```

The training script supports the following datasets:
```bash
# ISIC 2018
python train.py --dataset isic --data_root ./ISIC2018 --epochs 50 --batch_size 16

# BUSI
python train.py --dataset busi --data_root ./Dataset_BUSI_with_GT --epochs 80 --batch_size 8

# Chest X-rays
python train.py --dataset xray --data_root ./XRay_with_masks --epochs 60 --batch_size 8
```

After training, the resulting models can be exported to ONNX for CPU-based inference in the Streamlit application.

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
    ├── metrics.py             # Dice & IoU calculation functions
    └── model_comparison.csv   # Full 10-architecture benchmark (Dice, IoU, params, ONNX size)
```

## ⚠️ Disclaimer
This is NOT a substitute for professional medical diagnosis. Always consult a licensed physician.