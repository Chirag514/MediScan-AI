#!/bin/bash
# Usage: bash setup_training.sh

conda activate mediscan

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.46.0
pip install segmentation-models-pytorch==0.3.4
pip install scikit-learn
pip install onnx onnxruntime
pip install Pillow numpy

echo "All packages installed."
echo ""
echo "Available --model_arch options (from train.py's MODEL_REGISTRY):"
echo "  segformer_b0                    (transformers, no smp needed)"
echo "  segformer_b1                    (transformers, no smp needed)"
echo "  segformer_b2                    (transformers, no smp needed)"
echo "  unet_resnet34                   (smp)"
echo "  unet_mobilenetv2                (smp)"
echo "  unetplusplus_resnet34           (smp)"
echo "  deeplabv3plus_resnet34          (smp)"
echo "  deeplabv3plus_efficientnetb0    (smp)"
echo "  fpn_resnet34                    (smp)"
echo "  manet_resnet34                  (smp)"
echo ""
echo "Run training with, e.g.:"
echo "  python train.py --dataset isic --model_arch segformer_b2 --data_root ./ISIC2018 --epochs 50 --batch_size 16"
echo "  python train.py --dataset busi --model_arch manet_resnet34 --data_root ./Dataset_BUSI_with_GT --epochs 80 --batch_size 8"
echo '  python train.py --dataset xray --model_arch deeplabv3plus_efficientnetb0 --data_root "./XRay_with_masks/Lung Segmentation" --epochs 60 --batch_size 8'
echo ""
echo "Swap --model_arch to any option above to reproduce the full benchmark sweep"
echo "(results are appended per-run to <save_dir>/results.csv)."