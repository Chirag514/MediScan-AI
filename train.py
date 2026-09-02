"""
train.py — Multi-Architecture Fine-tuning for Medical Image Segmentation
Supports datasets: ISIC 2018 (skin), BUSI (ultrasound), Chest X-ray (montgomery+shenzhen)
Supports architectures: SegFormer (b0/b1/b2), U-Net, U-Net++, DeepLabV3+, FPN, MAnet
                        (all encoder-swappable via segmentation_models_pytorch)

All models are wrapped to share one interface — model(imgs) -> (B, 1, H, W) logits
at full input resolution — so train/eval/export code is architecture-agnostic.
This keeps ONNX export identical across models, which is what you want for a
Streamlit comparison app: one inference wrapper, N .onnx files.

Usage:
    python train.py --dataset isic --model_arch segformer_b2 --data_root ./ISIC2018 --epochs 50
    python train.py --dataset busi --model_arch unet_resnet34 --data_root ./Dataset_BUSI_with_GT --epochs 80
    python train.py --dataset xray --model_arch deeplabv3plus_resnet34 --data_root "./XRay_with_masks/Lung Segmentation" --epochs 60

Install extra deps for the smp-based architectures:
    pip install segmentation-models-pytorch
"""

import os
import re
import csv
import argparse
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    dice = (2 * inter + smooth) / (
        pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth
    )
    return dice.mean().item()


def iou_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean().item()


# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        inter = (probs * targets).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + smooth) / (
            probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        return bce_loss + dice.mean()


# ── Dataset Builders ─────────────────────────────────────────────────────────

def get_isic_pairs(data_root):
    root = Path(data_root)
    img_dir = root / "ISIC2018_Task1-2_Training_Input"
    mask_dir = root / "ISIC2018_Task1_Training_GroundTruth"

    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}_segmentation.png"
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))

    print(f"ISIC: found {len(pairs)} image-mask pairs")
    return pairs


def get_busi_pairs(data_root):
    root = Path(data_root)
    pairs = []

    for category in ["benign", "malignant"]:
        cat_dir = root / category
        if not cat_dir.exists():
            continue

        for img_path in sorted(cat_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue

            stem = img_path.stem
            mask_path = cat_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                continue

            extra_masks = sorted(cat_dir.glob(f"{stem}_mask_*.png"))
            pairs.append((
                str(img_path),
                str(mask_path),
                [str(m) for m in extra_masks],
            ))

    print(f"BUSI: found {len(pairs)} image-mask pairs")
    return [(p[0], p[1], p[2]) for p in pairs]


def get_xray_pairs(data_root):
    root = Path(data_root)

    candidates = [
        (root / "data" / "Lung Segmentation" / "CXR_png",
         root / "data" / "Lung Segmentation" / "masks"),
        (root / "Lung Segmentation" / "CXR_png",
         root / "Lung Segmentation" / "masks"),
        (root / "CXR_png", root / "masks"),
    ]

    img_dir = mask_dir = None
    for img_d, mask_d in candidates:
        if img_d.exists() and mask_d.exists():
            img_dir = img_d
            mask_dir = mask_d
            break

    if img_dir is None:
        print("ERROR: Could not find CXR_png/masks folders.")
        print(f"  Searched under: {data_root}")
        return []

    print(f"X-Ray images: {img_dir}")
    print(f"X-Ray masks:  {mask_dir}")

    pairs = []
    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        for mask_name in [
            f"{stem}_mask.png",
            f"{stem}.png",
            f"{stem}_seg.png",
            f"{stem}_segmentation.png",
        ]:
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                pairs.append((str(img_path), str(mask_path)))
                break

    print(f"X-Ray: found {len(pairs)} image-mask pairs")
    if len(pairs) == 0:
        sample_masks = list(mask_dir.glob("*.png"))[:5]
        print(f"  Sample mask filenames: {[m.name for m in sample_masks]}")
        sample_imgs = list(img_dir.glob("*.png"))[:3]
        print(f"  Sample image filenames: {[i.name for i in sample_imgs]}")
    return pairs


# ── Augmentation Transforms ───────────────────────────────────────────────────

class SegAugment:
    """Paired augmentation for image + mask."""
    def __init__(self, img_size=512, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        img_tensor = TF.to_tensor(image)
        mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
        mask_tensor = (mask_tensor > 127).float()

        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return img_tensor, mask_tensor


# ── Dataset Class ─────────────────────────────────────────────────────────────

class MedSegDataset(Dataset):
    def __init__(self, pairs, dataset_type, transform):
        self.pairs = pairs
        self.dataset_type = dataset_type
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def _load_busi(self, item):
        img_path, mask_path, extra_masks = item
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert("L"))

        for em_path in extra_masks:
            em = np.array(Image.open(em_path).convert("L"))
            mask = np.clip(mask.astype(int) + em.astype(int), 0, 255).astype(np.uint8)

        return image, Image.fromarray(mask)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        if self.dataset_type == "busi":
            image, mask = self._load_busi(item)
        else:
            img_path, mask_path = item
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

        img_tensor, mask_tensor = self.transform(image, mask)
        return img_tensor, mask_tensor


# ── Model Zoo ─────────────────────────────────────────────────────────────────
# Every wrapper below exposes the SAME interface: forward(imgs) -> (B, 1, H, W)
# raw logits at full input resolution. This is what makes ONNX export and the
# Streamlit inference code identical across architectures.

class SegformerWrapper(nn.Module):
    """Wraps a HF SegFormer model: upsamples to input res and takes the
    foreground channel, so downstream code never has to know it's SegFormer."""

    def __init__(self, hf_name="nvidia/mit-b2", img_size=512, num_labels=2):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hf_name, num_labels=num_labels, ignore_mismatched_sizes=True,
        )
        self.img_size = img_size

    def forward(self, imgs):
        logits = self.model(pixel_values=imgs).logits         # (B, num_labels, H/4, W/4)
        logits_up = F.interpolate(
            logits, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        return logits_up[:, 1:2, :, :]                         # (B, 1, H, W)


class SmpWrapper(nn.Module):
    """Wraps a segmentation_models_pytorch model. smp decoders already upsample
    to input resolution and we ask for classes=1, so this is a passthrough —
    but the wrapper keeps every architecture on the same forward() signature."""

    def __init__(self, arch, encoder_name, encoder_weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp

        arch_map = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "manet": smp.MAnet,
        }
        self.model = arch_map[arch](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, imgs):
        return self.model(imgs)                                # (B, 1, H, W) already


# Registry: name -> factory(img_size) -> nn.Module implementing the shared interface
MODEL_REGISTRY = {
    "segformer_b0": lambda img_size: SegformerWrapper("nvidia/mit-b0", img_size),
    "segformer_b1": lambda img_size: SegformerWrapper("nvidia/mit-b1", img_size),
    "segformer_b2": lambda img_size: SegformerWrapper("nvidia/mit-b2", img_size),
    "unet_resnet34": lambda img_size: SmpWrapper("unet", "resnet34"),
    "unet_mobilenetv2": lambda img_size: SmpWrapper("unet", "mobilenet_v2"),
    "unetplusplus_resnet34": lambda img_size: SmpWrapper("unetplusplus", "resnet34"),
    "deeplabv3plus_resnet34": lambda img_size: SmpWrapper("deeplabv3plus", "resnet34"),
    "deeplabv3plus_efficientnetb0": lambda img_size: SmpWrapper("deeplabv3plus", "efficientnet-b0"),
    "fpn_resnet34": lambda img_size: SmpWrapper("fpn", "resnet34"),
    "manet_resnet34": lambda img_size: SmpWrapper("manet", "resnet34"),
}


def build_model(model_arch, img_size):
    if model_arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_arch: {model_arch}. "
            f"Choices: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_arch](img_size)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)                                   # (B, 1, H, W)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(torch.sigmoid(logits).detach().cpu(), masks.cpu())

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks)
        preds = torch.sigmoid(logits).cpu()

        total_loss += loss.item()
        total_dice += dice_score(preds, masks.cpu())
        total_iou += iou_score(preds, masks.cpu())

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_onnx(model, save_path, img_size, device):
    """Every wrapper takes a plain image tensor and returns (B,1,H,W) logits,
    so this export code is identical no matter which model_arch was trained —
    which is exactly what you want for loading N .onnx files into Streamlit
    with one shared onnxruntime inference function."""
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {save_path}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def log_result(csv_path, row):
    """Append one row to the shared results CSV, writing the header if new."""
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged results to {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = args.img_size
    print(f"Device: {device} | Dataset: {args.dataset} | Model: {args.model_arch} | Image size: {img_size}")

    if args.dataset == "isic":
        all_pairs = get_isic_pairs(args.data_root)
        train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.15, random_state=42)
    elif args.dataset == "busi":
        all_pairs = get_busi_pairs(args.data_root)
        train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.20, random_state=42)
    elif args.dataset == "xray":
        all_pairs = get_xray_pairs(args.data_root)
        train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.20, random_state=42)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    train_tfm = SegAugment(img_size=img_size, is_train=True)
    val_tfm = SegAugment(img_size=img_size, is_train=False)

    train_ds = MedSegDataset(train_pairs, args.dataset, train_tfm)
    val_ds = MedSegDataset(val_pairs, args.dataset, val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model = build_model(args.model_arch, img_size).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    best_iou_at_best = 0.0
    best_epoch = 0
    tag = f"{args.model_arch}_{args.dataset}"
    best_path = save_dir / f"{tag}_best.pth"

    print("\n" + "=" * 60)
    print(f"Starting training: {args.epochs} epochs | {args.model_arch}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou_at_best = val_iou
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")

    print("\nExporting to ONNX...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    onnx_path = save_dir / f"{tag}.onnx"
    export_onnx(model, str(onnx_path), img_size, device)

    onnx_size_kb = onnx_path.stat().st_size / 1024
    num_params = count_params(model)

    log_result(
        save_dir / "results.csv",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model_arch": args.model_arch,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "img_size": img_size,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_val_dice": round(best_dice, 4),
            "best_val_iou": round(best_iou_at_best, 4),
            "num_params": num_params,
            "onnx_size_kb": round(onnx_size_kb, 1),
        },
    )

    print("\n" + "=" * 60)
    print(f"DONE. Files saved in {save_dir}/")
    print(f"  Best checkpoint : {best_path}")
    print(f"  ONNX model      : {onnx_path}")
    print(f"  Best Val Dice   : {best_dice:.4f}")
    print(f"  Best Val IoU    : {best_iou_at_best:.4f}")
    print(f"  Results log     : {save_dir / 'results.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["isic", "busi", "xray"])
    parser.add_argument("--model_arch", default="segformer_b2",
                        choices=sorted(MODEL_REGISTRY.keys()),
                        help="Which architecture to train")
    parser.add_argument("--data_root", required=True, help="Root folder of the dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--save_dir", default="./checkpoints_2")
    args = parser.parse_args()
    main(args)