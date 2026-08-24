"""
train.py — SegFormer-b2 Fine-tuning for Medical Image Segmentation
Supports: ISIC 2018 (skin), BUSI (ultrasound), Chest X-ray (montgomery+shenzhen)

Usage:
    python train.py --dataset isic   --data_root ./ISIC2018         --epochs 50
    python train.py --dataset busi   --data_root ./Dataset_BUSI_with_GT --epochs 80
    python train.py --dataset xray  --data_root ./images            --epochs 60
"""

import os
import re
import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F

# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ── Metrics ───────────────────────────────────────────────────────────────────
def dice_score(pred, target, smooth=1e-6):
    pred   = (pred > 0.5).float().view(-1)
    target = target.float().view(-1)
    inter  = (pred * target).sum()
    return ((2 * inter + smooth) / (pred.sum() + target.sum() + smooth)).item()

def iou_score(pred, target, smooth=1e-6):
    pred   = (pred > 0.5).float().view(-1)
    target = target.float().view(-1)
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()

# ── Loss ──────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs    = torch.sigmoid(logits)
        smooth   = 1e-6
        inter    = (probs * targets).sum(dim=(1, 2, 3))
        dice     = 1 - (2 * inter + smooth) / (
            probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
        )
        return bce_loss + dice.mean()

# ── Dataset Builders ─────────────────────────────────────────────────────────

def get_isic_pairs(data_root):
    """
    ISIC 2018 Task 1 structure:
      ISIC2018_Task1-2_Training_Input/ISIC_XXXXXXX.jpg
      ISIC2018_Task1_Training_GroundTruth/ISIC_XXXXXXX_segmentation.png
    """
    root      = Path(data_root)
    img_dir   = root / "ISIC2018_Task1-2_Training_Input"
    mask_dir  = root / "ISIC2018_Task1_Training_GroundTruth"

    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem      = img_path.stem                        # ISIC_XXXXXXX
        mask_path = mask_dir / f"{stem}_segmentation.png"
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))

    print(f"ISIC: found {len(pairs)} image-mask pairs")
    return pairs


def get_busi_pairs(data_root):
    """
    BUSI structure (benign + malignant only, skip normal):
      benign/benign (N).png
      benign/benign (N)_mask.png        ← primary mask
      benign/benign (N)_mask_1.png      ← secondary mask (merge with primary)
    """
    root  = Path(data_root)
    pairs = []

    for category in ["benign", "malignant"]:
        cat_dir = root / category
        if not cat_dir.exists():
            continue

        # collect all base images (no _mask in name)
        for img_path in sorted(cat_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue

            stem      = img_path.stem          # e.g. "benign (1)"
            # primary mask
            mask_path = cat_dir / f"{stem}_mask.png"
            if not mask_path.exists():
                continue

            # check for additional masks (e.g. _mask_1.png) — merge them
            extra_masks = sorted(cat_dir.glob(f"{stem}_mask_*.png"))
            pairs.append((
                str(img_path),
                str(mask_path),
                [str(m) for m in extra_masks],   # may be empty
            ))

    print(f"BUSI: found {len(pairs)} image-mask pairs")
    # normalise to (img, mask) tuples by merging extra masks at load time
    return [(p[0], p[1], p[2]) for p in pairs]


def get_xray_pairs(data_root):
    """
    XRay_with_masks dataset structure:
      data/Lung Segmentation/CXR_png/MCUCXR_XXXX.png  ← images
      data/Lung Segmentation/masks/MCUCXR_XXXX_mask.png ← masks
    """
    root     = Path(data_root)

    # Try multiple known layouts
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
            img_dir  = img_d
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
        # try common mask naming conventions
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
        # show what masks actually look like for debugging
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
        # resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask  = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.is_train:
            # random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # random rotation ±30°
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

            # color jitter on image only
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))

        # to tensor
        img_tensor  = TF.to_tensor(image)                    # (3, H, W) float [0,1]
        mask_tensor = torch.from_numpy(
            np.array(mask)
        ).float().unsqueeze(0)                                # (1, H, W)

        # normalise mask to binary [0, 1]
        mask_tensor = (mask_tensor > 127).float()

        # normalise image (ImageNet stats)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )
        return img_tensor, mask_tensor

# ── Dataset Class ─────────────────────────────────────────────────────────────

class MedSegDataset(Dataset):
    def __init__(self, pairs, dataset_type, transform):
        self.pairs        = pairs
        self.dataset_type = dataset_type
        self.transform    = transform

    def __len__(self):
        return len(self.pairs)

    def _load_busi(self, item):
        img_path, mask_path, extra_masks = item
        image = Image.open(img_path).convert("RGB")
        mask  = np.array(Image.open(mask_path).convert("L"))

        # merge extra masks via OR
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
            mask  = Image.open(mask_path).convert("L")

        img_tensor, mask_tensor = self.transform(image, mask)
        return img_tensor, mask_tensor

# ── Model ─────────────────────────────────────────────────────────────────────

def build_segformer(num_labels=2):
    """
    SegFormer-b2 pretrained on ImageNet-22k, fine-tuned for binary segmentation.
    num_labels=2 → background + foreground.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model

# ── Training Loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, img_size):
    model.train()
    total_loss = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=imgs)
        logits  = outputs.logits                 # (B, num_labels, H/4, W/4)

        # upsample to original size
        logits_up = F.interpolate(
            logits, size=(img_size, img_size),
            mode="bilinear", align_corners=False
        )

        # binary: use foreground channel (index 1)
        fg_logits = logits_up[:, 1:2, :, :]     # (B, 1, H, W)

        loss = criterion(fg_logits, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(torch.sigmoid(fg_logits).detach().cpu(), masks.cpu())

    n = len(loader)
    if n == 0:
        raise ValueError("Training dataset produced no batches")
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, img_size):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou  = 0

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        outputs   = model(pixel_values=imgs)
        logits    = outputs.logits
        logits_up = F.interpolate(
            logits, size=(img_size, img_size),
            mode="bilinear", align_corners=False
        )
        fg_logits = logits_up[:, 1:2, :, :]
        loss      = criterion(fg_logits, masks)
        preds     = torch.sigmoid(fg_logits).cpu()

        total_loss += loss.item()
        total_dice += dice_score(preds, masks.cpu())
        total_iou  += iou_score(preds,  masks.cpu())

    n = len(loader)
    if n == 0:
        raise ValueError("Validation dataset produced no batches")
    return total_loss / n, total_dice / n, total_iou / n

# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_onnx(model, save_path, img_size, device):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        model,
        {"pixel_values": dummy},
        save_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits":       {0: "batch"},
        },
        opset_version=14,
    )
    print(f"ONNX model saved to {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = args.img_size
    print(f"Device: {device} | Dataset: {args.dataset} | Image size: {img_size}")

    # ── build pairs ────────────────────────────────────────────────────────
    if args.dataset == "isic":
        all_pairs = get_isic_pairs(args.data_root)
        # ISIC has its own validation split — use training set and split manually
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=0.15, random_state=42
        )
    elif args.dataset == "busi":
        all_pairs = get_busi_pairs(args.data_root)
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=0.20, random_state=42
        )
    elif args.dataset == "xray":
        all_pairs = get_xray_pairs(args.data_root)
        train_pairs, val_pairs = train_test_split(
            all_pairs, test_size=0.20, random_state=42
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(all_pairs) < 2:
        raise ValueError("At least two image/mask pairs are required for training")

    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ── transforms ─────────────────────────────────────────────────────────
    train_tfm = SegAugment(img_size=img_size, is_train=True)
    val_tfm   = SegAugment(img_size=img_size, is_train=False)

    train_ds = MedSegDataset(train_pairs, args.dataset, train_tfm)
    val_ds   = MedSegDataset(val_pairs,   args.dataset, val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ── model + optimizer ──────────────────────────────────────────────────
    model     = build_segformer().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── training ───────────────────────────────────────────────────────────
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    best_path = save_dir / f"segformer_{args.dataset}_best.pth"

    print("\n" + "="*60)
    print(f"Starting training: {args.epochs} epochs")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, img_size
        )
        val_loss, val_dice, val_iou = evaluate(
            model, val_loader, criterion, device, img_size
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} IoU: {val_iou:.4f}"
        )

        # save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")

    print(f"\nTraining complete. Best Val Dice: {best_dice:.4f}")

    # ── export to ONNX ─────────────────────────────────────────────────────
    print("\nExporting to ONNX...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    onnx_path = save_dir / f"segformer_{args.dataset}.onnx"
    export_onnx(model, str(onnx_path), img_size, device)

    print("\n" + "="*60)
    print(f"DONE. Files saved in {save_dir}/")
    print(f"  Best checkpoint : {best_path}")
    print(f"  ONNX model      : {onnx_path}")
    print(f"  Best Val Dice   : {best_dice:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,
                        choices=["isic", "busi", "xray"])
    parser.add_argument("--data_root",  required=True,
                        help="Root folder of the dataset")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=6e-5)
    parser.add_argument("--img_size",   type=int, default=512)
    parser.add_argument("--save_dir",   default="./checkpoints")
    args = parser.parse_args()
    main(args)