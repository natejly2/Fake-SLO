# Fake-SLO

Train a paired model that generates synthetic SLO images from vessel masks.

## Colab Notebooks

- `train_mask_to_slo_colab.ipynb`
- `inference_slo_from_mask_colab.ipynb`

Both notebooks are preconfigured for:
- `/content/drive/MyDrive/Kodiak/STORAGE/fake_slo/SLO_VESSELS`
- `/content/drive/MyDrive/Kodiak/STORAGE/fake_slo/runs/slo_unet_v6e1`

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data

Expected paired files in `SLO_VESSELS/`:

- `vessels_1.tiff` <-> `slo_1.tiff`
- `vessels_2.tiff` <-> `slo_2.tiff`
- ...

## Train

```bash
python scripts/train_mask_to_slo.py \
  --data-dir SLO_VESSELS \
  --output-dir runs/slo_unet \
  --image-width 1024 \
  --image-height 512 \
  --epochs 200 \
  --batch-size 2 \
  --hflip-prob 0.5 \
  --vflip-prob 0.0 \
  --rotate-deg 10 \
  --amp
```

Notes:
- `--hflip-prob` controls horizontal flips.
- `--vflip-prob` controls vertical flips (default `0.0`).
- `--rotate-deg` applies a random rotation in `[-deg, +deg]`.
- Augmentations are synchronized between mask and SLO target.

## Generate

From a single mask:

```bash
python scripts/generate_slo_from_mask.py \
  --checkpoint runs/slo_unet/best.pt \
  --mask SLO_VESSELS/vessels_1.tiff \
  --output-dir runs/slo_unet/generated
```

From all masks in a folder:

```bash
python scripts/generate_slo_from_mask.py \
  --checkpoint runs/slo_unet/best.pt \
  --mask SLO_VESSELS \
  --output-dir runs/slo_unet/generated
```
