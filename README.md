# G10 — Focal and Hard-Example Learning for Lithography Hotspot Detection

This project implements the G10 DA-2 methodology on the five ICCAD-12 benchmarks.

## Experimental matrix

1. Baseline: lightweight CNN + BCE
2. Focal: lightweight CNN + Focal Loss
3. OHEM: lightweight CNN + BCE + online hard-example mining
4. Focal+OHEM: lightweight CNN + Focal Loss + OHEM

All primary experiments keep the benchmark, preprocessing, split seed, architecture, optimizer, batch size and evaluation protocol fixed.

## Dataset layout

iccad-official/
  iccad1/ ... iccad5/
    train/train_hs/*.png
    train/train_nhs/*.png
    test/test_hs/*.png
    test/test_nhs/*.png

Images are resized to 64x64 grayscale for the lightweight CNN.

## Important

Do not merge the five benchmarks for the primary experiment. Train/evaluate each benchmark separately.

Use the same train/validation split for every model within a benchmark. The official test set is never used for training or model selection.

## Suggested run

python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode bce --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode focal --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode ohem --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode focal_ohem --epochs 5

Then evaluate each checkpoint with:

python evaluate_g10.py --root /path/to/iccad-official --benchmark 1 --checkpoint checkpoints/b1_focal.pt

Repeat for benchmarks 1–5.

## Focal loss

Default alpha=0.75 and gamma=2.0. Gamma can be changed for the ablation study.

## OHEM

Per batch, the highest-loss 50% of examples are retained for the optimization step. The hard fraction is configurable.

## Reported metrics

Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score and confusion matrix.

The test set is evaluated only after training/model selection.
