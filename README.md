# G10 — Lithography Hotspot Detection Using Focal Loss and Hard-Example Learning

**AI and Machine Learning for IC Design — Digital Assignment 2**

**Group G10 | Vellore Institute of Technology, Chennai**

---

## 1. Overview

This repository contains the complete implementation and experimental code for Group G10's Digital Assignment 2 project on **lithography hotspot detection**.

The project investigates whether training strategies that emphasize difficult and minority samples can improve hotspot detection under the severe class imbalance present in the ICCAD-12 lithography hotspot benchmarks.

A common lightweight convolutional neural network (CNN) is used throughout the study. Four training strategies are investigated:

1. **Binary Cross-Entropy (BCE)** — baseline
2. **Focal Loss** — primary proposed method
3. **Online Hard Example Mining (OHEM)**
4. **Focal Loss + OHEM**

The five ICCAD-12 benchmarks are treated independently.

### Research Question

> **Can a learning strategy that emphasizes difficult and minority samples improve lithography hotspot detection under the severe class imbalance of the ICCAD-12 benchmarks?**

---

# 2. Repository Contents

```text
G10-Lithography-Hotspot-Detection/
│
├── README.md
├── requirements.txt
│
├── train_g10.py
├── train_g10_ohem.py
├── evaluate_g10.py
├── evaluate_g10_ohem.py
├── generate_confusion_matrices.py
├── generate_plots.py
├── consolidate_results.py
└── generate_analysis.py
```

### Script overview

| File                             | Purpose                                 |
| -------------------------------- | --------------------------------------- |
| `train_g10.py`                   | Main CNN training implementation        |
| `train_g10_ohem.py`              | OHEM training implementation            |
| `evaluate_g10.py`                | Model evaluation and metric calculation |
| `evaluate_g10_ohem.py`           | Evaluation for OHEM experiments         |
| `generate_confusion_matrices.py` | Generates confusion-matrix plots        |
| `generate_plots.py`              | Generates experiment comparison plots   |
| `consolidate_results.py`         | Consolidates experiment results         |
| `generate_analysis.py`           | Generates a textual experiment summary  |
| `requirements.txt`               | Python dependencies                     |
| `README.md`                      | Reproduction and project documentation  |

---

# 3. Dataset

## ICCAD-12 Lithography Hotspot Benchmark

The experiments use the **ICCAD-2012 lithography hotspot detection benchmark suite**.

The ICCAD-2012 CAD contest was introduced as a benchmark for identifying layout topologies that may cause yield/manufacturing problems and explicitly addresses challenges including widely different classes and limited data.

### Original benchmark reference

**IEEE / ICCAD-2012 CAD Contest:**

https://ieeexplore.ieee.org/document/6386635

The original contest download infrastructure is no longer consistently accessible. For this reason, researchers commonly use preserved copies/mirrors of the ICCAD-2012 benchmark.

One publicly available lithography-hotspot repository provides a dataset download mirror:

https://github.com/Intelectron6/Lithography-Hotspot-Detection

The repository identifies its dataset as the ICCAD-12 benchmark and provides a download link.

**Important:** The dataset is not redistributed in this repository. Obtain the benchmark through the course-provided dataset or an appropriate authorized source.

---

# 4. Dataset Organization

After downloading and extracting the dataset, the expected directory structure is:

```text
iccad-official/
│
├── iccad1/
│   ├── train/
│   │   ├── train_hs/
│   │   └── train_nhs/
│   │
│   └── test/
│       ├── test_hs/
│       └── test_nhs/
│
├── iccad2/
│   ├── train/
│   │   ├── train_hs/
│   │   └── train_nhs/
│   │
│   └── test/
│       ├── test_hs/
│       └── test_nhs/
│
├── iccad3/
│   ├── train/
│   └── test/
│
├── iccad4/
│   ├── train/
│   └── test/
│
└── iccad5/
    ├── train/
    └── test/
```

where:

* `HS` = hotspot
* `NHS` = non-hotspot

The five benchmarks must remain separate for the primary experiments.

---

# 5. Dataset Statistics

The dataset used for the experiments contains:

| Benchmark | Train HS | Train NHS | Test HS | Test NHS |
| --------- | -------: | --------: | ------: | -------: |
| B1        |       99 |       340 |     226 |    4,679 |
| B2        |      174 |     5,285 |     498 |   41,298 |
| B3        |      909 |     4,643 |   1,808 |   46,333 |
| B4        |       95 |     4,452 |     177 |   31,890 |
| B5        |       26 |     2,716 |      41 |   19,327 |

The severe imbalance between hotspot and non-hotspot samples is a central motivation for the project.

---

# 6. Environment

The code is designed to run with Python and PyTorch.

The experiments were developed and tested in **Google Colab using an NVIDIA Tesla T4 GPU**.

A GPU is recommended for training but is not required for inspecting the code or running lightweight post-processing.

---

# 7. Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR-USERNAME>/G10-Lithography-Hotspot-Detection.git
```

Enter the repository:

```bash
cd G10-Lithography-Hotspot-Detection
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Google Colab

Clone the repository directly into Colab:

```python
!git clone https://github.com/<YOUR-USERNAME>/G10-Lithography-Hotspot-Detection.git
%cd G10-Lithography-Hotspot-Detection
```

Then install dependencies:

```python
!pip install -r requirements.txt
```

---

# 8. Verify the Environment

Run:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Expected GPU output for the original experimental environment:

```text
CUDA available: True
GPU: Tesla T4
```

Different hardware may produce different training and inference times.

---

# 9. Data Preprocessing

Each layout image undergoes the following preprocessing:

1. Load using PIL.
2. Convert to grayscale.
3. Resize to `64 × 64`.
4. Convert to floating-point representation.
5. Normalize pixel values to `[0, 1]`.
6. Convert to a PyTorch tensor.

The CNN therefore receives:

```text
[batch_size, 1, 64, 64]
```

The same preprocessing is used for the BCE and Focal experiments.

---

# 10. Train / Validation / Test Protocol

Each benchmark is processed independently.

The official training set is divided into:

```text
80% → Training
20% → Validation
```

The split is:

* stratified
* generated using random seed `42`

The official test set is kept completely separate.

### Model selection

The best checkpoint is selected using:

```text
Validation Balanced Accuracy
```

### Final evaluation

After model selection, the selected checkpoint is evaluated on the official test set.

The test set is **not used for training or model selection**.

---

# 11. Model Architecture

The project uses a lightweight CNN.

Each basic block consists of:

```text
Conv2D
12 filters
3 × 3 kernel
ELU

Conv2D
12 filters
3 × 3 kernel
ELU

Conv2D
12 filters
3 × 3 kernel
No activation

Batch Normalization
ELU
MaxPool 2 × 2
```

Two basic blocks are used.

The classifier is:

```text
Dropout(0.3)
      ↓
Linear → 10
      ↓
ELU
      ↓
Linear → 1
```

The model contains:

```text
21,249 trainable parameters
```

---

# 12. Training Configuration

The controlled configuration is:

| Parameter                | Value                        |
| ------------------------ | ---------------------------- |
| Image size               | 64 × 64                      |
| Channels                 | 1                            |
| Train/Validation split   | 80/20                        |
| Split                    | Stratified                   |
| Random seed              | 42                           |
| Batch size               | 128                          |
| Optimizer                | NAdam                        |
| Learning rate            | 1e-3                         |
| Dropout                  | 0.3                          |
| Epochs                   | 5                            |
| Model-selection metric   | Validation Balanced Accuracy |
| Classification threshold | 0.5                          |
| Primary metric           | Balanced Accuracy            |

---

# 13. Training Methods

## 13.1 BCE Baseline

The baseline uses standard Binary Cross-Entropy loss.

This provides the reference point for evaluating the hard-example learning strategies.

---

## 13.2 Focal Loss

The primary proposed method uses Focal Loss.

Configuration:

```text
alpha = 0.75
gamma = 2.0
```

Focal Loss reduces the relative contribution of easy examples and emphasizes examples that are more difficult for the classifier.

---

## 13.3 Online Hard Example Mining

OHEM calculates an individual loss for each sample in a training batch.

The samples are ranked according to their loss and the highest-loss samples are retained for optimization.

Configuration:

```text
hard_fraction = 0.50
```

Therefore, approximately the highest-loss 50% of samples in each batch are used for the optimization step.

---

## 13.4 Focal Loss + OHEM

The combined experiment applies:

```text
Focal Loss
+
Online Hard Example Mining
```

This tests whether explicit hard-example selection provides additional information beyond the example weighting already performed by Focal Loss.

---

# 14. Running the Experiments

## 14.1 BCE — Benchmark 1

```bash
python train_g10.py \
    --root /path/to/iccad-official \
    --benchmark 1 \
    --mode bce \
    --epochs 5
```

---

## 14.2 Focal — Benchmark 1

```bash
python train_g10.py \
    --root /path/to/iccad-official \
    --benchmark 1 \
    --mode focal \
    --epochs 5
```

---

## 14.3 OHEM — Benchmark 1

```bash
python train_g10.py \
    --root /path/to/iccad-official \
    --benchmark 1 \
    --mode ohem \
    --epochs 5
```

---

## 14.4 Focal + OHEM — Benchmark 1

```bash
python train_g10.py \
    --root /path/to/iccad-official \
    --benchmark 1 \
    --mode focal_ohem \
    --epochs 5
```

The benchmark number can be changed to:

```text
1
2
3
4
5
```

---

# 15. Running All Five BCE/Focal Experiments

For the primary experiment, run BCE and Focal independently for Benchmarks 1–5.

### BCE

```bash
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode bce --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 2 --mode bce --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 3 --mode bce --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 4 --mode bce --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 5 --mode bce --epochs 5
```

### Focal

```bash
python train_g10.py --root /path/to/iccad-official --benchmark 1 --mode focal --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 2 --mode focal --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 3 --mode focal --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 4 --mode focal --epochs 5
python train_g10.py --root /path/to/iccad-official --benchmark 5 --mode focal --epochs 5
```

---

# 16. OHEM Experiments

The completed OHEM experiments were performed on Benchmarks 4 and 5.

### Benchmark 4

```bash
python train_g10_ohem.py \
    --root /path/to/iccad-official \
    --benchmark 4 \
    --mode ohem \
    --epochs 5
```

```bash
python train_g10_ohem.py \
    --root /path/to/iccad-official \
    --benchmark 4 \
    --mode focal_ohem \
    --epochs 5
```

### Benchmark 5

```bash
python train_g10_ohem.py \
    --root /path/to/iccad-official \
    --benchmark 5 \
    --mode ohem \
    --epochs 5
```

```bash
python train_g10_ohem.py \
    --root /path/to/iccad-official \
    --benchmark 5 \
    --mode focal_ohem \
    --epochs 5
```

**Important:** OHEM was not completed for all five benchmarks in the final experiment set. Therefore, OHEM results must not be presented as five-benchmark averages.

---

# 17. Checkpoints

Training saves model checkpoints containing information such as:

* model state dictionary
* benchmark
* training mode
* image size
* best epoch
* validation Balanced Accuracy
* random seed

Example checkpoint names:

```text
b1_bce.pt
b1_focal.pt
b4_ohem.pt
b4_focal_ohem.pt
```

The original experimental checkpoints are not included in this source-code repository.

---

# 18. Evaluation

After training, evaluate the selected checkpoint using:

```bash
python evaluate_g10.py \
    --root /path/to/iccad-official \
    --benchmark 1 \
    --checkpoint /path/to/checkpoint.pt
```

The model produces a binary prediction using:

```text
sigmoid(logit) >= 0.5
```

---

# 19. Evaluation Metrics

The following metrics are calculated:

### Balanced Accuracy

```text
Balanced Accuracy = (Recall + Specificity) / 2
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall / Sensitivity

```text
Recall = TP / (TP + FN)
```

### Specificity

```text
Specificity = TN / (TN + FP)
```

### F1-score

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

A confusion matrix is also generated.

---

# 20. Generating Confusion Matrices

After obtaining the experiment results:

```bash
python generate_confusion_matrices.py
```

This generates confusion-matrix visualizations for the completed experiments.

---

# 21. Generating Performance Plots

Run:

```bash
python generate_plots.py
```

This generates the metric comparison plots used for analysis and reporting.

---

# 22. Consolidating Results

To consolidate individual experiment outputs:

```bash
python consolidate_results.py
```

The resulting files can be used for:

* benchmark-wise comparisons
* metric tables
* average metric calculations
* report preparation

---

# 23. Generating the Experimental Analysis

Run:

```bash
python generate_analysis.py
```

This generates a textual summary of the available experimental results.

---

# 24. Reproducing the Main Experimental Results

To reproduce the principal BCE/Focal experiment:

### Step 1 — Obtain the dataset

Obtain the course-provided ICCAD-12 dataset or an appropriate authorized copy.

### Step 2 — Prepare the directory structure

Ensure the five benchmark directories are located under:

```text
/path/to/iccad-official/
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train BCE

Run BCE for Benchmarks 1–5.

### Step 5 — Train Focal

Run Focal Loss for Benchmarks 1–5.

### Step 6 — Evaluate

Evaluate the selected checkpoints on the official test sets.

### Step 7 — Generate results

Run:

```bash
python consolidate_results.py
python generate_confusion_matrices.py
python generate_plots.py
python generate_analysis.py
```

### Step 8 — Compare BCE and Focal

Calculate the arithmetic mean of each metric across Benchmarks 1–5.

The primary metric is Balanced Accuracy.

---

# 25. Reproducibility Configuration

For the primary BCE/Focal experiments, use exactly:

```text
Random seed       : 42
Image size        : 64 × 64
Image channels    : 1
Train/validation  : 80/20
Split             : Stratified
Batch size        : 128
Optimizer         : NAdam
Learning rate     : 1e-3
Dropout           : 0.3
Epochs            : 5
Threshold         : 0.5

Focal alpha       : 0.75
Focal gamma       : 2.0

OHEM fraction     : 0.50
```

---

# 26. Main Experimental Results

The arithmetic mean across the five BCE/Focal benchmark experiments was:

| Metric            |    BCE |      Focal |
| ----------------- | -----: | ---------: |
| Balanced Accuracy | 82.78% | **93.02%** |
| Precision         | 27.07% | **31.96%** |
| Recall            | 86.73% | **96.45%** |
| Specificity       | 78.84% | **89.58%** |
| F1-score          | 36.88% | **42.58%** |

These values are calculated from the completed benchmark-wise test results.

Focal Loss produced a higher arithmetic mean for the reported metrics in the completed BCE/Focal comparison.

The effect varies across individual benchmarks, so the benchmark-level results should also be inspected rather than relying only on the average.

---

# 27. Inference Performance

The implemented CNN contains:

```text
21,249 trainable parameters
```

A model-only inference measurement was performed on an NVIDIA Tesla T4.

Measurement setup:

```text
Input              : 1 × 1 × 64 × 64
Warm-up iterations : 20
Measured runs      : 1000
```

Measured result:

```text
Average inference time : 0.8949 ms/image
Throughput             : 1117.46 images/sec
```

This measurement represents the model forward pass only and does not include:

* image loading
* disk I/O
* image resizing
* preprocessing
* batch construction

Inference performance is hardware-dependent.

---

# 28. Important Experimental-Control Note

The BCE and Focal experiments use the same:

* preprocessing
* architecture
* optimizer
* learning rate
* batch size
* random seed
* stratified splitting protocol
* evaluation threshold

The OHEM experiments use a separate training implementation with a different file enumeration order.

Therefore, OHEM follows the same split protocol and random seed, but the exact individual train/validation file assignment should not be assumed to be identical to the BCE/Focal experiments.

---

# 29. Results Interpretation

The primary conclusion is based on the controlled BCE vs Focal comparison.

Under the experimental setup used in this project, Focal Loss substantially improved Balanced Accuracy across the five benchmark arithmetic mean.

The improvement is related to the severe class imbalance and the different treatment of difficult examples during training.

However, no single metric should be interpreted in isolation.

For example, a classifier can achieve high hotspot recall while simultaneously generating many false positives. Such a classifier may have:

* high Recall,
* low Specificity,
* low Precision,
* and relatively lower Balanced Accuracy.

This is why Balanced Accuracy, Precision, Recall, Specificity, F1-score, and confusion matrices are all reported.

---

# 30. Limitations

The current implementation has several limitations:

1. A single lightweight CNN architecture is used.
2. The principal experiments use one random seed (`42`).
3. The main Focal Loss experiment uses `alpha = 0.75` and `gamma = 2`.
4. A complete Focal Loss gamma sweep was not included in the completed experiment set.
5. A complete OHEM hard-fraction sweep was not included.
6. OHEM and Focal + OHEM were completed for Benchmarks 4 and 5 rather than all five benchmarks.
7. GPU inference time depends on hardware and measurement methodology.
8. The study does not establish universal superiority of one training strategy across all lithography datasets.

---

# 31. Future Work

Possible extensions include:

* Focal Loss gamma ablation
* Focal Loss alpha ablation
* OHEM hard-fraction ablation
* Multi-seed experiments
* Difficult-sample analysis
* Additional lightweight architectures
* Classification threshold optimization
* Calibration analysis
* Additional data augmentation
* Cross-dataset generalization
* Comparison with architecture-level hotspot detection methods

---

# 32. Data and Model Availability

The ICCAD-12 dataset is not redistributed in this repository.

The dataset should be obtained through the course-provided source or another authorized source.

The trained `.pt` checkpoints are also not included in this source-code repository.

This repository contains:

* source code
* dependency information
* training configuration
* evaluation code
* result-generation utilities
* reproducibility instructions

---

# 33. References

### Dataset / Benchmark

ICCAD-2012 CAD Contest:

https://ieeexplore.ieee.org/document/6386635

A publicly available lithography hotspot detection repository providing an ICCAD-12 dataset mirror:

https://github.com/Intelectron6/Lithography-Hotspot-Detection

### Research Papers

1. H. Yang, Y. Lin, B. Yu, and E. F. Y. Young, “Lithography hotspot detection: From shallow to deep learning,” IEEE SOCC, 2017.

2. V. Borisov and J. Scheible, “Lithography Hotspots Detection Using Deep Learning,” IEEE SMACD, 2018.

3. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object Detection,” IEEE ICCV, 2017.

4. A. Shrivastava, A. Gupta, and R. Girshick, “Training Region-Based Object Detectors With Online Hard Example Mining,” IEEE CVPR, 2016.

5. L. Liao et al., “Lithography Hotspot Detection Method Based on Transfer Learning Using Pre-Trained Deep Convolutional Neural Network,” Applied Sciences, 2022.

6. Y. Chen et al., “Lightweight Hotspot Detection Model Fusing SE and ECA Mechanisms,” Micromachines, 2024.

7. M. Lin et al., “An Improved YOLOv5 Model for Lithographic Hotspot Detection,” Micromachines, 2025.

---

# 34. Team

### Group G10

Vinayak Shreenivas Salunke

Harsh Vardhan Singh

Alavala Vishnu Koushik Reddy

**Vellore Institute of Technology, Chennai**

---

## Course

**AI and Machine Learning for IC Design — BEVD402L**


