# Lithography Hotspot Detection Using Focal Loss and Hard-Example Learning

### AI and Machine Learning for IC Design — Digital Assignment 2

**Group G10 | Vellore Institute of Technology, Chennai**

---

## Abstract

Lithography hotspot detection is a critical machine-learning problem in semiconductor manufacturing, where layout patterns that are difficult to print reliably can lead to manufacturing defects such as bridging and pinching.

A major challenge in lithography hotspot detection is the severe class imbalance between hotspot and non-hotspot patterns. In highly imbalanced datasets, conventional classification losses can be dominated by the large number of easy non-hotspot samples, potentially limiting the detector's ability to learn difficult hotspot examples.

This project investigates training strategies that explicitly emphasize difficult and minority samples. A lightweight convolutional neural network (CNN) is used as the common architecture, and conventional Binary Cross-Entropy (BCE) training is compared with Focal Loss. Additional experiments investigate Online Hard Example Mining (OHEM) and a combination of Focal Loss with OHEM.

Experiments are conducted independently on all five ICCAD-12 lithography hotspot benchmarks. Balanced Accuracy is used as the primary evaluation metric because of the severe class imbalance, with Precision, Recall, Specificity, F1-score, and confusion matrices reported as additional measures.

---

## Research Question

> **Can a learning strategy that emphasizes difficult and minority samples improve lithography hotspot detection under the severe class imbalance of the ICCAD-12 benchmarks?**

### Objectives

The project aims to:

- Establish a conventional BCE-based baseline.
- Investigate Focal Loss as a training-level approach for handling difficult and minority samples.
- Investigate Online Hard Example Mining (OHEM).
- Examine whether combining Focal Loss and OHEM provides complementary benefits.
- Evaluate the methods independently across the five ICCAD-12 benchmarks.
- Analyze the precision–recall and specificity trade-offs produced by different training strategies.

---

## Methods

Four training configurations are considered:

| Method | Architecture | Loss | Hard-Example Selection |
|---|---|---|---|
| **BCE** | Lightweight CNN | Binary Cross-Entropy | None |
| **Focal** | Lightweight CNN | Focal Loss | None |
| **OHEM** | Lightweight CNN | BCE | Online Hard Example Mining |
| **Focal + OHEM** | Lightweight CNN | Focal Loss | Online Hard Example Mining |

The **BCE vs Focal Loss comparison** is the primary five-benchmark experiment.

OHEM and Focal + OHEM are additional hard-example learning experiments performed on selected benchmarks.

---

## Dataset

The experiments use the five ICCAD-12 lithography hotspot benchmarks supplied for the course assignment.

The dataset is divided into:

- Hotspot (HS)
- Non-hotspot (NHS)

The official dataset is **not included in this repository**.

### Dataset Distribution

| Benchmark | Train HS | Train NHS | Test HS | Test NHS |
|:--|--:|--:|--:|--:|
| B1 | 99 | 340 | 226 | 4,679 |
| B2 | 174 | 5,285 | 498 | 41,298 |
| B3 | 909 | 4,643 | 1,808 | 46,333 |
| B4 | 95 | 4,452 | 177 | 31,890 |
| B5 | 26 | 2,716 | 41 | 19,327 |

The class distribution is highly imbalanced, particularly in the test sets. This motivates the use of Balanced Accuracy and the investigation of training strategies designed to emphasize difficult examples.

### Expected Dataset Structure

Place the course-provided dataset locally using the following structure:

```text
iccad-official/
│
├── iccad1/
│   ├── train/
│   │   ├── train_hs/
│   │   └── train_nhs/
│   └── test/
│       ├── test_hs/
│       └── test_nhs/
│
├── iccad2/
│   ├── train/
│   └── test/
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
