# G10 — Focal / Hard-Example Learning for Lithography Hotspot Detection

<p align="center">
  <strong>AI and Machine Learning for IC Design — Digital Assignment 2</strong>
</p>

<p align="center">
  <strong>Group G10</strong>
</p>

---

## Overview

Lithography hotspot detection is an important problem in semiconductor physical design. A lithography hotspot is a layout pattern that may cause manufacturing failures such as bridging or pinching.

A major challenge in hotspot detection is the severe imbalance between hotspot and non-hotspot samples in the ICCAD-12 benchmarks. Conventional classification training can become dominated by the large number of easy non-hotspot examples.

This project investigates whether training strategies that emphasize difficult and minority samples can improve hotspot detection performance.

The study uses a common lightweight convolutional neural network (CNN) and compares:

1. Binary Cross-Entropy (BCE)
2. Focal Loss
3. Online Hard Example Mining (OHEM)
4. Focal Loss + OHEM

The five ICCAD-12 benchmarks are evaluated independently.

---

## Research Question

> Can a learning strategy that emphasizes difficult and minority samples improve lithography hotspot detection under the severe class imbalance of the ICCAD-12 benchmarks?

### Sub-questions

- Does Focal Loss improve hotspot detection compared with conventional BCE training?
- Does explicit Online Hard Example Mining provide additional benefits?
- Does combining Focal Loss with OHEM provide complementary improvements?
- How do the methods affect the precision–recall and specificity trade-off across different levels of class imbalance?

---

## Experimental Methods

| Method | Model | Loss | Hard-example selection |
|---|---|---|---|
| BCE | Lightweight CNN | Binary Cross-Entropy | None |
| Focal | Lightweight CNN | Focal Loss | None |
| OHEM | Lightweight CNN | BCE | OHEM |
| Focal + OHEM | Lightweight CNN | Focal Loss | OHEM |

The primary comparison is **BCE vs Focal Loss across all five benchmarks**.

OHEM and Focal+OHEM provide additional hard-example learning comparisons for Benchmarks 4 and 5.

---

## Dataset

The experiments use the five ICCAD-12 lithography hotspot benchmarks supplied for the assignment.

### Dataset distribution

| Benchmark | Train HS | Train NHS | Test HS | Test NHS |
|---|---:|---:|---:|---:|
| B1 | 99 | 340 | 226 | 4,679 |
| B2 | 174 | 5,285 | 498 | 41,298 |
| B3 | 909 | 4,643 | 1,808 | 46,333 |
| B4 | 95 | 4,452 | 177 | 31,890 |
| B5 | 26 | 2,716 | 41 | 19,327 |

Where:

- **HS** = Hotspot
- **NHS** = Non-hotspot

The strong class imbalance makes Balanced Accuracy particularly important.

The dataset itself is not included in this repository.

---

## Dataset Directory Structure

Place the supplied ICCAD-12 dataset locally using the following structure:

```text
iccad-official/
├── iccad1/
│   ├── train/
│   │   ├── train_hs/
│   │   └── train_nhs/
│   └── test/
│       ├── test_hs/
│       └── test_nhs/
│
├── iccad2/
├── iccad3/
├── iccad4/
└── iccad5/
