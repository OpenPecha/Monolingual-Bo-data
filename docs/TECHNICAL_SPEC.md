# Technical Specification: Monolingual Tibetan Data Cleaning Pipeline

## 1. Project Overview

This document outlines the architecture and implementation of a large-scale data cleaning pipeline designed to process **400GB** of raw Tibetan digital artifacts. The pipeline reduces noise, removes duplicates, and classifies text quality, resulting in a **45GB** high-quality dataset containing **5 billion tokens**.

---

## 2. Technical Architecture

The pipeline follows a five-step sequential process to move from raw, multi-format files to categorized, clean text.

### Step 1: File Format Conversion

* **Goal:** Standardize all inputs into UTF-8 encoded `.txt`.
* **Supported Formats:** `.doc`, `.docx`, `.pdf`, `.html`, `.rtf`.
* **Tooling:** `OpenPecha/TibCleaner`.
* **Validation:** post-conversion scripts verify UTF-8 encoding integrity and log failed conversions for manual review.

### Step 2: Unicode Filtering

* **Methodology:** Byte-level scanning to identify non-Unicode encodings.
* **Criteria:** Maintain content within the Tibetan Unicode block (`U+0F00`–`U+0FFF`).
* **Mixed Content:** Files with <5% Tibetan Unicode are discarded; others are stripped of non-Unicode strings.

### Step 3: Document De-duplication

To prevent model over-fitting, we utilize **MinHash Locality Sensitive Hashing (LSH)**.

* **Parameters:** Threshold τ=0.85, n=128 permutations.
* **Granularity:** Syllable-level shingling (splitting by the Tibetan *tsek* ་).

### Step 4: Linguistic Filtering & Segmentation

* **Tool:** Modified `Botok` Library.
* **Syllable Filter:** Sentences with <4 syllables are removed.
* **Language Filter:** Sentences must contain >80% Tibetan script characters.

### Step 5: OCR Quality Classification

We use a **KenLM 5-gram model** trained on 15GB of "Gold Standard" clean data to calculate sentence perplexity (PPL).

---

## 3. Dataset Statistics

| Metric | Value |
| --- | --- |
| **Raw Input Size** | 400 GB |
| **Cleaned Output Size** | 45 GB |
| **Total Sentences** | 357 Million |
| **Total Tokens** | 5 Billion |
| **Classification Accuracy** | 90% |

---

## 4. Pipeline Execution Order

```
raw_data/ 
    │
    ▼ [Step 1: format_converter.py]
converted/
    │
    ▼ [Step 2: unicode_filter.py]
filtered/
    │
    ▼ [Step 3: deduplicator.py]
deduplicated/
    │
    ▼ [Step 4: segmenter.py]
segmented/
    │
    ▼ [Step 5: classifier.py]
classified/{A,B,C}/
```
