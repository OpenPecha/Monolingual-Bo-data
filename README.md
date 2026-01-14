# Monolingual Tibetan Data Cleaning Pipeline

A large-scale data cleaning pipeline for processing raw Tibetan digital artifacts. This pipeline reduces noise, removes duplicates, and classifies text quality to produce high-quality training data for Tibetan language models.

## Overview

| Metric | Value |
|--------|-------|
| Raw Input Size | 400 GB |
| Cleaned Output Size | 45 GB |
| Total Sentences | 357 Million |
| Total Tokens | 5 Billion |
| Classification Accuracy | 90% |

## Pipeline Architecture

```
raw_data/
    |
    v [Step 1: Format Conversion]
converted/          -> UTF-8 .txt files
    |
    v [Step 2: Unicode Filtering]
filtered/           -> Tibetan content only
    |
    v [Step 3: Deduplication]
deduplicated/       -> Unique documents
    |
    v [Step 4: Segmentation]
segmented/          -> Clean sentences
    |
    v [Step 5: Classification]
classified/
    |-- A/          -> High quality (PPL <= 100)
    |-- B/          -> Medium quality (100 < PPL <= 500)
    +-- C/          -> Low quality (PPL > 500)
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Monolingual-Bo-data.git
cd Monolingual-Bo-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install KenLM (required for Step 5)
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Run the Full Pipeline

```bash
# Run all steps
python scripts/run_pipeline.py \
    --input ./data/raw \
    --output ./data/output \
    --model ./models/tibetan_model.bin \
    --workers 8

# Run specific steps only
python scripts/run_pipeline.py \
    --input ./data/raw \
    --output ./data/output \
    --steps 1,2,3
```

### Run Individual Steps

```bash
# Step 1: Format Conversion
python scripts/step1_format_converter.py \
    --input ./data/raw \
    --output ./data/converted \
    --workers 8

# Step 2: Unicode Filtering
python scripts/step2_unicode_filter.py \
    --input ./data/converted \
    --output ./data/filtered \
    --workers 8 \
    --min-ratio 0.05

# Step 3: Deduplication
python scripts/step3_deduplicator.py \
    --input ./data/filtered \
    --output ./data/deduplicated \
    --workers 8 \
    --threshold 0.85

# Step 4: Segmentation
python scripts/step4_segmenter.py \
    --input ./data/deduplicated \
    --output ./data/segmented \
    --workers 8 \
    --min-syllables 4 \
    --min-tibetan-ratio 0.8

# Step 5: Classification
python scripts/step5_classifier.py \
    --input ./data/segmented \
    --output ./data/classified \
    --model ./models/tibetan_model.bin \
    --workers 8 \
    --threshold-a 100 \
    --threshold-b 500
```

## Project Structure

```
Monolingual-Bo-data/
|-- README.md
|-- requirements.txt
|-- docs/
|   |-- TECHNICAL_SPEC.md      # Detailed technical documentation
|   +-- KENLM_TRAINING.md      # KenLM model training guide
|-- scripts/
|   |-- run_pipeline.py        # Main orchestrator
|   |-- step1_format_converter.py
|   |-- step2_unicode_filter.py
|   |-- step3_deduplicator.py
|   |-- step4_segmenter.py
|   +-- step5_classifier.py
|-- data/
|   |-- raw/                   # Input: raw documents
|   |-- converted/             # Step 1 output
|   |-- filtered/              # Step 2 output
|   |-- deduplicated/          # Step 3 output
|   |-- segmented/             # Step 4 output
|   +-- classified/            # Step 5 output (A/B/C)
|-- models/
|   +-- tibetan_model.bin      # KenLM language model
+-- logs/
    +-- *.log                  # Processing logs
```

## Pipeline Steps

### Step 1: File Format Conversion

Converts various document formats to UTF-8 encoded plain text.

Supported formats: .doc, .docx, .pdf, .html, .htm, .rtf, .txt

```bash
python scripts/step1_format_converter.py --input raw/ --output converted/
```

### Step 2: Unicode Filtering

Filters content to maintain Tibetan Unicode block (U+0F00 to U+0FFF).

- Files with less than 5% Tibetan content are discarded
- Non-Tibetan strings are stripped from remaining files

```bash
python scripts/step2_unicode_filter.py --input converted/ --output filtered/ --min-ratio 0.05
```

### Step 3: Document Deduplication

Uses MinHash Locality Sensitive Hashing (LSH) for efficient duplicate detection.

- Threshold: 0.85 (Jaccard similarity)
- Permutations: n = 128
- Shingling: Syllable-level (Tibetan tsek)

```bash
python scripts/step3_deduplicator.py --input filtered/ --output deduplicated/ --threshold 0.85
```

### Step 4: Linguistic Filtering and Segmentation

Uses the Botok library (https://github.com/OpenPecha/Botok) for Tibetan tokenization.

Filters applied:
- Syllable filter: Sentences with less than 4 syllables removed
- Language filter: Sentences must contain more than 80% Tibetan characters

```bash
python scripts/step4_segmenter.py --input deduplicated/ --output segmented/ --min-syllables 4
```

### Step 5: OCR Quality Classification

Uses KenLM 5-gram perplexity scoring to classify sentence quality.

| Class | Perplexity Range | Description |
|-------|------------------|-------------|
| A | PPL <= 100 | High quality, clean text |
| B | 100 < PPL <= 500 | Medium quality, minor errors |
| C | PPL > 500 | Low quality, OCR noise |

```bash
python scripts/step5_classifier.py --input segmented/ --output classified/ --model model.bin
```

## KenLM Model Training

**KenLM Repository:** https://github.com/kpu/kenlm

To train a KenLM language model for quality classification:

```bash
# Clone KenLM
git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir build && cd build
cmake .. && make -j$(nproc)

# Train 5-gram model
./bin/lmplz -o 5 -S 80% < gold_corpus.txt > tibetan.arpa

# Convert to binary
./bin/build_binary tibetan.arpa tibetan_model.bin
```

See docs/KENLM_TRAINING.md for detailed instructions.

## Configuration Options

### Pipeline Runner (run_pipeline.py)

| Option | Default | Description |
|--------|---------|-------------|
| --input | required | Input directory with raw documents |
| --output | required | Base output directory |
| --model | None | KenLM model path for classification |
| --workers | CPU count | Number of parallel workers |
| --steps | 1,2,3,4,5 | Steps to run (comma-separated) |
| --threshold | 0.85 | Deduplication similarity threshold |
| --threshold-a | 100 | Class A perplexity ceiling |
| --threshold-b | 500 | Class B perplexity ceiling |
| --dry-run | False | Print commands without executing |

## Performance

With 8 CPU cores and SSD storage:

| Step | Processing Speed | Notes |
|------|------------------|-------|
| Step 1 | ~500 files/min | Depends on file formats |
| Step 2 | ~2,000 files/min | I/O bound |
| Step 3 | ~1,000 files/min | Memory-intensive for LSH index |
| Step 4 | ~1,500 files/min | Botok initialization overhead |
| Step 5 | ~3,000 files/min | Model inference parallelizes well |

## Dependencies

- **Botok** (https://github.com/OpenPecha/Botok) - Tibetan word tokenization
- **KenLM** (https://github.com/kpu/kenlm) - N-gram language modeling
- **datasketch** (https://github.com/ekzhu/datasketch) - MinHash LSH implementation
- **pdfplumber** (https://github.com/jsvine/pdfplumber) - PDF text extraction
- **python-docx** (https://github.com/python-openxml/python-docx) - DOCX processing

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- OpenPecha (https://github.com/OpenPecha) for Tibetan NLP tools
- KenLM team for the efficient language model implementation
