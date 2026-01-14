# KenLM Training Guide for Tibetan Language Model

## Overview

This guide explains how to train a KenLM n-gram language model on Tibetan text for quality classification in the data cleaning pipeline.

## Repository

**Official KenLM Repository:** https://github.com/kpu/kenlm

## Installation

### Option 1: From Source (Recommended for Custom Builds)

```bash
# Clone the repository
git clone https://github.com/kpu/kenlm.git
cd kenlm

# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake libboost-all-dev libbz2-dev liblzma-dev

# macOS with Homebrew
brew install cmake boost

# Build KenLM
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Install Python bindings
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Option 2: pip (Simpler)

```bash
pip install kenlm
```

## Training a Tibetan Language Model

### Step 1: Prepare Training Data

Your training corpus should be:
- UTF-8 encoded
- One sentence per line
- Clean, high-quality "Gold Standard" Tibetan text (~15GB recommended)

```bash
# Combine all gold standard files
cat gold_data/*.txt > training_corpus.txt

# Remove empty lines and normalize
sed '/^$/d' training_corpus.txt > clean_corpus.txt
```

### Step 2: Train the Model

```bash
# Navigate to KenLM build directory
cd kenlm/build/bin

# Train a 5-gram model (recommended for Tibetan)
./lmplz -o 5 -S 80% -T /tmp < /path/to/clean_corpus.txt > tibetan_5gram.arpa

# Parameters:
# -o 5     : 5-gram order
# -S 80%   : Use 80% of available RAM
# -T /tmp  : Temporary directory for sorting
```

### Step 3: Convert to Binary Format

Binary format loads much faster and uses less memory:

```bash
./build_binary tibetan_5gram.arpa tibetan_model.bin

# For even faster loading (trie structure):
./build_binary trie tibetan_5gram.arpa tibetan_model.bin
```

### Step 4: Validate the Model

```python
import kenlm

model = kenlm.Model('tibetan_model.bin')

# Test with sample sentences
test_sentences = [
    "བོད་སྐད་ནི་བོད་པའི་སྐད་ཡིག་རེད།",  # Clean Tibetan
    "asdf jkl; random noise",              # Noise
]

for sent in test_sentences:
    ppl = model.perplexity(sent)
    print(f"Sentence: {sent[:30]}... | PPL: {ppl:.2f}")
```

## Quality Classification Thresholds

Based on empirical testing with Tibetan OCR data:

| Class | Perplexity Range | Description |
|-------|------------------|-------------|
| A | PPL ≤ 100 | High quality, clean text |
| B | 100 < PPL ≤ 500 | Medium quality, minor errors |
| C | PPL > 500 | Low quality, OCR noise |

## Advanced: Training Tips for Tibetan

### Syllable-Based Tokenization

Tibetan is best tokenized by syllables (tsek-separated):

```python
def tokenize_tibetan(text):
    """Tokenize Tibetan text by syllable boundaries."""
    return text.replace('་', ' ་ ').replace('།', ' །')
```

### Handling Mixed Scripts

Pre-filter training data to ensure high Tibetan content ratio:

```python
import re

def tibetan_ratio(text):
    tib_chars = len(re.findall(r'[\u0f00-\u0fff]', text))
    return tib_chars / max(len(text), 1)

# Only use sentences with >80% Tibetan characters
clean_lines = [l for l in lines if tibetan_ratio(l) > 0.8]
```

### Optimal Parameters

| Parameter | Recommended Value | Rationale |
|-----------|-------------------|-----------|
| N-gram order | 5 | Captures Tibetan phrase patterns |
| Pruning | None for <50GB | Preserve rare syllable combinations |
| Smoothing | Modified Kneser-Ney (default) | Best for morphologically rich languages |

## Troubleshooting

### "Model file too large"

Use quantization:

```bash
./build_binary -q 8 -b 8 trie tibetan_5gram.arpa tibetan_model.bin
```

### "Out of memory during training"

Reduce memory usage:

```bash
./lmplz -o 5 -S 4G -T /path/to/large/tmp < corpus.txt > model.arpa
```

### "Python binding crashes"

Ensure the binary was built with the same architecture:

```bash
pip uninstall kenlm
pip install --no-cache-dir kenlm
```

## References

- KenLM Paper: https://kheafield.com/papers/avenue/kenlm.pdf
- GitHub: https://github.com/kpu/kenlm
- Documentation: https://kheafield.com/code/kenlm/
