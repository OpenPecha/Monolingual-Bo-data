# Data Cleaning Process Documentation

## Overview
<img width="1248" alt="Untitled" src="https://github.com/OpenPecha/Monolingual-Bo-data/assets/72848416/e63bc7b5-41b3-4d78-b8e5-463b1c9a5443">

This documentation outlines the comprehensive data cleaning process undertaken for a large monolingual dataset primarily consisting of Tibetan texts. The dataset initially contained various file formats totaling approximately 400GB, including documents from the Openpecha project. Our objective was to standardize the cleaning process to ensure reliability and replicability.

## Data Conversion to Text

### Step 1: Converting Non-Text Files to Text

- **Objective**: Convert all non-text files (.doc, .docx, .pdf, .html, .rtf, etc.) into text (.txt) format.
- **Tools**: Scripts from the [OpenPecha/TibCleaner repository](https://github.com/OpenPecha/TibCleaner/tree/main/src/tibcleaner) on GitHub.
- **Process**: Utilized existing scripts to systematically convert various file formats to UTF-8 encoded text files.

### Step 2: Unicode Filtering

- **Objective**: Filter out all non-Unicode text to ensure dataset uniformity.
- **Process**: Applied scripts to identify and remove files or content not encoded in Unicode, focusing on maintaining Tibetan texts and eliminating other languages.

### Step 3: De-duplication

- **Objective**: Remove duplicate documents on a document level.
- **Tools**: Utilized the `datasketch` library, specifically MinHash LSH for efficient large-scale deduplication. [Datasketch MinHash LSH documentation](https://ekzhu.com/datasketch/lsh.html#minhash-lsh).
- **Process**: Implemented MinHash LSH to identify and remove duplicate documents, enhancing dataset uniqueness.

## Text Cleaning and Language Filtering

### Step 4: Non-Tibetan Text Filtering

- **Objective**: Isolate Tibetan texts by removing non-Tibetan words and sentences.
- **Tools**: Utilized the [Botok library](https://github.com/OpenPecha/Botok), specifically modified for our requirements. Botok is designed for word tokenization and sentence segmentation of Tibetan texts, among other features.
- **Process**: Adapted the Botok library to not only segment texts into sentences but also to remove non-Tibetan words or sentences from the dataset. This modification was crucial in filtering out non-Tibetan texts, ensuring the dataset predominantly contained clean Tibetan sentences.

### Step 5: OCR Data Cleaning

- **Objective**: Clean OCRed data which contained significant noise.
- **Research**: Investigated the use of RoBERTa and a method from Meta using KenLM for quality assessment based on perplexity scores ([Meta's approach](https://arxiv.org/pdf/1911.00359.pdf)).
- **Tools**: Trained a KenLM 5-gram model on 15GB of cleaned data to classify sentences into three quality classes (A, B, C) based on perplexity scores.
- **Process**: Assigned quality scores to sentences, organizing them into folders based on their classification to segregate data by quality.

## Final Dataset Composition

- **Result**: The cleaning process yielded approximately 45GB of high-quality text, encompassing around 357 million sentences and 5 billion tokens.
- **Quality Assessment**: Achieved a classification accuracy of 90% with the KenLM model. However, challenges such as undetected noise (e.g., random Tibetan numbers, repeating characters) were noted for future improvement.
## Downloading the Dataset Using AWS CLI

To download files from the S3 bucket using the AWS CLI, you can use the `aws s3 cp` command for individual files or `aws s3 sync` to download entire directories. Below are examples of how to download data from each quality class folder:
## Storing Cleaned Data on S3

The cleaned dataset is organized into three quality categories (A, B, C) and stored in corresponding folders on Amazon S3. Each folder contains a list of text files with the cleaned and classified text data.

- **S3 Bucket Path**: `s3://monolingual.data/`
- **Folder Structure**:
  - `A/`: Contains text files classified as highest quality (Class A).
  - `B/`: Contains text files classified as medium quality (Class B).
  - `C/`: Contains text files classified as lowest quality, mostly containing noise (Class C).

The dataset is publicly accessible, allowing for easy download and use.

### Prerequisites
Ensure you have the AWS CLI installed and configured with the necessary access permissions to the S3 bucket.

### Downloading Files from a Specific Quality Class

- **Class A (Highest Quality):**
  ```sh
  aws s3 sync s3://monolingual.data/A/ ./local_directory/A/ --no-sign-request
  ```

- **Class B (Medium Quality):**
  ```sh
  aws s3 sync s3://monolingual.data/B/ ./local_directory/B/ --no-sign-request
  ```

- **Class C (Lowest Quality, Mostly Noise):**
  ```sh
  aws s3 sync s3://monolingual.data/C/ ./local_directory/C/ --no-sign-request
  ```

Replace `./local_directory/` with the path to the directory on your local machine where you want the files to be downloaded.

## Notes and Observations

- There are still noice that kenlm failed to detect such as random tibetan numbers in sentences 
- Kenlm classification fails to detected repeating characters as noice and might show low perplexity score so there are sentences with repeating character
- Using the sentence segementation there are many sentences that are less then 4 syllables in the data 


