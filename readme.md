# NuExtract-2.0 SFT: Document Information Extraction

This repository contains the implementation and evaluation of Supervised Fine-Tuning (SFT) on the **NuExtract-2.0-4B** model. The project focuses on structured information extraction from identity documents (specifically Arizona Driver's Licenses), transforming raw pixels into standardized JSON formats.

## Model Overview
**NuExtract-2.0-4B** is a Vision-Language Model (VLM) designed for structured document extraction. It utilizes a `Qwen2.5-VL` base and is optimized to take a JSON template and an image as input, returning the extracted information "verbatim" into the specified schema. https://huggingface.co/numind/NuExtract-2.0-4B

---

## Dataset Description
The model was trained and evaluated using the **IdNet AZ Dataset**. https://zenodo.org/records/13852757
* **Source:** Arizona (AZ) synthetic/positive identity document samples.
* **Volume:** Approximately 6,000 document pairs (images and metadata).
* **Holdout Set:** 10% of the data (597 images) was reserved for zero-shot baseline testing and final post-SFT evaluation.
* **Label Mapping:** Raw dataset keys were mapped from the source format to a standardized DL schema to ensure consistency.
    * *Mapping:* `first_name` → `first_name`, `birthday` → `DOB`, `issue_date` → `ISS`, `license_number` → `DLN`, etc.

---

## SFT Process
To improve the model's accuracy on complex fields like Issue Dates (ISS) and License Numbers (DLN)—which the base model struggled with—we performed Supervised Fine-Tuning.

### 1. Pre-processing
* **Optimization:** Used `aria2c` for high-speed dataset retrieval and extraction.
* **Normalization:** Developed a custom evaluation script to normalize dates (handling MM/DD/YYYY vs YYYY-MM-DD) and text (case-folding and punctuation removal) to ensure accuracy metrics reflect semantic correctness rather than formatting discrepancies.

### 2. Training Configuration
* **Hardware:** Optimized for NVIDIA L4 GPUs.
* **Precision:** Loaded in `bfloat16` to leverage 22.5GB VRAM efficiently.
* **Attention:** `flash_attention_2` enabled for faster inference and training throughput.
* **Template-Guided:** The model was trained to follow a strict JSON schema template to ensure output consistency.

---

## Evaluation Results

The following tables compare the performance of the **Stock NuExtract-2.0-4B** model against the **Fine-Tuned (SFT)** version on the 597-image holdout set.

### Baseline (Stock Model)
The stock model performed well on common fields (Sex, WGT) but failed significantly on document-specific IDs and specific date formats.

| FIELD | ACCURACY | CORRECT / TOTAL |
| :--- | :--- | :--- |
| first_name | 92.29% | 551 / 597 |
| last_name | 91.46% | 546 / 597 |
| address | 98.32% | 587 / 597 |
| DOB | 99.83% | 596 / 597 |
| SEX | 100.00% | 597 / 597 |
| CLASS | 97.32% | 581 / 597 |
| ISS | 12.40% | 74 / 597 |
| EXP | 86.43% | 516 / 597 |
| HGT | 45.23% | 270 / 597 |
| WGT | 100.00% | 597 / 597 |
| EYES | 86.60% | 517 / 597 |
| DLN | 8.21% | 49 / 597 |
| **TOTAL** | **76.51%** | **5,481 / 7,164** |

### After SFT (Fine-Tuned)
Fine-tuning resolved the field-mapping confusion and significantly improved the extraction of license numbers and height/weight attributes.

| FIELD | ACCURACY | CORRECT / TOTAL |
| :--- | :--- | :--- |
| first_name | 100.00% | 597 / 597 |
| last_name | 99.33% | 593 / 597 |
| address | 99.50% | 594 / 597 |
| DOB | 99.83% | 596 / 597 |
| SEX | 100.00% | 597 / 597 |
| CLASS | 100.00% | 597 / 597 |
| ISS | 99.66% | 595 / 597 |
| EXP | 99.66% | 595 / 597 |
| HGT | 100.00% | 597 / 597 |
| WGT | 100.00% | 597 / 597 |
| EYES | 100.00% | 597 / 597 |
| DLN | 99.83% | 596 / 597 |
| **TOTAL** | **99.82%** | **7,151 / 7,164** |

---

## Installation & Usage

### Dependencies
```bash
pip install qwen_vl_utils bitsandbytes accelerate flash-attn --no-build-isolation
apt-get install -y aria2
