
---

# 🧠 PRISM — Hallucination Detection System for AI-Generated Images

PRISM is a multimodal hallucination detection system designed to identify inconsistencies between AI-generated images and their corresponding text prompts.

This project focuses on analyzing prompt-image alignment using pretrained vision-language models (VLMs) and structured evaluation pipelines.

---

## 🚀 Project Overview

Large image generation models often produce outputs that do not fully align with the original prompt. These inconsistencies are called **hallucinations**.

PRISM aims to:

* Detect semantic mismatches between prompts and generated images
* Analyze object presence and attribute correctness
* Score hallucination severity
* Provide an interactive UI for evaluation

---

## 🏗️ Project Architecture

```
PRISM/
│
├── data/
│   ├── prepare_dataset.py
│   ├── diffusiondb_subset/
│
├── models/
│   ├── vlm_model.py
│   ├── scoring.py
│
├── database/
│   ├── embeddings.db
│
├── app.py (Streamlit UI)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

We use:

* **DiffusionDB**
  Source: Hugging Face dataset
  Subset: `large_random_1k`

Dataset includes:

* Prompt text
* Generated image
* Metadata

You can either:

1. Stream from Hugging Face
2. Use an already downloaded local dataset (recommended for faster performance)

---

## 🧠 Models Used

* Vision-Language Model (VLM) for prompt-image alignment
* CLIP-based similarity scoring
* PEFT-compatible models (requires `peft >= 0.17.0`)

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/samdani0511/prism.git
cd prism
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv sic
source sic/bin/activate  # Linux / Mac
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you face PEFT version issues:

```bash
pip install --upgrade peft
```

Required:

```
peft >= 0.17.0
```

---

## 🗂️ Dataset Preparation

To prepare dataset:

```bash
python data/prepare_dataset.py
```

This will:

* Download DiffusionDB subset
* Extract prompt-image pairs
* Store locally for offline usage

---

## ▶️ Running the Application

### Option 1: Using Downloaded Dataset

```bash
streamlit run app.py
```

Choose:

> ✅ Use Existing Database

This loads locally stored embeddings and images.

---

### Option 2: Streaming from Hugging Face

Modify configuration to enable streaming mode.

---

## 🔄 System Flow

1. User selects image + prompt
2. VLM encodes image and text
3. Similarity score computed
4. Object-level mismatch analysis performed
5. Hallucination score generated
6. Output displayed in Streamlit dashboard

---

## 📊 Hallucination Scoring Logic

PRISM evaluates:

* Object Presence Error
* Attribute Mismatch
* Spatial Relationship Errors
* Semantic Consistency

Final Score:

```
Hallucination Score = 1 - Similarity(prompt, image)
```

Higher score → Higher hallucination probability

---

## 🖥️ Streamlit UI Features

* Upload image + prompt
* Select from dataset samples
* Use local embeddings database
* Real-time hallucination score
* Visual similarity metrics

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* CLIP
* Streamlit
* SQLite (for embeddings DB)

---


## 🎯 Future Improvements

* Fine-tuned hallucination classifier
* Object grounding with bounding boxes
* Explainable hallucination detection
* Benchmark against GPT-4V
* Deploy as web service (FastAPI + React frontend)

---

## 👨‍💻 Author

Shaik Mohammed Samdani
AIML Engineer | Researcher | Builder

