This is the official repository for the paper:

# 'When Reviews Disagree: Fine-Grained Contradiction Analysis in Scientific Peer Reviews'

We introduce a fine-grained formulation of reviewer contradiction analysis that moves beyond binary detection by explicitly identifying contradiction evidence spans and assigning graded disagreement intensity scores.

The repository contains the code for our structured multi-agent framework **IMPACT** and its distilled, efficient counterpart **TIDE**.

---

## 🌐 Live Demo & Testing

We provide a web-based interface for you to test or check our IMPACT framework in real-time. You can explore the system's capabilities using the demo link below:

* 
**Demo Website**: [https://peer-review-ai-contradiction-detector.hf.space/](https://peer-review-ai-contradiction-detector.hf.space/) 


* **Sample Inputs**: Ready-to-use sample real review inputs for the demo have been added to the `Data/` folder.

---

## Getting Started

### Step 1: Install Dependencies

Install all required libraries for model training, inference, and evaluation.

```bash
pip install -r requirements.txt

```

### Step 2: Training (TIDE)

Train **TIDE**, a Small Language Model (SLM) distilled from IMPACT's deliberative reasoning traces.

```bash
python TIDE/Train.py

```

* The student model is fine-tuned using LoRA adapters on Llama-3-8B-Instruct.


* Training is performed for 5 epochs with a fixed learning rate of .



### Step 3: Inference

Run prediction on your test data using the trained student model.

```bash
python TIDE/Inference.py

```

* TIDE predicts contradiction evidence, intensity labels, and reasoning in a single forward pass.



### Step 4: Multi-Agent Detection (IMPACT)

Run the full **IMPACT** framework involving aspect-conditioned evidence extraction and multi-agent debate.

```bash
python IMPACT/IMPACT_P.py

```

* This system coordinates an Aspect-Conditioned Evidence Agent (ACEA), two Deliberative Intensity Agents (DIAs), and an Adjudication Agent to resolve disagreements.



### Step 5: Evaluation

Evaluate model performance using Hungarian matching for evidence overlap and agreement metrics for intensity.

```bash
python Evaluate.py

```

* The script computes ROUGE-L for evidence alignment.


* It further reports Cohen's Kappa (), Spearman's (), and Kendall's () for intensity agreement.



---

## Data Formatting and Paths

Since this repository does not include the raw dataset, you must provide your own files in the `Data/` folder. The framework requires different data formats depending on the script being run.

### 1. Data Formats

#### **A. IMPACT & Evaluation Format (JSON)**

The **IMPACT** framework and **Evaluate.py** expect a JSON file where keys represent paper IDs, containing the full text of individual reviews.

```json
{
  "paper_id_001": {
    "Review_1_full": "The full text of review A...",
    "Review_2_full": "The full text of review B..."
  }
}

```

* **Keys**: Review text keys must start with `Review_` and end with `_full`.

#### **B. TIDE Format (Chat-Template JSONL)**

For training and inference, data must be in a `.jsonl` format following the Llama-3 chat-template structure.

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert at analyzing peer reviews..."},
    {"role": "user", "content": "TASK: ... Paper ID: 001\nReview A: ...\nReview B: ..."},
    {"role": "assistant", "content": "[{\"evidence\": [\"Quote A\", \"Quote B\"], \"intensity_reasoning\": \"...\", \"intensity\": 3}]"}
  ]
}

```

* **System/User Role**: Includes the expert persona, detailed task instructions, the scoring rubric, and the raw reviews.

**Assistant Role**: Contains the structured JSON list of identified contradictions.



### 2. Updating File Paths

You **must** update the global path variables in the configuration section of each file to match the directory structure of your local environment:

* **`TIDE/Train.py`**: Update `TRAIN_JSONL` and `OUTPUT_DIR`.
* **`TIDE/Inference.py`**: Update `ADAPTER_CHECKPOINT`, `TEST_DATA_PATH`, and `OUTPUT_FOLDER`.
* **`Evaluate.py`**: Update `GT` (Ground Truth) and `PRED` (Model Predictions) in the `main` block.
* **`IMPACT/IMPACT_P.py`**: Update `GROUND_TRUTH_FILE` and `OUTPUT_FILE`.

---
