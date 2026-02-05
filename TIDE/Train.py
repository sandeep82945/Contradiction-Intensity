import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# CONFIG
# =========================
BASE_MODEL = "meta-llama/Llama-3-8B-Instruct"
TRAIN_JSONL = "data/jsonl_train.jsonl"
OUTPUT_DIR = "output_llama3"
VAL_SPLIT_RATIO = 0.1

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# DATASET
# =========================
dataset = load_dataset("json", data_files=TRAIN_JSONL, split="train")
dataset_split = dataset.train_test_split(test_size=VAL_SPLIT_RATIO, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# =========================
# CHAT TEMPLATE
# =========================
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

# =========================
# LoRA CONFIG (CORRECT FOR LLAMA-3)
# =========================
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

# =========================
# TRAINING CONFIG (FIXED)
# =========================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # batching
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 8
    per_device_eval_batch_size=1,

    # training length
    num_train_epochs=5,

    # optimization (FIXED LR)
    learning_rate=5e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    # evaluation / saving
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # precision
    bf16=True,
    fp16=False,

    # CRITICAL FIX: assistant-only loss
    completion_only_loss=True,

    # sequence handling
    max_length=4096,

    # logging
    report_to="none",
)

# =========================
# MODEL
# =========================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# =========================
# TRAINER
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    processing_class=tokenizer,
    peft_config=peft_config
)

# =========================
# TRAIN
# =========================
trainer.train()

# =========================
# SAVE
# =========================
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
