"""
Fine-tunes a base LLM on GTA VC SCM data.
Uses HuggingFace transformers + PEFT (LoRA) for efficient fine-tuning.
Recommended base models:
  - codellama/CodeLlama-7b-Instruct-hf  (best for code/script tasks)
  - mistralai/Mistral-7B-Instruct-v0.2   (good general + code)
  - microsoft/phi-2                       (small, fast, decent code)

Requirements:
  pip install transformers peft trl datasets accelerate bitsandbytes torch
"""

import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch

# ── Config ──────────────────────────────────────────────
BASE_MODEL = "codellama/CodeLlama-7b-Instruct-hf"
DATASET_PATH = "data/processed/training_pairs.jsonl"
OUTPUT_DIR = "models/gtavc_scm_lora"
MAX_SEQ_LEN = 2048

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

TRAINING_ARGS = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_steps=200,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    save_total_limit=3,
)
# ────────────────────────────────────────────────────────

def load_dataset(path: str) -> Dataset:
    """Load JSONL training pairs and format for SFTTrainer"""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get('messages', [])
            text = format_chat(messages)
            if text:
                records.append({'text': text})
    return Dataset.from_list(records)


def format_chat(messages: list) -> str:
    """
    Format chat messages into CodeLlama B_INST/E_INST instruction format.

    Correct format merges the system block with the FIRST user turn:
        [INST] <<SYS>>
        {system}
        <</SYS>>

        {first_user_message} [/INST] {assistant_response}

    Subsequent user turns:
        [INST] {user_message} [/INST] {assistant_response}
    """
    result = ""
    system_content = ""
    system_merged = False  # whether system has been merged into a user [INST] block

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            system_content = content

        elif role == 'user':
            if system_content and not system_merged:
                # Merge system + first user turn into one [INST] block
                result += f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{content} [/INST] "
                system_merged = True
            else:
                result += f"[INST] {content} [/INST] "

        elif role == 'assistant':
            result += f"{content}\n"

    return result.strip()


def load_model_and_tokenizer(model_name: str):
    """Load model with 4-bit quantization for memory efficiency"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def train():
    print(f"[Finetune] Loading base model: {BASE_MODEL}")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL)

    print("[Finetune] Applying LoRA adapters...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    print(f"[Finetune] Loading dataset from {DATASET_PATH}")
    dataset = load_dataset(DATASET_PATH)
    print(f"[Finetune] Dataset size: {len(dataset)} examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TRAINING_ARGS,
        packing=False,
    )

    print("[Finetune] Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[Finetune] Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
