#!/usr/bin/env python3
"""
Simple SFT training script using HuggingFace Transformers.
This bypasses the complex ROLL pipeline and uses standard HF training.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from transformers import TrainerCallback


@dataclass
class SFTConfig:
    model_name_or_path: str = field(default="Qwen/Qwen3-4B-Instruct-2507")
    data_path: str = field(default="data/sft/sokoban_train_io.jsonl")
    val_data_path: str = field(default="data/sft/sokoban_val_io.jsonl")
    output_dir: str = field(default="./output/sft_simple")
    max_steps: int = field(default=2000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=5e-5)
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    max_length: int = field(default=4096)
    num_train_epochs: int = field(default=1)


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['instruction'])):
        text = f"User: {examples['instruction'][i]}\nAssistant: {examples['output'][i]}"
        output_texts.append(text)
    return output_texts


def main():
    # Parse arguments
    config = SFTConfig()
    if len(sys.argv) > 1:
        config.data_path = sys.argv[1]
    if len(sys.argv) > 2:
        config.val_data_path = sys.argv[2]
    if len(sys.argv) > 3:
        config.output_dir = sys.argv[3]

    print(f"Loading model from {config.model_name_or_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load dataset
    print(f"Loading training data from {config.data_path}...")
    dataset = load_dataset("json", data_files=config.data_path, split="train")

    # Tokenize
    def preprocess_function(examples):
        return tokenizer(
            [f"User: {i}\nAssistant: {o}" for i, o in zip(examples['instruction'], examples['output'])],
            max_length=config.max_length,
            truncation=True,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to=["tensorboard"],
        save_safetensors=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {config.output_dir}...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
