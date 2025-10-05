import sys
import logging
import os
import datasets
from datasets import load_dataset
import torch
import transformers
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig


# run training:
# accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 sft.py
# tmux new -d -s training "accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 sft.py"

###################
# Hyper-parameters
###################
processed_dataset_path = "../datasets/tulu-v3"
model_path = "../models/FlexiDepth-Llama-3-8B-Instruct"
output_dir = "./checkpoints"

training_config = {
    "do_eval": False,
    "learning_rate": 1e-4,
    "log_level": "info",
    "logging_steps": 10,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1.0,
    "max_steps": -1,
    "output_dir": output_dir,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 1000,
    "save_total_limit": 1,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.03,
    "dataloader_drop_last": True,
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "max_length": 2048,
}
sft_config = SFTConfig(**training_config)

################
# Model Loading
################
checkpoint_path = model_path
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="sdpa",
    dtype=torch.bfloat16,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)

# Freeze all parameters except router and router_proj
print("Freezing all parameters except router and router_proj...")
for name, param in model.named_parameters():
    if 'router' in name.lower():
        param.requires_grad = True
        print(f"Keeping gradient for: {name}")
    else:
        param.requires_grad = False

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

##################
# Data Loading
##################

dataset_dict = datasets.load_from_disk(processed_dataset_path)
train_dataset = dataset_dict['train']

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

# Start training
train_result = trainer.train()

# Log final metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)