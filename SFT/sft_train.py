import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import os
from sft_trainer import *
from dataset_loader import load_data, get_available_datasets
import torch.distributed as dist
import random
import numpy as np
from datetime import datetime


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-s1", help="Job Name")
    parser.add_argument(
        "--train_data", 
        type=str, 
        default="simplescaling/s1K", 
        choices=get_available_datasets(),
        help=f"Dataset to use for training. Available choices: {get_available_datasets()}"
    )
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )
    parser.add_argument(
        "--max_train_samples", 
        type=int, 
        default=None, 
        help="Maximum number of training samples to use. If None, use all available samples."
    )
    parser.add_argument(
        "--use_fast_preprocessing", 
        action="store_true", 
        default=True,
        help="Use fast parallel preprocessing (default: True)"
    )

    return parser.parse_args()


# Model loading with LoRA integration
def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    # Load model
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Applying LoRA model
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)  # Cast fp32 lora params to bf16

    return tokenizer, model


# Dataset loading is now handled in dataset_loader.py


# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset = load_data(args, tokenizer)

    name = args.job_name + "_" + args.train_data.split("/")[-1] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=2,
        save_steps=100,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        # report_to="wandb" if not args.debugging else "none",
        report_to="swanlab" if not args.debugging else "none",
        run_name=name,
        remove_unused_columns=False,
    )

    # Create optimizer and scheduler
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size * args.grad_accum_steps * torch.cuda.device_count())
    )
    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    init_seed(42)
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args)

    # Train the model
    train_model(args, tokenizer, model)
