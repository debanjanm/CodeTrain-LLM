import os

import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper import create_prompt_squad_v2, print_trainable_parameters

def set_cuda_device(device_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return torch.cuda.is_available()

def load_model_and_tokenizer(model_path, use_fp16=True):
    print(f"Loading model and tokenizer from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def configure_model(model, target_modules):
    print("Configuring model...")
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce the number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model

def load_dataset_from_csv(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    data_frame = pd.read_csv(dataset_path)
    dataset = Dataset.from_pandas(data_frame)
    return dataset

def tokenize_dataset(dataset, tokenizer, create_prompt_fn):
    print("Tokenizing dataset...")
    mapped_dataset = dataset.map(lambda samples: tokenizer(create_prompt_fn(samples["context"], samples["question"], samples["text"])))
    return mapped_dataset

def train_model(model, train_dataset, tokenizer, output_dir, fp16=True):
    print("Training model...")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=100,
            learning_rate=1e-3,
            fp16=fp16,
            logging_steps=1,
            output_dir=output_dir,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # Save the fine-tuned model in the specified output directory
    trainer.save_model(output_dir)

if __name__ == "__main__":
    # Configuration
    model_path = "models/pre-trained/Cerebras-GPT-590M"
    target_modules = ["c_attn", "c_proj"]
    dataset_path = "datasets/squad_v2/train-squad.csv"
    output_model_dir = "models/fine-tuned/Cerebras-GPT-590M_QA"

    # CUDA Device
    cuda_device_id = 0
    if set_cuda_device(cuda_device_id):
        print(f"CUDA Device {cuda_device_id} is available.")
    else:
        print("CUDA is not available. Training will continue on CPU.")

    # Load Model and Tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, use_fp16=True)

    # Configure Model
    model = configure_model(model, target_modules)

    # Load Dataset
    dataset = load_dataset_from_csv(dataset_path)

    # Tokenize Dataset
    mapped_dataset = tokenize_dataset(dataset, tokenizer, create_prompt_squad_v2)

    # Train Model and Save Fine-Tuned Model
    train_model(model, mapped_dataset, tokenizer, output_model_dir, fp16=True)
