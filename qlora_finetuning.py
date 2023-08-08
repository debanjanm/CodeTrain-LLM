import os

import torch
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    pipeline,
)
from trl import SFTTrainer

from datasets import load_dataset


def set_cuda_device(device_map):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return torch.cuda.is_available()


def load_model_and_tokenizer(
    model_name,
    use_4bit,
    bnb_4bit_quant_type,
    bnb_4bit_compute_dtype,
    use_nested_quant,
    lora_alpha,
    lora_dropout,
    lora_r,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return tokenizer, model, peft_config


def configure_model(model, peft_config):
    model.config.peft_config = peft_config


def load_dataset_from_csv(data_file, column_names):
    return load_dataset(
        "csv", data_files={"train": data_file}, column_names=column_names
    )["train"]


def tokenize_dataset(dataset, tokenizer):
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["question"])):
            text = f"### Context: {example['context'][i]}\n ### Question: {example['question'][i]}\n ### Answer: {example['text'][i]}"
            output_texts.append(text)
        return output_texts

    return dataset.map(formatting_prompts_func, batched=True)


def train_model(trainer, train_dataset):
    trainer.train(train_dataset)


if __name__ == "__main__":
    # Configuration
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    use_4bit = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_compute_dtype = "float16"
    use_nested_quant = False
    lora_alpha = 16
    lora_dropout = 0.1
    lora_r = 64
    device_map = {"": 0}

    # CUDA Device
    cuda_device_id = 0
    if set_cuda_device(cuda_device_id):
        print(f"CUDA Device {cuda_device_id} is available.")
    else:
        print("CUDA is not available. Training will continue on CPU.")

    tokenizer, model, peft_config = load_model_and_tokenizer(
        model_name,
        use_4bit,
        bnb_4bit_quant_type,
        bnb_4bit_compute_dtype,
        use_nested_quant,
        lora_alpha,
        lora_dropout,
        lora_r,
    )

    configure_model(model, peft_config)

    data_file = "validation-squad.csv"
    column_names = ["context", "question", "text"]
    train_dataset = load_dataset_from_csv(data_file, column_names)
    train_dataset = tokenize_dataset(train_dataset, tokenizer)

    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_dataset, args=training_arguments
    )

    train_model(trainer, train_dataset)

    new_model = "Llama-2-7b-chat-hf_squad_v2"
    trainer.model.save_pretrained(new_model)
