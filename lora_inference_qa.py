import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_and_tokenizer(BASE_MODEL, LORA_WEIGHTS):
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    finetuned_model = PeftModel.from_pretrained(
        base_model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map="auto"
    )

    return tokenizer, finetuned_model


def make_inference(context, question, tokenizer, finetuned_model):
    batch = tokenizer(
        f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n",
        return_tensors="pt",
    )
    batch.to("cuda")

    with torch.cuda.amp.autocast():
        output_tokens = finetuned_model.generate(**batch, max_new_tokens=200)

    print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))


if __name__ == "__main__":
    BASE_MODEL = "models/pre-trained/Cerebras-GPT-590M"  # "bigscience/bloomz-560m"
    LORA_WEIGHTS = "models/fine-tuned/Cerebras-GPT-590M_QA"
    tokenizer, fn_model = load_model_and_tokenizer(BASE_MODEL, LORA_WEIGHTS)

    context = "Cheese is the best food."
    question = "What is the best food?"
    make_inference(context, question, tokenizer, fn_model)
