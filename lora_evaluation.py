import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge import Rouge 
# from rouge_score import rouge_scorer

def calculate_and_return_remaining_string(larger_str, smaller_str):
    length_difference = len(larger_str) - len(smaller_str)
    
    if length_difference >= 0:
        remaining_string = larger_str[length_difference:]
    else:
        print("Error: The smaller string is larger than the larger string.")
        remaining_string = None
        
    return remaining_string

def evaluate_prompt_completions(llm_name, prompt_completions):
    # Load language model
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name)
    
    # Initialize evaluation metrics
    bleu_scores = []
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    perplexity_scores = []
    
    # Initialize Rouge scorer
    rouge_scorer = Rouge()

    for prompt, completion in prompt_completions:
        # Generate completion from prompt using the language model
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = model.generate(input_ids)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # generated_text = calculate_and_return_remaining_string(generated_text,prompt)
        completion = prompt + completion
        print(f"generated_text:{generated_text}")
        print(f"completion:{completion}")
        # generated_text = "The quick brown foxes are a great way to get a little bit of a kick out of your"
        # completion = "The quick brown foxes jumps over the lazy dog."

        # generated_text = "Once upon a time, the world was a place of great beauty and great danger. The world was"
        # completion = "Once upon a time there was a magical kingdom"
        

        # Calculate BLEU score
        reference = [completion.split()]
        candidate = generated_text.split()
        bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
        
        # Calculate ROUGE scores
        rouge_scores_batch = rouge_scorer.get_scores(generated_text, completion)
        for metric in rouge_scores:
            rouge_scores[metric].append(rouge_scores_batch[0][metric])
        
        # Calculate perplexity
        input_ids = tokenizer.encode(completion, return_tensors='pt')
        with torch.no_grad():
            logits = model(input_ids).logits
        perplexity = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1)).item()
        perplexity_scores.append(perplexity)
    
    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    # avg_rouge = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}
    avg_perplexity = sum(perplexity_scores) / len(perplexity_scores)

    avg_rouge = {}

    for key, value_list in rouge_scores.items():
        f_scores = [d['f'] for d in value_list]
        average_f = sum(f_scores) / len(f_scores)
        avg_rouge[key] = average_f
    
    evaluation_metrics = {
        'BLEU': avg_bleu,
        'ROUGE-1': avg_rouge['rouge-1'],
        'ROUGE-2': avg_rouge['rouge-2'],
        'ROUGE-L': avg_rouge['rouge-l'],
        'Perplexity': avg_perplexity
    }
    
    return evaluation_metrics

# Example usage
prompt_completions = [
    ("The quick brown fox", "jumps over the lazy dog."),
    ("Once upon a time,", "there was a magical kingdom."),
    # Add more prompt completion pairs as needed
]

llm_name = "gpt2"  # Replace with the name of the desired language model

evaluation_metrics = evaluate_prompt_completions(llm_name, prompt_completions)
print(evaluation_metrics)


# Example usage
# larger_string = "The quick brown foxes are a great way to get a little bit of a kick out of your."
# smaller_string = "The quick brown fox"
# remaining_string = calculate_and_return_remaining_string(larger_string, smaller_string)

# if remaining_string is not None:
#     print("Remaining String:", remaining_string)
