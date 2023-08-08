import pandas as pd

from datasets import Dataset, load_dataset


def process_dataset_from_pandas(data_set_path, create_prompt, use_create_prompt=True):
    # Load CSV file using pandas
    data_frame = pd.read_csv(data_set_path)

    if use_create_prompt:
        # Apply create_prompt function to generate 'prompt' column
        data_frame["prompt"] = data_frame.apply(
            lambda x: create_prompt(x.context, x.question, x.text), axis=1
        )

    # Convert pandas DataFrame to datasets Dataset object
    dataset = Dataset.from_pandas(data_frame)

    return dataset


def process_dataset_load_dataset(
    data_set_path, tokenizer, create_prompt, use_create_prompt=True
):
    # Load CSV file using load_dataset
    dataset = load_dataset("csv", data_files=data_set_path)

    if use_create_prompt:
        # Apply create_prompt function to generate token_ids
        dataset = dataset.map(
            lambda samples: tokenizer(
                create_prompt(samples["context"], samples["question"], samples["text"])
            )
        )

    return dataset


# Example usage

# data_set_path = 'datasets/squad_v2/validation-squad.csv'
# use_create_prompt = True
# from template_wrapper import create_prompt_qa

## Example 1:
# processed_dataset = process_dataset_from_pandas(data_set_path,create_prompt_qa, use_create_prompt)

## Example 2:
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("models/pre-trained/Cerebras-GPT-590M")
# processed_dataset = process_dataset_load_dataset(data_set_path,tokenizer, create_prompt_qa, use_create_prompt)
# Now you can use the processed_dataset for further analysis or tasks
