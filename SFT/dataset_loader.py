import torch
from datasets import load_dataset
from tqdm import tqdm
import random
from sft_trainer import dLLMSFTDataset


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

eot_token = "<|eot_id|>"


def extract_s1k_data(data_item):
    """Extract question, reasoning, and answer from s1k dataset format"""
    question = data_item["question"]
    reasoning = data_item["thinking_trajectories"][0]
    answer = data_item["attempt"]
    return question, reasoning, answer


def extract_deepmath_data(data_item):
    """Extract question, reasoning, and answer from DeepMath-103K dataset format"""
    question = data_item["question"]
    reasoning = data_item["r1_solution_1"]
    answer = data_item["final_answer"]
    return question, reasoning, answer


# Registry of available datasets
DATASET_REGISTRY = {
    "simplescaling/s1K": {
        "extract_fn": extract_s1k_data,
        "split": "train"
    },
    "zwhe99/DeepMath-103K": {
        "extract_fn": extract_deepmath_data,
        "split": "train"
    }
}


def preprocess_single_example(example, tokenizer, max_length):
    """Process a single example for tokenization"""
    # Apply system prompt format
    formatted_question = SYSTEM_PROMPT + "\n\n" + example["question"]
    trajectory = f"<reasoning>{example['reasoning']}</reasoning>\n<answer>{example['answer']}</answer>"
    
    # Create chat template format
    prompt = [{"role": "user", "content": formatted_question}]
    response = [{"role": "assistant", "content": trajectory}]
    inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False, add_generation_prompt=False)
    prompt_only = tokenizer.apply_chat_template(prompt, tokenize=False)

    # Remove the part after the final eot_token
    if eot_token in inputs:
        last_eot_index = inputs.rfind(eot_token)
        inputs = inputs[:last_eot_index + len(eot_token)]

    # Tokenize
    tokenized_input = tokenizer(
        inputs, truncation=True, max_length=max_length, padding="max_length"
    )
    
    tokenized_prompt = tokenizer(prompt_only, truncation=True, max_length=max_length)
    
    return {
        "input_ids": tokenized_input["input_ids"],
        "prompt_lengths": sum(tokenized_prompt["attention_mask"]),
    }


def preprocess_batch(examples, tokenizer, max_length):
    """Process a batch of examples for tokenization"""
    batch_inputs = []
    batch_prompt_lengths = []
    
    for i in range(len(examples["question"])):
        # Apply system prompt format
        formatted_question = SYSTEM_PROMPT + "\n\n" + examples["question"][i]
        trajectory = f"<reasoning>{examples['reasoning'][i]}</reasoning>\n<answer>{examples['answer'][i]}</answer>"
        
        # Create chat template format
        prompt = [{"role": "user", "content": formatted_question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False, add_generation_prompt=False)
        prompt_only = tokenizer.apply_chat_template(prompt, tokenize=False)

        # Remove the part after the final eot_token
        if eot_token in inputs:
            last_eot_index = inputs.rfind(eot_token)
            inputs = inputs[:last_eot_index + len(eot_token)]

        batch_inputs.append(inputs)
        batch_prompt_lengths.append(prompt_only)
    
    # Batch tokenize for efficiency - return lists to avoid multiprocessing tensor issues
    tokenized_inputs = tokenizer(
        batch_inputs, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length",
        return_tensors=None  # Return lists to avoid multiprocessing issues
    )
    
    tokenized_prompts = tokenizer(
        batch_prompt_lengths, 
        truncation=True, 
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    
    # Calculate prompt lengths 
    prompt_lengths = [sum(mask) for mask in tokenized_prompts["attention_mask"]]
    
    return {
        "input_ids": tokenized_inputs["input_ids"],  # List of lists
        "prompt_lengths": prompt_lengths  # List of integers
    }


def preprocess_extracted_data_fast(extracted_data, tokenizer, max_length, test_split=0.01):
    """
    Fast preprocessing using Hugging Face datasets' map function with multiprocessing and batching
    """
    from datasets import Dataset
    import multiprocessing
    
    # Convert to HF dataset format
    dataset_dict = {
        "question": [item[0] for item in extracted_data],
        "reasoning": [item[1] for item in extracted_data], 
        "answer": [item[2] for item in extracted_data]
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Get number of CPU cores for parallel processing
    num_proc = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid overwhelming
    batch_size = 1000  # Process in batches for better efficiency
    
    print(f"Using {num_proc} processes with batch size {batch_size} for parallel preprocessing...")
    
    # Apply preprocessing with multiprocessing and batching
    processed_dataset = dataset.map(
        lambda examples: preprocess_batch(examples, tokenizer, max_length),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Preprocessing dataset"
    )
    
    # Convert batched lists to tensors (vectorized and fast)
    print("Converting batched data to tensors...")
    
    # Flatten all batches into single lists
    all_input_ids = [sample["input_ids"] for sample in processed_dataset]
    all_prompt_lengths = [sample["prompt_lengths"] for sample in processed_dataset]
    
    # Convert to tensors in one go (much faster than individual conversions)
    input_ids_tensor = torch.tensor(all_input_ids)  # Shape: (N, max_length)
    prompt_lengths_tensor = torch.tensor(all_prompt_lengths).unsqueeze(-1)  # Shape: (N, 1)
    print(input_ids_tensor.shape, prompt_lengths_tensor.shape)
    
    # Create list of individual samples
    preprocessed_data = []
    for i in tqdm(range(input_ids_tensor.shape[0]), desc="Converting to tensors"):
        preprocessed_data.append({
            "input_ids": input_ids_tensor[i],
            "prompt_lengths": prompt_lengths_tensor[i]  # Shape: (1,) - matches original format
        })
    
    # Shuffle and split
    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data


def load_data(args, tokenizer):
    """
    Load and preprocess data for different datasets based on dataset name
    """
    dataset_name = args.train_data
    
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    dataset_config = DATASET_REGISTRY[dataset_name]
    extract_fn = dataset_config["extract_fn"]
    split = dataset_config["split"]
    
    # Load raw dataset
    print(f"Loading dataset: {dataset_name}")
    raw_data = load_dataset(dataset_name, split=split)
    
    # Extract question, reasoning, answer for each item
    print(f"Extracting data from {len(raw_data)} samples...")
    extracted_data = []
    for item in raw_data:
        try:
            question, reasoning, answer = extract_fn(item)
            extracted_data.append((question, reasoning, answer))
        except Exception as e:
            print(f"Warning: Failed to extract data from item: {e}")
            continue
    
    print(f"Successfully extracted {len(extracted_data)} samples")
    
    # Apply max samples limit if specified
    if hasattr(args, 'max_train_samples') and args.max_train_samples is not None:
        if args.max_train_samples < len(extracted_data):
            print(f"Limiting dataset to {args.max_train_samples} samples (from {len(extracted_data)})")
            # Shuffle before limiting to get a random subset
            random.shuffle(extracted_data)
            extracted_data = extracted_data[:args.max_train_samples]
        else:
            print(f"Requested {args.max_train_samples} samples, but dataset only has {len(extracted_data)} samples. Using all available.")
    
    # Apply common preprocessing
    if getattr(args, 'use_fast_preprocessing', True):
        print("Using fast parallel preprocessing...")
        train_data, eval_data = preprocess_extracted_data_fast(extracted_data, tokenizer, args.max_length)
    else:
        print("Using standard sequential preprocessing...")
        # Keep the old function for backward compatibility
        train_data, eval_data = preprocess_extracted_data_sequential(extracted_data, tokenizer, args.max_length)
    
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    
    # Create datasets
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    
    return train_dataset, eval_dataset


def preprocess_extracted_data_sequential(extracted_data, tokenizer, max_length, test_split=0.01):
    """
    Original sequential preprocessing (kept for backward compatibility)
    """
    preprocessed_data = []
    
    for i, (question, reasoning, answer) in enumerate(tqdm(extracted_data, desc="Preprocessing dataset")):
        # Apply system prompt format
        formatted_question = SYSTEM_PROMPT + "\n\n" + question
        trajectory = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
        
        # Create chat template format
        prompt = [{"role": "user", "content": formatted_question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False, add_generation_prompt=False)
        prompt_only = tokenizer.apply_chat_template(prompt, tokenize=False)

        # Remove the part after the final eot_token
        if eot_token in inputs:
            last_eot_index = inputs.rfind(eot_token)
            inputs = inputs[:last_eot_index + len(eot_token)]

        # Tokenize
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        )["input_ids"].squeeze(0)
        
        tokenized_prompt = tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=max_length)
        
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt["attention_mask"].sum(-1),  # This returns a scalar tensor
            }   
        )

    # Shuffle and split
    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data


def get_available_datasets():
    """Return list of available dataset names"""
    return list(DATASET_REGISTRY.keys()) 