import os
import re
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Audio


COLS_TO_REMOVE = ["source", "speaker_id", "denoised_audio"]
CHARS_TO_REMOVE_REGEX = r'[\,\-\?\.\!\-\;\:\"\“\%\‘\”\�\'\(\)\/\<\>\[\]\}\{\«\»\'\’\…\‹\›\-\d\\\\_]'
REPLACEMENTS = {
        'â': 'ã',
        'à': 'ã',
        'î': 'ĩ',
        'ô': 'õ',
        'ō': 'õ',
        'û': 'ũ',
        'ù': 'ũ',
        'é': 'ẽ',
        'è': 'ẽ',
        'ê': 'ẽ',
    }


def log(message):
    print(f"[LOG] {message}")


def load_data(
        data_path, 
        split="train"
        ):
    
    print(f"Loading data from {data_path}...")
    dataset = load_dataset(
        data_path, 
        split=split
    )
    
    return dataset


def preprocess_data(
        dataset, 
        remove_cols=COLS_TO_REMOVE, 
        chars_to_remove_regex=CHARS_TO_REMOVE_REGEX,
        replacements=REPLACEMENTS
        ):
    
    print("Preprocessing data...")
    
    # Remove specified columns
    dataset = dataset.remove_columns(remove_cols)

    # List of special character replacements
    replacements = replacements
    
    # Function to remove special characters
    def remove_special_characters(batch):
        # Remove characters based on regex and lower the text
        batch["text"] = re.sub(chars_to_remove_regex, '', batch["text"]).lower()
        
        # Replace special characters
        for old_char, new_char in replacements.items():
            batch["text"] = batch["text"].replace(old_char, new_char)
        
        return batch

    print("Removing special characters...")
    dataset = dataset.map(remove_special_characters)

    return dataset


def prepare_dataset(
        batch, 
        processor
    ):

    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    batch["labels"] = processor(text=batch["text"]).input_ids
    return batch


def filter_empty_samples(batch):
    # Check for empty input_values or labels
    if len(batch["input_values"]) == 0 or len(batch["labels"]) == 0:
        return False  # Discard the sample
    return True


def login_hugging_face():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the Hugging Face token from the environment variable
    token = os.getenv("HF_TOKEN")
    
    if token:
        # Log in to Hugging Face
        login(token=token)
        print("Successfully logged in to Hugging Face.")
    else:
        print("Hugging Face token not found. Please check your .env file.")
