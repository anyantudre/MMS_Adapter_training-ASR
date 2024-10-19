import json
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2Processor,
)


TARGET_LANGUAGE = "mos"
HF_REPO_NAME = "wav2vec2-large-mms-1b-mos-V2"
DATASET_TEXT_COL = "text"



class ProcessorManager:
    def __init__(self, target_lang=TARGET_LANGUAGE, repo_name=HF_REPO_NAME):
        """
        Initialize the ProcessorManager with the target language and repository name.

        Args:
            target_lang (str): The target language code.
            repo_name (str): The name of the Hugging Face repository to push the tokenizer.
        """
        self.target_lang = target_lang
        self.repo_name = repo_name
        self.tokenizer = None
        self.feature_extractor = None
        self.processor = None

    def init_and_push_tokenizer_to_hub(self):
        """
        Initialize the tokenizer and push it to the Hugging Face Hub.

        Returns:
            Wav2Vec2CTCTokenizer: The initialized tokenizer.
        """
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "./",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
            target_lang=self.target_lang
        )
        self.tokenizer.push_to_hub(self.repo_name)
        return self.tokenizer


    def init_feature_extractor_and_processor(self, feature_size=1, sampling_rate=16000,
                                              padding_value=0.0, do_normalize=True,
                                              return_attention_mask=True):
        """
        Initialize the feature extractor and processor.

        Args:
            feature_size (int): Size of the features.
            sampling_rate (int): Sampling rate of the audio.
            padding_value (float): Padding value for the features.
            do_normalize (bool): Whether to normalize the features.
            return_attention_mask (bool): Whether to return attention mask.
        """
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            do_normalize=do_normalize,
            return_attention_mask=return_attention_mask
        )
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer
        )


    def extract_all_chars(self, batch, text_col=DATASET_TEXT_COL):
        """
        Extract all unique characters from the text in the specified column.

        Args:
            batch (dict): The batch of data containing text.
            text_col (str): The column name that contains the text.

        Returns:
            dict: A dictionary with the vocabulary and all text.
        """
        all_text = " ".join(batch[text_col])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}


    def create_and_save_new_vocab_dict(self, vocab):
        """
        Create a new vocabulary dictionary and save it as a JSON file.

        Args:
            vocab (dict): The vocabulary extracted from the dataset.

        Returns:
            dict: The new vocabulary dictionary.
        """
        vocab_list = list(set(vocab["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

        vocab_dict["|"] = vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        del vocab_dict[" "]

        print(f"New vocabulary length: {len(vocab_dict)}")
        print(f"New vocabulary tokens: {vocab_dict}")

        new_vocab_dict = {self.target_lang: vocab_dict}
        with open('vocab.json', 'w') as vocab_file:
            json.dump(new_vocab_dict, vocab_file)

        return new_vocab_dict


    def create_tokenizer(self, dataset):
        """
        Build the tokenizer and vocabulary from the dataset.

        Args:
            dataset: The dataset to extract text from.

        Returns:
            Wav2Vec2CTCTokenizer: The created tokenizer.
        """
        print("Building tokenizer and vocabulary...")

        vocab = dataset.map(
            self.extract_all_chars,
            batched=True,
            batch_size=-1,
            keep_in_memory=True,
            remove_columns=dataset.column_names
        )

        new_vocab_dict = self.create_and_save_new_vocab_dict(vocab)
        tokenizer = self.init_and_push_tokenizer_to_hub()
        return tokenizer
