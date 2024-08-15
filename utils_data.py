import torch
import logging
from transformers import AutoTokenizer
from datasets import load_dataset
import sys

logger = logging.getLogger(__name__)


class PreprocessingTraining():
    """
        Handles data loading, tokenization, splitting, and batching.

        This class takes raw text data and a tokenizer as input. It performs
        the following steps:
        1.  Tokenizes the entire text corpus into a sequence of token IDs.
        2.  Splits the token IDs into training, validation, and test sets based on specified ratios.
        3.  Provides a `get_batch` method to randomly sample sequences of a fixed length (context window) from the specified data split, creating input (x) and target (y) pairs suitable for language model training. These tensors are then stacked to create a batch where each batch has batch_size sequences.
        4.  Provides encoding and decoding utilities.
        """
    def __init__(self, text: str, tokenizer: object, batch_size: int = 4, time_steps: int = 64):
        """Initializes the data preprocessing pipeline."""
        self.text = text
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.time_steps = time_steps
        
        logger.info("Tokenizing the entire dataset..")
        self.all_token_ids = self.tokenizer.encode(self.text)  # it tokeizes and encodes are input text, so even repeated tokens such as hello hello would be encoded separately.
        logger.info(f"Tokenization complete. Total tokens: {len(self.all_token_ids)}")
        
        self.vocab_size = self.tokenizer.vocab_size  # vocab size is an attribute of tokenizer object
        
        # split dataset into train, test, validation 
        self._split_tokenized_data()  # this function will always be used within the class, hence preceding function name with _
        
    def _split_tokenized_data(self, train_test_ratio = 0.9, val_train_ratio = 0.1):
        """
        Splits the tokenized data into train, validation, and test sets.
        81% train, 9% val, 10% test split
        """
        n_tokens = len(self.all_token_ids)
        all_tokens_tensor = torch.tensor(self.all_token_ids, dtype=torch.long)

        train_val_end = int(n_tokens * train_test_ratio)
        train_val_data_tokens = all_tokens_tensor[:train_val_end]
        self.test_tokens = all_tokens_tensor[train_val_end:]

        val_size = int(len(train_val_data_tokens) * val_train_ratio)
        train_size = len(train_val_data_tokens) - val_size
        self.train_tokens = train_val_data_tokens[:train_size]
        self.val_tokens = train_val_data_tokens[train_size:]

        logger.info(
            f"Train tokens: {len(self.train_tokens)}, "
            f"Val tokens: {len(self.val_tokens)}, "
            f"Test tokens: {len(self.test_tokens)}"
        )
        if not all(
            [
                self.train_tokens.numel(),
                self.val_tokens.numel(),
                self.test_tokens.numel()
            ]
        ):
            logger.warning("One or more data splits are empty")
        """
        Check if any of the data splits (train, val, test) are empty.
        numel() returns the total number of elements in the tensor across all dimensions.
        For example, a tensor of shape (3, 5) will have numel() = 15.
        If any split has 0 elements, log a warning.
        """
        
    def encode_string(self, s:str):
        """ Encode a string into a list of token IDs"""
        return self.tokenizer.encode(s)

    def decode_ids(self, ids: list):
        """ Decodes a list of token IDs back into string"""
        return self.tokenizer.decode(ids)

    def get_batch(self, split: str = 'train'):
        """
        Generates a batch of input sequences (x) and target sequences (y)
        from the specified tokenized data split (train, validation or test).
        """
        
        if split == 'validation':
            split = 'val'  #handle 'validation' as well as 'val' split names
            
        data_tokens = getattr(self, f"{split}_tokens", None)
        if data_tokens is None:
            raise ValueError(f"Unknown split type: {split}")

        # Ensure that there is enough tokens in the data_tokens to form a complete sequence of time_steps + 1
        max_start_idx = len(data_tokens) - self.time_steps - 1
        if max_start_idx < 0:
            raise ValueError(
                f"Dataset split {split} is too small ({len(data_tokens)} tokens) for time steps={self.time_steps}. Need at least {self.time_steps + 1} tokens.\nConsider a smaller time_steps, larger_dataset, or different split ratios."
            ) 
        
        # randomly select `batch_size` number starting indices from data_tokens to pull batch_size sequences from data_tokens
        # these indices should be between 0 and max_start_idx + 1, as we can have a sequence starting from max_start_idx
        ix = torch.randint(0, max_start_idx + 1, (self.batch_size,))  # in pytorch the third argument in randint has to be a tuple, even if we want a 1D tensor
        
        """
        Create input sequences (x) and target sequences (y) from token IDs.
        We are creating batches such as this because we are doing self-attention training with masking.
        Example:
        If x sequence is: ['twinkle', 'twinkle', 'little', 'star']
        The model's task for each position (due to causal masking) is:
        - given 'twinkle', predict 'twinkle'
        - given 'twinkle twinkle', predict 'little'
        - given 'twinkle twinkle little', predict 'star'
        - given 'twinkle twinkle little star', predict ',' (or whatever the next token in the full text is)
        """
        x = torch.stack([data_tokens[i:i + self.time_steps] for i in ix])
        y = torch.stack([data_tokens[i + 1:i + self.time_steps + 1] for i in ix])
        return x, y

def setup_data_and_tokenizer(cfg):
    """Loads the dataset and tokenizer based on config"""
    logger.info(f"Loading {cfg.DATASET_NAME} dataset.")
    try:
        dataset = load_dataset(cfg.DATASET_NAME, cfg.DATASET_VARIANT)
        logger.info('Successfully loaded dataset.')
        raw_text = '\n'.join(dataset['train']['text'])  # concatenate all of the dataset into a single string
        if cfg.RAW_TEXT_LIMIT:
            raw_text = raw_text[:cfg.RAW_TEXT_LIMIT]  # since we are training on a cpu, we can limit our dataset to cgf.RAW_TEXT_LIMIT character limit
            logger.info(f"Truncated {cfg.DATASET_NAME} to {len(raw_text)} characters for CPU training.")
    except Exception as e:
        logger.critical(f"Could not load dataset: {e}", exc_info=True)  # exc_info tells the logger to include the full traceback of the exception in the log message
        raise # immediately exit runtime with an error
    
    logger.info(f"Loading tokenizer '{cfg.TOKENIZER_NAME}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            """
            If the tokenizer doesn't have a pad_token (common in models like GPT-2), set it to the eos_token so padding operations can still work.
            Padding is needed when batching sequences of different lengths — since tensors must have the same shape, shorter sequences are padded with a special token (pad_token) so they match the longest one in the batch.
            For example, if your batch has sequences of lengths [4, 6, 5], they will be padded to length 6 with the pad_token.
            """
    except Exception as e:
        logger.critical(f"Could not load tokenizer: {e}", exc_info = True)
        raise
        
    return raw_text, tokenizer