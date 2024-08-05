import torch
import logging
from transformers import AutoTokenizer #
from datasets import load_dataset #

logger = logging.getLogger(__name__)

class PreprocessingTraining: #
    def __init__(self, text, tokenizer, batch_size=4, time_steps=64): #
        self.text = text #
        self.tokenizer = tokenizer #
        self.batch_size = batch_size #
        self.time_steps = time_steps #
        
        logger.info("Tokenizing the entire dataset...") #
        self.all_token_ids = self.tokenizer.encode(self.text) #
        logger.info(f"Tokenization complete. Total tokens: {len(self.all_token_ids)}") #
        
        self.vocab_size = self.tokenizer.vocab_size #
        logger.info(f"Vocabulary size: {self.vocab_size}") #
        
        self._split_tokenized_data() #
    
    def _split_tokenized_data(self, train_val_ratio=0.9, val_ratio_of_train_val=0.1): #
        n_tokens = len(self.all_token_ids) #
        train_val_idx_end = int(n_tokens * train_val_ratio) #
        all_tokens_tensor = torch.tensor(self.all_token_ids, dtype=torch.long) #
        train_val_data_tokens = all_tokens_tensor[:train_val_idx_end] #
        self.test_tokens = all_tokens_tensor[train_val_idx_end:] #
        val_idx_start = int(len(train_val_data_tokens) * (1 - val_ratio_of_train_val)) #
        self.train_tokens = train_val_data_tokens[:val_idx_start] #
        self.val_tokens = train_val_data_tokens[val_idx_start:] #
        logger.info(f"Train tokens: {len(self.train_tokens)}, Val tokens: {len(self.val_tokens)}, Test tokens: {len(self.test_tokens)}") #
        if len(self.train_tokens) == 0 or len(self.val_tokens) == 0 or len(self.test_tokens) == 0: #
            logger.warning("One or more data splits are empty. Check dataset size and split ratios.") #
            
    def encode_string(self, s: str): #
        return self.tokenizer.encode(s) #
    
    def decode_ids(self, ids: list): #
        return self.tokenizer.decode(ids) #

    def get_batch(self, split='train'): #
        if split == 'train': #
            data_tokens = self.train_tokens #
        elif split == 'validation': #
            data_tokens = self.val_tokens #
        elif split == 'test': #
            data_tokens = self.test_tokens #
        else:
            raise ValueError(f"Unknown split type: {split}") #

        max_start_idx = len(data_tokens) - self.time_steps - 1 #
        if max_start_idx < 0: #
             raise ValueError(
                 f"Dataset split '{split}' is too small ({len(data_tokens)} tokens) " #
                 f"for time_steps={self.time_steps}. Need at least {self.time_steps + 1} tokens. " #
                 f"Consider a smaller time_steps, larger dataset, or different split ratios." #
            )
        
        ix = torch.randint(0, max_start_idx + 1, (self.batch_size,)) #
        x = torch.stack([data_tokens[i : i + self.time_steps] for i in ix]) #
        y = torch.stack([data_tokens[i + 1 : i + self.time_steps + 1] for i in ix]) #
        
        return x, y