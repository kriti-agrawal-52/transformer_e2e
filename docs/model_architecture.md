# Project Architecture and Class Interaction

This document outlines the overall architecture and the flow of interaction between different classes within the project.

## 1. Configuration (`config.yml`, `utils_config.py`)

- YML file that centralizes all hyperparameters, control flags (e.g., `SHOULD_TRAIN_MAIN`, `SHOULD_TUNE`), file paths, and Weights & Biases (W&B) settings. It's designed for easy modification of parameters.
- YML style configuration files (like config.yml) are important and better (than a simple config.py type configuration file) because they are purely data, separating configuration from code logic.
- This makes projects more secure (YML isn't executable), easier to understand,
- and allows them to work with other programming languages (like Go or Java) since YML is a language-agnostic standard.
- However, since YML is just data (even if it has lists), we need a "loader" (like utils_config.py) to read the YML file and turn that data into Python objects that our project can use.

---

## 2. Main Execution (`if __name__ == "__main__":` -> `main()`)

- The script initiates by calling the **`main()` function**.
- `main()` is responsible for invoking `setup_data_and_tokenizer()` if either main training or hyperparameter tuning is enabled.
- Based on the flags configured in `config_for_script.py`, it subsequently calls either **`run_main_training()` or `run_hyperparameter_search()`, or both**.

---

## 3. Data Handling (`PreprocessingTraining` class)

- This class is **instantiated by `TrainingManager`**.
- It takes raw text data and a tokenizer as inputs.
- **`__init__`**: Tokenizes the entire input text, calculates the vocabulary size, and then calls `_split_tokenized_data`.
- **`_split_tokenized_data`**: Divides the tokenized IDs into **training, validation, and test sets**.
- **`get_batch()`**: This method is called by `TransformerModel.train_loop` and `TransformerModel.evaluate_validation_loss` to retrieve batches of data during training and evaluation.

---

## 4. Core Model (`TransformerModel` class)

- This is the **central class** that defines the Transformer architecture.
- **`__init__`**:
  - Initializes **token and positional embedding layers**.
  - Instantiates multiple **`TransformerBlock` objects** (representing the layers of the Transformer) and stores them in an `nn.ModuleList`.
  - Creates a final layer normalization (`ln_f`) and an output projection layer (`proj`).
- **`forward(x_indices, targets)`**: Defines the data flow through the entire model.
  - Input `x_indices` (token IDs) are processed through token and positional embeddings.
  - The combined embeddings are then sequentially processed by each `TransformerBlock` within `self.transformer_blocks`.
  - The output from the final block passes through `ln_f` and then `proj` to generate logits.
  - If `targets` are provided, it computes the **cross-entropy loss**.
- **`train_loop()`**: Manages the actual training steps, including optimizer updates, validation checks, **early stopping**, and **checkpointing** (saving both "latest" and "best" model states). It calls `_save_checkpoint` and `evaluate_validation_loss`.
- **`evaluate_validation_loss()`**: Calculates the loss on a specified validation or test set.
- **`generate()`**: Performs **autoregressive text generation**.

---

## 5. Transformer Layer (`TransformerBlock` class)

- Represents a **single layer or block** within the Transformer.
- **`__init__`**:
  - Instantiates a **`MultiHeadAttention` module** for self-attention.
  - Creates a **Feed-Forward Network (FFN)**.
  - Sets up two Layer Normalization layers (`ln1`, `ln2`).
- **`forward(x)`**: Defines the data flow within a single block.
  - Input `x` first passes through `ln1`, then `MultiHeadAttention` (`self.attn`), followed by a **residual connection**.
  - The result then goes through `ln2`, the FFN (`self.ffn`), and another **residual connection**.

---

## 6. Attention Mechanism (`MultiHeadAttention` class)

- Implements the **multi-head self-attention mechanism**.
- **`__init__`**:
  - Sets up linear layers to project inputs into **Queries (Q), Keys (K), and Values (V)**.
  - Defines an output projection layer.
  - Registers a buffer for the **causal mask (`tril`)**.
- **`forward(x)`**:
  - Takes input `x` (token embeddings + positional encodings from `TransformerModel`).
  - Projects `x` into Q, K, V and reshapes them for multiple attention heads.
  - Computes attention scores (scaled dot-product of Q and K.T).
  - Applies the **causal mask** to prevent attending to future tokens.
  - Applies softmax to obtain attention weights.
  - Computes the **weighted sum of V** (the attention output).
  - Concatenates outputs from different heads and passes them through the final output projection layer.

---

## 7. Training Orchestration (`TrainingManager` class)

- This class **orchestrates an entire training run** (either a main run or one iteration of a hyperparameter search).
- **`__init__`**: Takes run-specific parameters (e.g., batch size, learning rate), the tokenizer, raw text, and a `is_main_run` flag. It also defines checkpoint paths based on a unique `run_id`.
- **`run()`**:
  - Initializes a **W&B run** (allowing resumption using `run_id`).
  - Creates instances of `PreprocessingTraining` and `TransformerModel`.
  - Calls `_load_checkpoint()` to attempt to **resume from a previous state**, if training was interrupted.
  - Calls `TransformerModel.train_loop()` to perform the training.
  - If `is_main_run` is true, it calls `_run_post_training_eval()` (which in turn calls `TransformerModel.evaluate_validation_loss` on the test set and `TransformerModel.generate`).
  - Calls `_log_artifact()` to **save the model to W&B**.
  - Ensures the W&B run is properly finished.
- **`_load_checkpoint()`**: Handles loading the model and optimizer state from disk.
- **`_run_post_training_eval()`**: Manages evaluation after training completion.
- **`_log_artifact()`**: Logs the trained model to Weights & Biases.

---

## Why use different droppout objects for different layers in the transformer architecture?

**Don't reuse the same `nn.Dropout` object across different layers!**

Each dropout layer randomly turns off (zeroes out) some outputs during training using a mask
(e.g., `[1, 0, 1, 0]`). This helps the model avoid relying too much on any single neuron.

**If you use the _same_ `nn.Dropout` object in multiple places:**

- The _same mask_ will be reused across those layers during a single forward pass.
- That means the same neurons get turned off together in every layer.
- This links the layers in a bad way — the model starts expecting certain neurons to always be off.
- It reduces randomness and breaks the regularization effect of dropout.

**What to do instead:**

- Always create a **separate** `nn.Dropout` object for each layer.
- This ensures each layer gets a fresh random mask during training.
- It helps the model learn independently in each layer and improves generalization.

---

## Why dropout_rate = 0.2 is not too low for our model

- Transformers are already heavily regularized by design

They use LayerNorm, residual connections, and attention masking, which already add stability and structure.
Too much dropout could impair convergence, especially with small models and CPUs (as in our use case).

- 0.1–0.3 is standard in Transformer literature

Some real-world defaults:

GPT-2 (small): uses 0.1–0.2 dropout
BERT-base: uses 0.1
T5 and GPT-J: use 0.1 to 0.3 depending on depth
