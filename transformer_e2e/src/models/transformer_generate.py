import torch
import torch.nn.functional as F
from model import TransformerModel


# Add the generation method to the existing TransformerModel class
def generate(self, input_ids, max_tokens_ahead, temperature, top_k, device):
    """
    Autoregressively generates text from a prompt in inference mode.

    Args:
        input_ids (torch.Tensor): Initial encoded prompt (token IDs).
        max_tokens_ahead (int): Max new tokens to generate.
        temperature (float): Controls randomness (0 for greedy, >0 for sampling).
        top_k (int, optional): Samples from top_k most probable tokens.

    Process:
    1.  **Input Context**: Truncates `input_ids` to `self.context_window`.
    2.  **Iterative Prediction**: Calls `self(cond)` (forward pass with `targets=None`)
        to get next token logits.
    3.  **Sampling**: Applies temperature/Top-K filtering, then samples/selects next token.
    4.  **Append**: Adds new token to `input_ids`. Repeats.

    @torch.no_grad():
    Disables gradient calculation for:
    -   **Memory Efficiency**: Reduces memory by skipping computation graph.
    -   **Speed**: Faster inference by avoiding gradient overhead.
    -   **Clarity**: Explicitly marks as inference-only.
    """
    self.eval()
    input_ids = input_ids.to(device)

    with torch.no_grad():
        for _ in range(max_tokens_ahead):
            # Get logits for the next token prediction
            cond = (
                input_ids[:, -self.context_window :]
                if input_ids.size(1) > self.context_window
                else input_ids
            )
            logits, _ = self(
                cond
            )  # forward the cond vector through the entire model, and we get logits and loss, for generation loss
            last = logits[:, -1, :]  # Focus on the last token's logits

            # Apply temperature scaling
            # temperature=0 can lead to issues with multinomial if not handled
            if temperature > 0:
                last = last / temperature
            # Temperature > 1 (e.g., 2.0): Dividing the logits by a value greater than 1 makes the logits smaller (closer to zero). When these smaller logits are passed through the softmax function, the resulting probability distribution becomes smoother and more uniform. This means more tokens will have similar (non-zero) probabilities, leading to more diverse and potentially more "creative" or "random" output.

            # Temperature = 1.0: This is the default or "no change" setting. The logits are used as they are.

            # Temperature < 1 (e.g., 0.5): Dividing the logits by a value less than 1 (e.g., 0.5 is equivalent to multiplying by 2) makes the logits larger (more extreme). When these more extreme logits are passed through softmax, the probability distribution becomes sharper, concentrating most of the probability mass on the few most likely tokens. This leads to less diverse and more "deterministic" or "conservative" output.

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(
                    last, min(top_k, last.size(-1))
                )  # identifies the top_k highest logit values (v) and their indices
                last[last < v[:, [-1]]] = -float(
                    "Inf"
                )  # v[:, [-1]]: This specifically extracts the k-th highest logit value.
                # Any logit value that is smaller than the k-th highest logit is set to negative infinity (-float('Inf')).

            # Get probabilities and sample the next token
            probs = F.softmax(
                last, dim=-1
            )  # When softmax is applied, -float('Inf') becomes 0 probability
            # top_k effectively zeroes out the probabilities of all tokens except for the k most probable ones. This restricts the sampling pool to only the most plausible next tokens.

            if temperature == 0:  # Deterministic: take the most probable token only
                next_t = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_t = torch.multinomial(
                    probs, num_samples=1
                )  # getting one token from all of the probabilities

            # Append the new token and continue
            input_ids = torch.cat((input_ids, next_t), dim=1)

    return input_ids


# Monkey patch the generate method to TransformerModel
TransformerModel.generate = generate
