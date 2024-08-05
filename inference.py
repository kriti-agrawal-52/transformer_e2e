import torch
import torch.nn.functional as F
import logging
import os
import argparse
import wandb # Import wandb
from transformers import AutoTokenizer

from model import TransformerModel # Import your model
from data_utils import PreprocessingTraining # Import for encode/decode methods, or just use AutoTokenizer

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def generate_text(model, tokenizer_obj, prompt_text, max_tokens_ahead=100, temperature=0.7, top_k=40):
    """
    Autoregressively generate new tokens, starting from input_ids.
    model: An instance of TransformerModel
    tokenizer_obj: An instance of AutoTokenizer
    input_ids: tensor of shape (B, T') where T' <= context_window
    Returns: generated text string
    """
    model.eval()
    
    input_ids = torch.tensor([tokenizer_obj.encode(prompt_text)], dtype=torch.long).to(DEVICE)

    for _ in range(max_tokens_ahead):
        # Truncate to the last `context_window` tokens if too long
        if input_ids.size(1) > model.context_window:
            input_condensed = input_ids[:, -model.context_window:]
        else:
            input_condensed = input_ids
        
        logits, _ = model.forward(input_condensed)
        
        last_logits = logits[:, -1, :]

        if temperature > 0:
            last_logits = last_logits / temperature
        
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(last_logits, dim=-1)
        
        if temperature == 0:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat((input_ids, next_token), dim=1)
    
    return tokenizer_obj.decode(input_ids[0].cpu().tolist())

def get_wandb_run_info(project_name="decoder only transformer training", entity=None):
    """
    Fetches information about runs in a WandB project and allows interactive selection.
    
    Args:
        project_name (str): The name of your WandB project.
        entity (str, optional): Your WandB entity (username or team name). If None, uses default.
    
    Returns:
        tuple: (wandb_run_id, model_config, model_artifact_path)
               or (None, None, None) if no selection or error.
    """
    api = wandb.Api()
    
    try:
        if entity:
            runs = api.runs(f"{entity}/{project_name}")
        else:
            runs = api.runs(project_name)
        
        if not runs:
            logger.info(f"No runs found in WandB project '{project_name}'.")
            print(f"No runs found in WandB project '{project_name}'.")
            return None, None, None

        print(f"\nFound {len(runs)} runs in WandB project '{project_name}':")
        run_options = []
        for i, run in enumerate(runs):
            # Only consider runs that logged a model artifact
            if any(a.type == 'model' for a in run.artifacts()):
                run_options.append(run)
                # Display relevant info like run name, ID, and a best_val_loss if available
                val_loss = run.summary.get('val/loss')
                train_loss = run.summary.get('train/loss') # or avg train loss
                steps = run.summary.get('_step') # last step
                
                info_str = f"[{i+1}] Name: {run.name} (ID: {run.id})"
                if val_loss is not None:
                    info_str += f", Val Loss: {val_loss:.4f}"
                if train_loss is not None:
                    info_str += f", Train Loss: {train_loss:.4f}"
                if steps is not None:
                    info_str += f", Steps: {steps}"
                
                # Show key config parameters for easy identification
                config_summary = []
                if 'batch_size' in run.config: config_summary.append(f"BS={run.config['batch_size']}")
                if 'context_window' in run.config: config_summary.append(f"CW={run.config['context_window']}")
                if 'learning_rate' in run.config: config_summary.append(f"LR={run.config['learning_rate']:.0e}")
                if config_summary:
                    info_str += f" (Config: {', '.join(config_summary)})"
                print(info_str)

        if not run_options:
            logger.info(f"No runs with 'model' artifacts found in WandB project '{project_name}'.")
            print(f"No runs with 'model' artifacts found in WandB project '{project_name}'.")
            return None, None, None

        while True:
            try:
                choice = input(f"Enter the number of the run to use (1-{len(run_options)}): ")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(run_options):
                    selected_run = run_options[choice_idx]
                    break
                else:
                    print("Invalid choice. Please enter a number within the range.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        logger.info(f"Selected WandB run: {selected_run.name} (ID: {selected_run.id})")
        
        # Load the model artifact
        model_artifact = None
        for artifact in selected_run.artifacts():
            if artifact.type == 'model':
                model_artifact = artifact
                break
        
        if model_artifact is None:
            logger.error(f"Selected run {selected_run.name} (ID: {selected_run.id}) does not have a 'model' artifact.")
            print(f"Error: Selected run does not have a 'model' artifact. Please choose another.")
            return None, None, None
            
        # Download the artifact if it's not cached locally
        artifact_dir = model_artifact.download()
        logger.info(f"Model artifact downloaded to: {artifact_dir}")
        
        # The actual model file name inside the artifact dir
        # Assuming your artifact only contains one .pt file
        model_file = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
        if not model_file:
            logger.error(f"No .pt file found in downloaded artifact directory: {artifact_dir}")
            print(f"Error: No model file (.pt) found in the downloaded artifact. Check training script's artifact logging.")
            return None, None, None
        model_path = os.path.join(artifact_dir, model_file[0])

        return selected_run.id, selected_run.config, model_path

    except Exception as e:
        logger.error(f"Error fetching WandB runs or artifacts: {e}", exc_info=True)
        print(f"Error connecting to WandB or fetching runs. Make sure you are logged in (wandb login). Error: {e}")
        return None, None, None


if __name__ == '__main__':
    logging.basicConfig(
        filename='transformer_inference.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger.info(f"Using device for inference: {DEVICE}")

    parser = argparse.ArgumentParser(description="Run Transformer Model Inference")
    parser.add_argument('--wandb_project', type=str, default='decoder only transformer training',
                        help='Name of the WandB project to load models from.')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Your WandB entity (username or team name). If None, uses default.')
    parser.add_argument('--model_id_or_path', type=str, default=None,
                        help='Optional: Directly specify a WandB Run ID (e.g., "abcdef12") or a local model path. '
                             'If not provided, an interactive selection from WandB will be used.')
    
    parser.add_argument('--prompt', type=str, default='Romeo, Romeo, wherefore art thou',
                        help='Initial text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature for generation (0.0 for greedy)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-K sampling for generation')
    
    args = parser.parse_args()

    model_config = None
    model_path = None

    if args.model_id_or_path:
        # Check if it's a WandB run ID or a local path
        if len(args.model_id_or_path) == 8 and all(c.isalnum() for c in args.model_id_or_path): # Heuristic for run ID
            try:
                api = wandb.Api()
                run = api.run(f"{args.wandb_entity if args.wandb_entity else api.default_entity}/{args.wandb_project}/{args.model_id_or_path}")
                logger.info(f"Directly specified WandB run: {run.name} (ID: {run.id})")

                model_artifact = None
                for artifact in run.artifacts():
                    if artifact.type == 'model':
                        model_artifact = artifact
                        break
                
                if model_artifact is None:
                    logger.critical(f"Specified run {run.name} (ID: {run.id}) does not have a 'model' artifact.")
                    print(f"Error: Specified run does not have a 'model' artifact. Please check the run ID.")
                    exit()

                artifact_dir = model_artifact.download()
                model_file = [f for f in os.listdir(artifact_dir) if f.endswith('.pt')]
                if not model_file:
                    logger.critical(f"No .pt file found in downloaded artifact directory: {artifact_dir}")
                    print(f"Error: No model file (.pt) found in the downloaded artifact. Check training script's artifact logging.")
                    exit()
                model_path = os.path.join(artifact_dir, model_file[0])
                model_config = run.config

            except Exception as e:
                logger.critical(f"Error loading model directly from WandB run ID {args.model_id_or_path}: {e}", exc_info=True)
                print(f"Error loading model directly from WandB run ID {args.model_id_or_path}. Error: {e}")
                exit()
        else: # Assume it's a local path
            model_path = args.model_id_or_path
            logger.info(f"Loading model from local path: {model_path}. You might need to manually specify model dimensions if not loaded from WandB config.")
            print("Loading model from local path. Make sure model configuration (vocab_size, context_window, etc.) matches the saved model.")
            # If loading from local path directly, and not from a WandB run,
            # we don't have the config automatically. You'd need to provide
            # these manually or save them alongside the .pt file.
            # For simplicity, we'll exit if not from WandB unless args are provided for these.
            print("For local models, please specify --vocab_size, --context_window etc. (or load config from a file). Exiting.")
            exit()
    else: # Interactive selection from WandB
        _, model_config, model_path = get_wandb_run_info(args.wandb_project, args.wandb_entity)
        if model_config is None or model_path is None:
            logger.critical("Failed to get model configuration and path from WandB. Exiting.")
            print("Failed to select a model. Exiting.")
            exit()

    # Initialize tokenizer (assuming GPT-2 tokenizer for WikiText-2 dataset)
    tokenizer_name = "gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ({tokenizer.eos_token})")
    except Exception as e:
        logger.critical(f"Could not load tokenizer '{tokenizer_name}'. Error: {e}", exc_info=True)
        print(f"Error: Could not load tokenizer '{tokenizer_name}'. Ensure internet access or cached tokenizer.")
        exit()

    # --- Initialize model using loaded config ---
    if model_config:
        try:
            vocab_size = model_config.get('tokenizer_vocab_size', tokenizer.vocab_size) # Prefer logged vocab_size
            # Fallback if specific config keys aren't found (though they should be if logged well)
            context_window = model_config.get('context_window', args.context_window if 'context_window' in args else 64)
            channel_dim = model_config.get('embedding_dim', args.channel_dim if 'channel_dim' in args else 128)
            num_heads = model_config.get('num_heads', args.num_heads if 'num_heads' in args else 8)
            num_layers = model_config.get('num_layers', args.num_layers if 'num_layers' in args else 6)

            model = TransformerModel(
                vocab_size=vocab_size,
                channel_dim=channel_dim,
                context_window=context_window,
                num_heads=num_heads,
                num_layers=num_layers
            ).to(DEVICE)

            # Load trained weights
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval() # Set to evaluation mode
            logger.info(f"Model loaded successfully from {model_path} with config: {model_config}")
            print(f"Model loaded with parameters: Vocab Size={vocab_size}, Context Window={context_window}, Channel Dim={channel_dim}")
        except Exception as e:
            logger.critical(f"Error initializing or loading model from config/path: {e}", exc_info=True)
            print(f"Error initializing or loading model. Check model architecture parameters or checkpoint integrity. Error: {e}")
            exit()
    else:
        logger.critical("No model configuration available. Exiting.")
        print("No model configuration available. Cannot initialize model. Exiting.")
        exit()

    # Generate text
    logger.info(f"Generating text with prompt: '{args.prompt}'")
    print(f"\n--- Generated Text (Prompt: '{args.prompt}') ---")
    generated_text_output = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_tokens_ahead=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print(generated_text_output)
    print("--- End of Generated Text ---")
    logger.info(f"Generated text: {generated_text_output}")