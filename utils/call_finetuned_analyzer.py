# Utility function to call the fine-tuned model locally
import logging
import time
import random
import os
import json
# Removed requests import
# Added new imports
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Custom Model Class (Must match the training definition) ---
class RestaurantReviewAnalyzer(nn.Module):
    """
    A custom model that uses a pre-trained transformer encoder (like XLM-RoBERTa)
    and adds separate regression heads to predict scores for different dimensions
    of a restaurant review (Taste, Service, Ambiance).
    """
    def __init__(self, pretrained_model_name="xlm-roberta-base", num_dimensions=3, dropout_prob=0.1):
        super().__init__()
        logging.info(f"Initializing custom model structure with base: {pretrained_model_name}")
        # Load the pre-trained base model specified by pretrained_model_name
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.encoder.config
        hidden_size = self.config.hidden_size # Get hidden size from the base model's config
        # Define the names of the dimensions to predict
        self.dimension_names = ["Taste", "Service", "Ambiance"] # Should match training setup and regression_heads.json
        if len(self.dimension_names) != num_dimensions:
             logging.warning(f"Number of dimensions specified ({num_dimensions}) does not match hardcoded dimension names ({len(self.dimension_names)}). Using hardcoded names.")
        # Create a ModuleDict to hold the separate regression head for each dimension
        self.regression_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Dropout(dropout_prob),       # Dropout layer
                nn.Linear(hidden_size, 64),     # First linear layer (example size)
                nn.GELU(),                      # Activation function
                nn.Linear(64, 1)                # Output linear layer (predicts a single value)
            ) for dim in self.dimension_names # Use the defined names
        })
        logging.info("Custom regression heads structure created.")

    # Define the forward pass: how input data flows through the model
    def forward(self, input_ids, attention_mask=None):
        # Pass input through the base encoder
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the output corresponding to the [CLS] token as the pooled representation
        # Shape: [batch_size, hidden_size]
        pooled_output = encoder_output.last_hidden_state[:, 0]
        results = {}
        # Pass the pooled output through each dimension's regression head
        for dim in self.dimension_names:
            if dim in self.regression_heads:
                score = self.regression_heads[dim](pooled_output)
                # Apply sigmoid and scale the output to be between 1.0 and 10.0
                # score shape: [batch_size, 1]
                results[dim] = 1.0 + 9.0 * torch.sigmoid(score)
            else:
                logging.warning(f"Dimension '{dim}' not found in regression heads during forward pass.")
        return results # Returns a dictionary: {'Taste': tensor, 'Service': tensor, 'Ambiance': tensor}

# --- Model Loading and Caching ---
# Global cache for loaded models and tokenizers
loaded_models_cache = {}

def _load_model_and_tokenizer(repo_id="c0sm1c9/restaurant-review-analyzer-dutch"):
    """Loads the model and tokenizer, caching them globally."""
    if repo_id in loaded_models_cache:
        logging.info(f"Using cached model and tokenizer for {repo_id}")
        return loaded_models_cache[repo_id]

    try:
        logging.info(f"Loading tokenizer from {repo_id}...")
        # Trust remote code if necessary for some tokenizers, though usually not for standard ones
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True) 
        logging.info("Tokenizer loaded.")

        logging.info("Initializing model structure (loads base encoder weights)...")
        # Ensure trust_remote_code=True if the model config requires it
        model = RestaurantReviewAnalyzer(pretrained_model_name=repo_id, num_dimensions=3) # Match num_dimensions to class definition/heads
        logging.info("Model structure initialized.")

        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Target device selected: {model_device}")
        model.to(model_device)
        logging.info(f"Model moved to {model_device}.")

        # --- Load Custom Regression Head Weights ---
        regression_heads_filename = "regression_heads.json"
        logging.info(f"Downloading custom weights '{regression_heads_filename}' from {repo_id}...")
        try:
            regression_heads_path = hf_hub_download(
                repo_id=repo_id,
                filename=regression_heads_filename,
                # Consider adding force_download=True during development if weights change
            )
            logging.info(f"Downloaded weights file to: {regression_heads_path}")
        except Exception as download_err:
            logging.error(f"Failed to download '{regression_heads_filename}' from {repo_id}: {download_err}", exc_info=True)
            raise download_err # Propagate error

        logging.info("Loading weights from JSON file...")
        with open(regression_heads_path, 'r') as f:
            regression_heads_dict_from_json = json.load(f)
        logging.info("JSON weights data loaded.")

        # Convert the loaded data (lists) back into a PyTorch state_dict
        regression_heads_state_dict = {}
        logging.info("Converting JSON weights to tensors on target device...")
        for dim_name, params in regression_heads_dict_from_json.items():
            if dim_name in model.regression_heads:
                # Get the state_dict of the corresponding head in the *current* model
                layer_state_dict = model.regression_heads[dim_name].state_dict()
                for param_name_from_json, param_value_list in params.items():
                    # Find the matching parameter key in the model's layer state_dict
                    # This handles potential key name differences (e.g., due to ModuleDict prefixing)
                    found_key = False
                    for model_param_key in layer_state_dict.keys():
                        # Check if the model key *ends* with the key from JSON (e.g., '1.weight' vs 'Taste.1.weight')
                        # Or handle direct match if no prefixing occurs
                        if model_param_key == param_name_from_json or model_param_key.endswith("." + param_name_from_json):
                            target_dtype = layer_state_dict[model_param_key].dtype
                            target_shape = layer_state_dict[model_param_key].shape
                            try:
                                # Create the tensor directly on the target device and with the correct dtype
                                tensor_value = torch.tensor(param_value_list, dtype=target_dtype, device=model_device)
                                # Verify the number of elements matches before reshaping
                                if tensor_value.numel() != torch.Size(target_shape).numel():
                                    raise RuntimeError(f"Shape mismatch for {dim_name}.{model_param_key}: JSON ({tensor_value.numel()}) vs Model ({torch.Size(target_shape).numel()})")
                                # Reshape the tensor
                                tensor_value = tensor_value.view(target_shape)
                                # Use the *model's* full key name for the final state dict
                                regression_heads_state_dict[model_param_key] = tensor_value
                                found_key = True
                                break # Found the matching key for this JSON param
                            except Exception as tensor_err:
                                logging.error(f"Error processing tensor for {dim_name}.{model_param_key}: {tensor_err}", exc_info=True)
                                raise tensor_err # Propagate error
                    if not found_key:
                         logging.warning(f"Parameter '{param_name_from_json}' from JSON not found in model head '{dim_name}'")

            else:
                logging.warning(f"Dimension '{dim_name}' from JSON weights not found in model's regression heads.")

        # Load the constructed state_dict into the `regression_heads` part of the model
        logging.info("Applying weights to the model's regression heads...")
        try:
            # Use strict=False initially for debugging, then True for production if sure
            model.regression_heads.load_state_dict(regression_heads_state_dict, strict=True)
            logging.info("Regression head weights loaded successfully into the model.")
        except RuntimeError as load_err:
             logging.error(f"Error loading state dict into regression heads: {load_err}", exc_info=True)
             logging.error(f"Keys in generated state_dict: {list(regression_heads_state_dict.keys())}")
             logging.error(f"Keys expected by model.regression_heads: {list(model.regression_heads.state_dict().keys())}")
             raise load_err # Propagate error

        model.eval() # Set to evaluation mode immediately after loading
        logging.info("Model is ready for inference.")

        # Cache the loaded components
        loaded_models_cache[repo_id] = {"model": model, "tokenizer": tokenizer, "device": model_device}
        return loaded_models_cache[repo_id]

    except Exception as e:
        logging.error(f"CRITICAL ERROR during model loading for {repo_id}: {e}", exc_info=True)
        # Don't cache on failure
        raise e # Re-raise to indicate failure

def analyze_reviews_with_finetuned_model(restaurant_id: str, reviews: list[str]) -> dict | None:
    """
    Analyzes reviews using the locally loaded fine-tuned model c0sm1c9/restaurant-review-analyzer-dutch.

    Args:
        restaurant_id: The unique ID of the restaurant (used for logging).
        reviews: A list of review text strings.

    Returns:
        A dictionary containing the averaged scores (Taste, Service, Ambiance) on a 1-10 scale,
        or None if analysis fails.
    """
    if not reviews:
        logging.warning(f"No reviews provided for restaurant {restaurant_id}. Skipping analysis.")
        return None # Return None for empty input list

    repo_id = "c0sm1c9/restaurant-review-analyzer-dutch"
    start_time = time.time()

    try:
        # Load model and tokenizer (or get from cache)
        # This might raise an error if loading fails, which is caught below
        model_components = _load_model_and_tokenizer(repo_id)
        model = model_components["model"]
        tokenizer = model_components["tokenizer"]
        device = model_components["device"]

        logging.info(f"Analyzing {len(reviews)} reviews for restaurant {restaurant_id} using {repo_id} on {device}...")

        # --- Batch Inference ---
        # Tokenize the batch of reviews
        # Ensure padding=True to handle variable lengths, truncation=True to prevent overflow
        inputs = tokenizer(
            reviews,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 512 # Use model's max length if available
        )

        # Move inputs to the model's device
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as move_err:
             logging.error(f"Error moving tensors to device {device} for {restaurant_id}: {move_err}", exc_info=True)
             return None # Cannot proceed if tensors aren't on the right device

        # Perform inference without calculating gradients
        with torch.no_grad():
            try:
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                # outputs is dict: {'Taste': tensor([[...], [...]]), 'Service': tensor([[...], [...]]), ...}
                # Each tensor has shape [batch_size, 1]
            except Exception as inference_err:
                 logging.error(f"Error during model inference for {restaurant_id}: {inference_err}", exc_info=True)
                 return None # Inference failed

        # Calculate average scores across the batch
        avg_scores = {}
        expected_dims = model.dimension_names
        for dim in expected_dims:
            if dim in outputs and outputs[dim].numel() > 0: # Check if dim exists and tensor is not empty
                score_tensor = outputs[dim]
                try:
                    # Calculate mean, convert to float, round to 1 decimal place
                    avg_score = round(score_tensor.mean().item(), 1)
                    avg_scores[dim] = avg_score
                except Exception as avg_err:
                     logging.error(f"Error calculating average for dimension '{dim}' for {restaurant_id}: {avg_err}", exc_info=True)
                     avg_scores[dim] = None # Mark dimension as failed
            else:
                 logging.warning(f"Dimension '{dim}' missing or empty in model output for {restaurant_id}.")
                 avg_scores[dim] = None # Mark dimension as failed

        # Filter out None scores if any dimension failed calculation
        final_scores = {k: v for k, v in avg_scores.items() if v is not None}

        if len(final_scores) != len(expected_dims):
             logging.warning(f"Could not calculate valid scores for all dimensions for {restaurant_id}. Results: {final_scores}")
             # Return partial results only if at least one score was calculated
             if not final_scores:
                  logging.error(f"No valid scores could be calculated for {restaurant_id}.")
                  return None

        elapsed_time = time.time() - start_time
        logging.info(f"Successfully calculated average scores for {restaurant_id} in {elapsed_time:.2f}s: {final_scores}")
        return final_scores
        # --- End Batch Inference ---

    except Exception as e:
        # Catch errors from _load_model_and_tokenizer or other unexpected issues
        logging.error(f"FATAL Error analyzing reviews for {restaurant_id} using local model {repo_id}: {e}", exc_info=True)
        return None # Return None on any failure


if __name__ == '__main__':
    # Example usage for testing this utility directly
    # This part no longer needs environment variables for the URL/Key
    print("Testing local fine-tuned review analyzer...")

    # --- Test Case 1: Basic ---
    test_id_1 = "restaurant_test_1"
    test_reviews_1 = [
        "Heerlijk gegeten bij dit restaurant! De service was top en de sfeer gezellig.", # Dutch: Good food, service, ambiance
        "Service was slow but the pasta was worth it.", # English: Ok service, good food
        "A bit pricey, but the ambiance is fantastic.", # English: Ok value, great ambiance
        "Zeer teleurstellend, smaakloos eten en onvriendelijke bediening.", # Dutch: Bad food, bad service
        "Decor is mooi, maar het eten was koud.", # Dutch: Good ambiance, bad food
    ]
    print(f"\n--- Testing Case 1: {test_id_1} ({len(test_reviews_1)} reviews) ---")
    scores_1 = analyze_reviews_with_finetuned_model(test_id_1, test_reviews_1)
    if scores_1:
        print(f"Received scores: {scores_1}")
    else:
        print("Failed to get scores.")

    # --- Test Case 2: Empty List ---
    test_id_2 = "restaurant_test_2"
    test_reviews_2 = []
    print(f"\n--- Testing Case 2: {test_id_2} ({len(test_reviews_2)} reviews) ---")
    scores_2 = analyze_reviews_with_finetuned_model(test_id_2, test_reviews_2)
    if scores_2 is None:
        print("Correctly returned None for empty review list.")
    else:
        print(f"ERROR: Should have returned None, but got: {scores_2}")

    # --- Test Case 3: Single Review ---
    test_id_3 = "restaurant_test_3"
    test_reviews_3 = ["Just okay, nothing special really."]
    print(f"\n--- Testing Case 3: {test_id_3} ({len(test_reviews_3)} reviews) ---")
    scores_3 = analyze_reviews_with_finetuned_model(test_id_3, test_reviews_3)
    if scores_3:
        print(f"Received scores: {scores_3}")
    else:
        print("Failed to get scores.")

    # --- Test Case 4: Multiple calls (test caching) ---
    print("\n--- Testing Case 4: Multiple calls (should use cache) ---")
    start_cache_test = time.time()
    scores_4a = analyze_reviews_with_finetuned_model(test_id_1, test_reviews_1[:2]) # Reuse test_id_1
    scores_4b = analyze_reviews_with_finetuned_model("another_id", test_reviews_1[2:])
    end_cache_test = time.time()
    print(f"Second set of calls finished in {end_cache_test - start_cache_test:.2f}s (should be faster if cached).")
    print(f"Scores 4a: {scores_4a}")
    print(f"Scores 4b: {scores_4b}")

    print("\n--- Testing Complete ---") 