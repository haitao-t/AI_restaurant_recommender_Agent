# Implementation for the fine-tuned restaurant review analyzer model
import logging
import time
import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- RestaurantReviewAnalyzer Model Class Definition ---
class RestaurantReviewAnalyzer(nn.Module):
    """
    A custom model that uses a pre-trained transformer encoder (like XLM-RoBERTa)
    and adds separate regression heads to predict scores for different dimensions
    of a restaurant review (Taste, Service, Ambiance).
    """
    def __init__(self, pretrained_model_name="xlm-roberta-base", num_dimensions=3, dropout_prob=0.1):
        super().__init__()
        logger.info(f"Initializing custom model structure with base: {pretrained_model_name}")
        # Load the pre-trained base model specified by pretrained_model_name
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.encoder.config
        hidden_size = self.config.hidden_size # Get hidden size from the base model's config
        # Define the names of the dimensions to predict
        self.dimension_names = ["Taste", "Service", "Ambiance"] # Should match training setup
        # Create a ModuleDict to hold the separate regression head for each dimension
        self.regression_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Dropout(dropout_prob),       # Dropout layer
                nn.Linear(hidden_size, 64),  # First linear layer
                nn.GELU(),                   # Activation function
                nn.Linear(64, 1)              # Output linear layer (predicts a single value)
            ) for dim in self.dimension_names[:num_dimensions]
        })
        logger.info("Custom regression heads structure created.")

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
            score = self.regression_heads[dim](pooled_output)
            # Apply sigmoid and scale the output to be between 1.0 and 10.0
            results[dim] = 1.0 + 9.0 * torch.sigmoid(score)
        return results

# --- Global variables to store model and tokenizer ---
MODEL = None
TOKENIZER = None
MODEL_DEVICE = None

# --- Hugging Face Model Configuration ---
MODEL_REPO_ID = "c0sm1c9/restaurant-review-analyzer-dutch"

def load_model_from_hub():
    """Load the model and tokenizer from Hugging Face Hub if not already loaded."""
    global MODEL, TOKENIZER, MODEL_DEVICE
    
    if MODEL is not None and TOKENIZER is not None:
        return

    try:
        logger.info(f"Loading restaurant review analyzer model from {MODEL_REPO_ID}...")
        
        # Select device (GPU if available, otherwise CPU)
        MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {MODEL_DEVICE}")
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
        logger.info("Tokenizer loaded successfully")
        
        # Initialize model structure
        MODEL = RestaurantReviewAnalyzer(pretrained_model_name=MODEL_REPO_ID)
        MODEL.to(MODEL_DEVICE)
        logger.info(f"Model structure initialized and moved to {MODEL_DEVICE}")

        # Load custom regression head weights
        regression_heads_filename = "regression_heads.json"
        regression_heads_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=regression_heads_filename
        )
        logger.info(f"Downloaded weights file to: {regression_heads_path}")

        # Load and convert weights
        with open(regression_heads_path, 'r') as f:
            regression_heads_dict_from_json = json.load(f)
        
        # Convert the loaded data (lists) back into a PyTorch state_dict
        regression_heads_state_dict = {}
        for dim_name, params in regression_heads_dict_from_json.items():
            if dim_name in MODEL.regression_heads:
                layer_state_dict = MODEL.regression_heads[dim_name].state_dict()
                for param_name, param_value_list in params.items():
                    for model_param_key in layer_state_dict.keys():
                        if model_param_key == param_name or model_param_key.endswith("." + param_name):
                            target_dtype = layer_state_dict[model_param_key].dtype
                            target_shape = layer_state_dict[model_param_key].shape
                            tensor_value = torch.tensor(param_value_list, dtype=target_dtype, device=MODEL_DEVICE)
                            if tensor_value.numel() != target_shape.numel():
                                raise RuntimeError(f"Shape mismatch for {dim_name}.{model_param_key}")
                            tensor_value = tensor_value.view(target_shape)
                            regression_heads_state_dict[f"{dim_name}.{model_param_key}"] = tensor_value
                            break

        # Load the state dict into the regression heads
        MODEL.regression_heads.load_state_dict(regression_heads_state_dict, strict=True)
        logger.info("Regression head weights loaded successfully")
        
        # Set model to evaluation mode
        MODEL.eval()
        logger.info("Model ready for inference")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        MODEL = None
        TOKENIZER = None
        raise

def analyze_reviews_with_finetuned_model(restaurant_id: str, reviews: list) -> dict | None:
    """
    Analyzes restaurant reviews using the fine-tuned model to get dimensional scores.

    Args:
        restaurant_id: The unique ID of the restaurant.
        reviews: A list of review text strings or dictionaries for this restaurant.

    Returns:
        A dictionary containing the scores (1-10 scale) for three dimensions:
        {"Taste": 8.5, "Service": 7.0, "Ambiance": 9.1}
        Returns None if the model fails or returns invalid data.
    """
    try:
        # Load model if not already loaded
        if MODEL is None or TOKENIZER is None:
            load_model_from_hub()
        
        # Process reviews to extract text
        review_texts = []
        for review in reviews:
            if isinstance(review, dict) and 'text' in review:
                review_texts.append(review['text'])
            elif isinstance(review, str):
                review_texts.append(review)
        
        # Filter out empty reviews
        review_texts = [text for text in review_texts if text and len(text.strip()) > 0]
        
        if not review_texts:
            logger.warning(f"No valid review texts found for restaurant {restaurant_id}")
            return None
        
        logger.info(f"Analyzing {len(review_texts)} reviews for restaurant {restaurant_id}")
        
        # Combine all reviews into one text to get an overall score
        # For a more sophisticated approach, you could analyze each review separately and average the scores
        combined_text = " ".join(review_texts[:10])  # Limit to first 10 reviews to avoid exceeding token limit
        
        # Tokenize the review text
        inputs = TOKENIZER(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(MODEL_DEVICE) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = MODEL(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # Convert tensor outputs to Python native types
        scores = {}
        for dim, score_tensor in outputs.items():
            scores[dim] = round(score_tensor.item(), 1)
            
        logger.info(f"Successfully analyzed reviews for {restaurant_id} (10-point scale): {scores}")
        return scores
        
    except Exception as e:
        logger.error(f"Error analyzing reviews for {restaurant_id}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example usage for testing this utility directly
    print("Testing restaurant review analyzer...")
    
    test_id = "example_restaurant"
    test_reviews = [
        "Heerlijk gegeten bij dit restaurant! De service was top en de sfeer gezellig.",
        "Service was slow but the pasta was worth it.",
        "A bit pricey for what you get, but the ambiance is nice."
    ]
    
    print(f"Testing review analysis for restaurant ID: {test_id}")
    print(f"Sample reviews: {test_reviews[:2]}...")
    
    scores = analyze_reviews_with_finetuned_model(test_id, test_reviews)
    if scores:
        print("\nAnalysis results:")
        for dimension, score in scores.items():
            print(f"  {dimension}: {score}")
    else:
        print("Failed to analyze reviews.") 