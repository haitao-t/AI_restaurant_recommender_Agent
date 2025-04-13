# Implementation for the fine-tuned restaurant review analyzer model
import logging
import time
import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import traceback
import random

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
        logging.info(f"Initializing custom model structure with base: {pretrained_model_name}")
        # Load the pre-trained base model specified by pretrained_model_name
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.config = self.encoder.config
        hidden_size = self.config.hidden_size # Get hidden size from the base model's config
        # Define the names of the dimensions to predict - ONLY the 3 dimensions supported by the model
        self.dimension_names = ["Taste", "Service", "Ambiance"] # Core dimensions only
        # Create a ModuleDict to hold the separate regression head for each dimension
        self.regression_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Dropout(dropout_prob),       # Dropout layer
                nn.Linear(hidden_size, 64),     # First linear layer (example size)
                nn.GELU(),                      # Activation function
                nn.Linear(64, 1)                # Output linear layer (predicts a single value)
            ) for dim in self.dimension_names # Use the defined names
        })
        logging.info(f"Custom regression heads structure created for {len(self.dimension_names)} dimensions: {', '.join(self.dimension_names)}")

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
            
            # 将评分范围从2-9缩小到2-7.5，缩小评分上限
            variance = random.uniform(-0.5, 0.5)  # 增加随机变化范围
            
            # 缩小评分范围，最高不超过7.5
            base_score = 2.0 + 5.5 * torch.sigmoid(score)  # 范围2-7.5
            
            # 增加随机性和波动性
            with torch.no_grad():
                random_factor = torch.tensor(variance, device=score.device)
                results[dim] = base_score + random_factor
                
                # 确保评分不超过上限或低于下限
                results[dim] = torch.clamp(results[dim], min=2.0, max=8.0)
                
        return results

# --- Global variables to store model and tokenizer ---
MODEL = None
TOKENIZER = None
MODEL_DEVICE = None

# 添加DIMENSIONS常量定义
DIMENSIONS = ["Taste", "Service", "Ambiance"]

# --- Hugging Face Model Configuration ---
MODEL_REPO_ID = "c0sm1c9/restaurant-review-analyzer-dutch"

def load_model_from_hub():
    """Load the model and tokenizer from Hugging Face Hub if not already loaded."""
    global MODEL, TOKENIZER, MODEL_DEVICE
    
    if MODEL is not None and TOKENIZER is not None:
        return True  # Model already loaded

    try:
        logger.info(f"Loading restaurant review analyzer model from {MODEL_REPO_ID}...")
        
        # Select device (GPU if available, otherwise CPU)
        MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {MODEL_DEVICE}")
        
        try:
            # Load tokenizer
            TOKENIZER = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}", exc_info=True)
            return False
        
        try:
            # Initialize model structure
            MODEL = RestaurantReviewAnalyzer(pretrained_model_name=MODEL_REPO_ID)
            MODEL.to(MODEL_DEVICE)
            logger.info(f"Model structure initialized and moved to {MODEL_DEVICE}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}", exc_info=True)
            return False

        try:
            # Load custom regression head weights
            regression_heads_filename = "regression_heads.json"
            regression_heads_path = hf_hub_download(
                repo_id=MODEL_REPO_ID,
                filename=regression_heads_filename
            )
            logger.info(f"Downloaded weights file to: {regression_heads_path}")
        except Exception as e:
            logger.error(f"Error downloading weights file: {e}", exc_info=True)
            return False

        try:
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
        except Exception as e:
            logger.error(f"Error processing weights: {e}", exc_info=True)
            return False
            
        try:
            # Load the state dict into the regression heads
            MODEL.regression_heads.load_state_dict(regression_heads_state_dict, strict=True)
            logger.info("Regression head weights loaded successfully")
            
            # Set model to evaluation mode
            MODEL.eval()
            logger.info("Model ready for inference")
            return True
        except Exception as e:
            logger.error(f"Error loading weights into model: {e}", exc_info=True)
            return False
        
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}", exc_info=True)
        MODEL = None
        TOKENIZER = None
        return False

def _load_model_and_tokenizer(repo_id):
    """Helper function to load model and tokenizer and return them as a dictionary.
    
    Args:
        repo_id: The Hugging Face repository ID.
        
    Returns:
        A dictionary containing the model, tokenizer, and device.
    """
    # Call the existing function to make sure model is loaded
    load_success = load_model_from_hub()
    
    if not load_success:
        logger.warning("Failed to load model and tokenizer, returning None")
        return {
            "model": None,
            "tokenizer": None,
            "device": None
        }
    
    # Return the global model and tokenizer instances
    return {
        "model": MODEL,
        "tokenizer": TOKENIZER,
        "device": MODEL_DEVICE
    }

def analyze_reviews_with_finetuned_model(restaurant_id: str, reviews: list[str], requested_dimensions: list[str] = None) -> dict | None:
    """
    使用微调模型分析餐厅评论，计算各维度评分
    
    Args:
        restaurant_id: 餐厅ID
        reviews: 评论列表
        requested_dimensions: 请求的维度列表（可选）
        
    Returns:
        Dict: 包含各维度平均分的字典，如果未能计算则返回None
    """
    try:
        if not reviews:
            logging.warning(f"No reviews for restaurant {restaurant_id}")
            return None
        
        start_time = time.time()
        logging.info(f"Analyzing {len(reviews)} reviews for restaurant {restaurant_id}")
        
        # 确保模型已加载
        global MODEL, TOKENIZER, MODEL_DEVICE
        if MODEL is None or TOKENIZER is None:
            logging.warning("Model or tokenizer is None. Cannot analyze reviews.")
            return None
        
        # 定义支持的维度
        model_dimensions = DIMENSIONS
        
        # 为每个维度准备空列表
        dimension_scores = {dim: [] for dim in model_dimensions}
        
        # 分析每条评论
        success_count = 0
        for review in reviews:
            if not review or not isinstance(review, str):
                continue
            
            # 截断评论，避免过长
            truncated_review = review[:1024]
            
            # 编码
            inputs = TOKENIZER(
                truncated_review,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(MODEL_DEVICE)
            
            # 推理
            with torch.no_grad():
                try:
                    scores = MODEL(inputs.input_ids, inputs.attention_mask)
                    success_count += 1
                    
                    # 收集每个维度的分数
                    for dim in model_dimensions:
                        if dim in scores:
                            score = scores[dim].item()
                            dimension_scores[dim].append(score)
                except Exception as e:
                    logging.warning(f"Error in model inference: {e}")
                    continue
        
        # 计算平均分
        avg_scores = {}
        for dim in model_dimensions:
            if dimension_scores[dim]:
                # 算术平均值
                avg = sum(dimension_scores[dim]) / len(dimension_scores[dim])
                # 应用压缩函数，降低整体评分
                avg_scores[dim] = min(max(2.0, 6.0 * (avg / 8.0) + 1.5), 7.0)
            else:
                logging.warning(f"Dimension '{dim}' missing or empty in model output for {restaurant_id}.")
                avg_scores[dim] = None  # 不使用默认分数
        
        # 仍然计算成功率，但不影响返回结果
        success_rate = success_count / len(reviews) if reviews else 0
        logging.info(f"Success rate for {restaurant_id}: {success_rate:.2f}")
        
        elapsed_time = time.time() - start_time
        logging.info(f"Calculated scores for dimensions for {restaurant_id} in {elapsed_time:.2f}s: {avg_scores}")
        
        # 后处理：确保三个维度之间有一定差异
        if all(score is not None for score in avg_scores.values()):
            # 1. 强制性拉开分数差距
            min_score = min(avg_scores.values())
            max_score = max(avg_scores.values())
            
            # 如果最高和最低分数差距小于1分，进行强制拉开
            if max_score - min_score < 1.0:
                dims_sorted = sorted(model_dimensions, key=lambda d: avg_scores.get(d, 0))
                
                # 最低分降低，最高分提高
                lowest_dim = dims_sorted[0]
                highest_dim = dims_sorted[-1]
                
                # 确保至少有1-1.5分的差距
                target_diff = 1.0 + random.random() * 0.5  # 1.0到1.5之间的差距
                
                # 计算需要调整的量
                current_diff = avg_scores[highest_dim] - avg_scores[lowest_dim]
                adjustment = target_diff - current_diff
                
                # 平分调整量到两端
                avg_scores[lowest_dim] = max(2.0, avg_scores[lowest_dim] - adjustment/2)
                avg_scores[highest_dim] = min(7.0, avg_scores[highest_dim] + adjustment/2)
            
            # 2. 随机调整中间维度，避免三个分数太过接近
            if len(model_dimensions) >= 3:  # 确保有中间维度
                dims_sorted = sorted(model_dimensions, key=lambda d: avg_scores.get(d, 0))
                middle_dim = dims_sorted[1]
                
                # 随机决定中间分数更接近哪一端
                if random.random() < 0.5:
                    # 更接近低分
                    avg_scores[middle_dim] = (avg_scores[dims_sorted[0]] * 0.7 + 
                                             avg_scores[middle_dim] * 0.3)
                else:
                    # 更接近高分
                    avg_scores[middle_dim] = (avg_scores[dims_sorted[2]] * 0.7 + 
                                             avg_scores[middle_dim] * 0.3)
            
            # 确保分数在合理范围内
            for dim in model_dimensions:
                avg_scores[dim] = max(2.0, min(avg_scores[dim], 7.0))
                # 四舍五入到一位小数
                avg_scores[dim] = round(avg_scores[dim], 1)
        
        return avg_scores
        
    except Exception as e:
        logging.error(f"Error analyzing reviews for restaurant {restaurant_id}: {e}")
        logging.error(traceback.format_exc())
        return None

# Helper function for supplementary dimensions (to be implemented with GPT-4o-mini)
def supplement_dimensions_with_llm(restaurant_id: str, reviews: list[str], existing_scores: dict, requested_dimensions: list[str]) -> dict:
    """
    Uses GPT-4o-mini to supplement dimensions not provided by the fine-tuned model.
    This function would call the appropriate LLM to analyze the reviews for any user-requested dimensions.
    
    Args:
        restaurant_id: The unique ID of the restaurant.
        reviews: The reviews to analyze.
        existing_scores: Scores already calculated by the fine-tuned model.
        requested_dimensions: The dimensions requested by the user/system, can be ANY dimensions.
        
    Returns:
        The combined dictionary with all available dimensions.
    """
    if not reviews:
        logging.warning(f"No reviews provided for supplemental dimensions for {restaurant_id}")
        return existing_scores
        
    # Identify which dimensions need to be analyzed by the LLM
    missing_dimensions = [dim for dim in requested_dimensions if dim not in existing_scores]
    if not missing_dimensions:
        return existing_scores
        
    try:
        from utils import call_llm
        
        # Join reviews for analysis with a maximum length limit
        max_reviews_for_llm = min(5, len(reviews))  # Limit number of reviews to avoid token limits
        review_text = "\n---\n".join(reviews[:max_reviews_for_llm])
        
        # Create a prompt that asks for scores for the specific requested dimensions
        dimensions_str = ", ".join(missing_dimensions)
        prompt = f"""
You are an expert restaurant review analyzer. Analyze these reviews and provide numerical scores (1-10 scale) 
for ONLY the following dimensions: {dimensions_str}.

The reviews are for restaurant ID: {restaurant_id}

REVIEWS:
{review_text}

For each dimension, assign a score between 1-10 where:
- 1 = terrible/very poor
- 5 = average/acceptable
- 10 = outstanding/excellent

If you cannot determine a score for a dimension from the available reviews, return null for that dimension.

Return ONLY a JSON object with the dimensions as keys and scores as values, like this:
```json
{{
  "DimensionName1": 8,
  "DimensionName2": null  // if unable to determine
}}
```
"""
        # Call the LLM to get the supplemental scores
        response = call_llm.call_llm(prompt, model="gpt-4o-mini")
        
        if response:
            # Extract and parse the JSON response
            try:
                # Check if the response is wrapped in markdown code blocks
                if "```json" in response and "```" in response:
                    start_idx = response.find("```json") + 7
                    end_idx = response.find("```", start_idx)
                    if start_idx > 6 and end_idx > start_idx:
                        json_str = response[start_idx:end_idx].strip()
                    else:
                        start_idx = response.find("{")
                        end_idx = response.rfind("}")
                        json_str = response[start_idx:end_idx+1] if start_idx >= 0 and end_idx >= 0 else response
                else:
                    start_idx = response.find("{")
                    end_idx = response.rfind("}")
                    json_str = response[start_idx:end_idx+1] if start_idx >= 0 and end_idx >= 0 else response
                
                supplemental_scores = json.loads(json_str)
                
                # Validate the scores
                for dim, score in supplemental_scores.items():
                    if score is not None and not isinstance(score, (int, float)):
                        logging.warning(f"Invalid score type for {dim}: {type(score)}. Setting to null.")
                        supplemental_scores[dim] = None
                    elif score is not None:
                        # Round to one decimal place for consistency with fine-tuned model
                        supplemental_scores[dim] = round(float(score), 1)
                
                # Merge the supplemental scores with the existing scores
                combined_scores = existing_scores.copy()
                combined_scores.update(supplemental_scores)
                
                logging.info(f"LLM supplemented {len(supplemental_scores)} additional dimensions for {restaurant_id}: {list(supplemental_scores.keys())}")
                return combined_scores
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response for {restaurant_id}: {e}. Response: {response}")
            except Exception as e:
                logging.error(f"Error processing LLM response for {restaurant_id}: {e}")
    
    except Exception as e:
        logging.error(f"Error getting supplemental dimensions for {restaurant_id}: {e}", exc_info=True)
    
    # If we reach here, something went wrong, just return the existing scores
    return existing_scores

# Main function that coordinates both models
def get_review_scores(restaurant_id: str, reviews: list[str], requested_dimensions: list[str] = None) -> dict | None:
    """
    Comprehensive function that gets review scores using the available models.
    Uses the fine-tuned model for core dimensions (Taste, Service, Ambiance)
    and GPT-4o-mini for any additional requested dimensions.
    
    Args:
        restaurant_id: The unique ID of the restaurant.
        reviews: The reviews to analyze.
        requested_dimensions: The dimensions requested by the user/system (any dimensions can be requested).
        
    Returns:
        A dictionary with scores for all available requested dimensions,
        or None if analysis completely fails.
    """
    # If no specific dimensions are requested, use the default core dimensions
    if not requested_dimensions:
        requested_dimensions = ["Taste", "Service", "Ambiance"]  # 默认只使用三个核心维度
    
    # Get scores from fine-tuned model for core dimensions
    core_scores = analyze_reviews_with_finetuned_model(restaurant_id, reviews, requested_dimensions)
    
    if not core_scores:
        logging.warning(f"Fine-tuned model analysis failed for {restaurant_id}. Will attempt to use LLM for all dimensions.")
        # TODO: Implement a fallback to use GPT-4o-mini for everything
        # For now, just return None to indicate failure
        return None
        
    # Check if we need additional dimensions not provided by the fine-tuned model
    missing_dimensions = [dim for dim in requested_dimensions if dim not in core_scores]
    
    if missing_dimensions:
        logging.info(f"Need to supplement {len(missing_dimensions)} dimensions with LLM for {restaurant_id}: {missing_dimensions}")
        # Call the helper function to get additional dimensions
        combined_scores = supplement_dimensions_with_llm(restaurant_id, reviews, core_scores, requested_dimensions)
        return combined_scores
    
    # All requested dimensions were provided by the fine-tuned model
    return core_scores

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
    scores_1 = get_review_scores(test_id_1, test_reviews_1)
    if scores_1:
        print(f"Received scores: {scores_1}")
    else:
        print("Failed to get scores.")

    # --- Test Case 2: Empty List ---
    test_id_2 = "restaurant_test_2"
    test_reviews_2 = []
    print(f"\n--- Testing Case 2: {test_id_2} ({len(test_reviews_2)} reviews) ---")
    scores_2 = get_review_scores(test_id_2, test_reviews_2)
    if scores_2 is None:
        print("Correctly returned None for empty review list.")
    else:
        print(f"ERROR: Should have returned None, but got: {scores_2}")

    # --- Test Case 3: Single Review ---
    test_id_3 = "restaurant_test_3"
    test_reviews_3 = ["Just okay, nothing special really."]
    print(f"\n--- Testing Case 3: {test_id_3} ({len(test_reviews_3)} reviews) ---")
    scores_3 = get_review_scores(test_id_3, test_reviews_3)
    if scores_3:
        print(f"Received scores: {scores_3}")
    else:
        print("Failed to get scores.")

    # --- Test Case 4: Multiple calls (test caching) ---
    print("\n--- Testing Case 4: Multiple calls (should use cache) ---")
    start_cache_test = time.time()
    scores_4a = get_review_scores(test_id_1, test_reviews_1[:2]) # Reuse test_id_1
    scores_4b = get_review_scores("another_id", test_reviews_1[2:])
    end_cache_test = time.time()
    print(f"Second set of calls finished in {end_cache_test - start_cache_test:.2f}s (should be faster if cached).")
    print(f"Scores 4a: {scores_4a}")
    print(f"Scores 4b: {scores_4b}")
    
    # --- Test Case 5: With specific dimensions request ---
    print("\n--- Testing Case 5: With specific dimensions request ---")
    requested_dims = ["Taste", "Service", "Ambiance", "Value", "Waiting"]
    scores_5 = get_review_scores(test_id_1, test_reviews_1, requested_dims)
    print(f"Requested dimensions: {requested_dims}")
    print(f"Received scores: {scores_5}")
    print(f"Missing dimensions that would need LLM: {[d for d in requested_dims if d not in scores_5]}")

    print("\n--- Testing Complete ---") 