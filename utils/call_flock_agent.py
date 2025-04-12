# Utility for calling the Flock Agent model or OpenAI as fallback

import logging
import json
import os
import time
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import requests

# Only import torch and transformers if Flock model is used
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration ---
# Flag to determine whether to use Flock model or fallback to OpenAI
USE_FLOCK_MODEL = os.environ.get("USE_FLOCK_MODEL", "false").lower() == "true"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Flock Model Configuration ---
FLOCK_MODEL_NAME = "flock-io/Flock_Web3_Agent_Model"
# Global variables for model and tokenizer to avoid reloading on every call
flock_model = None
flock_tokenizer = None
device = None
flock_available = False

# Set device only if using Flock model and torch is available
if USE_FLOCK_MODEL and TRANSFORMERS_AVAILABLE:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_flock_model():
    """
    Loads the Flock model if it's available and configured.
    Returns True if the model was successfully loaded, False otherwise.
    """
    global flock_model, flock_tokenizer, flock_available
    
    if not USE_FLOCK_MODEL:
        logger.info("Flock model usage is disabled by configuration")
        return False
    
    try:
        logger.info("Attempting to load Flock model...")
        
        # Import necessary libraries
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Check if CUDA is available and set the appropriate device
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            device = "cuda"
        else:
            logger.info("CUDA is not available, using CPU")
            device = "cpu"
        
        # Load the model and tokenizer
        model_name = "irhambuckle/flock-agent-instruct-llama3-8b"
        
        flock_tokenizer = AutoTokenizer.from_pretrained(model_name)
        flock_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Move model to the appropriate device if not using device_map="auto"
        if device == "cpu" or (device == "cuda" and flock_model.device.type == "cpu"):
            flock_model = flock_model.to(device)
        
        logger.info(f"Successfully loaded Flock model on {flock_model.device}")
        flock_available = True
        return True
        
    except ImportError as e:
        logger.warning(f"Could not import required libraries for Flock model: {e}")
        flock_available = False
        return False
    except Exception as e:
        logger.error(f"Error loading Flock model: {e}", exc_info=True)
        flock_available = False
        return False

def call_openai_for_action(messages: List[Dict[str, str]], tools: List[Dict]) -> Optional[Dict]:
    """
    Calls the OpenAI API to determine the next action or retrieve Web3 information.
    
    Args:
        messages: A list of message dictionaries to send to the OpenAI API.
        tools: A list of tool definitions that the model can use.
        
    Returns:
        A dictionary containing the chosen action, or None if an error occurred.
    """
    try:
        import openai
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using GPT-4o-mini by default
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Extract the message content
        response_message = response.choices[0].message
        
        # Check if there's a tool call
        if response_message.tool_calls and len(response_message.tool_calls) > 0:
            tool_call = response_message.tool_calls[0]
            
            # Parse the function call
            function_call = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
            
            return function_call
            
        return None
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return None

def decide_next_action_with_flock(context: str, tools: List[Dict]) -> Optional[Dict]:
    """
    Decides the next action to take using OpenAI's GPT-4o-mini model.
    The function name is kept for backward compatibility.
    
    Args:
        context: A string describing the current conversation context and state.
        tools: A list of tool definitions that the model can use.
        
    Returns:
        A dictionary containing the chosen action, or None if an error occurred.
    """
    # Always use OpenAI for decision making, regardless of USE_FLOCK_MODEL setting
    logger.info("Using OpenAI GPT-4o-mini to decide next action")
    
    # Format the messages for OpenAI
    messages = [
        {"role": "system", "content": "You are an AI assistant helping with restaurant recommendations. Based on the context provided, choose the most appropriate next action from the available tools."},
        {"role": "user", "content": context}
    ]
    
    # Call the OpenAI API
    return call_openai_for_action(messages, tools)

def get_restaurant_web3_info(restaurants: List[Dict], max_restaurants: int = 5, timeout: int = 10) -> List[Dict]:
    """
    Enriches restaurant data with Web3-related information such as cryptocurrency 
    acceptance, token programs, and NFT rewards.
    
    Args:
        restaurants: A list of restaurant dictionaries to enrich with Web3 information.
        max_restaurants: Maximum number of restaurants to process (default: 5).
        timeout: Maximum time in seconds to wait for each model call (default: 10).
        
    Returns:
        A list of restaurant dictionaries enriched with Web3 information.
    """
    # Initialize Flock model if it's enabled and not already loaded
    global flock_available, flock_model, flock_tokenizer
    
    if USE_FLOCK_MODEL and not flock_available and flock_model is None:
        flock_available = load_flock_model()
    
    # Define the Web3 tools that can be used
    web3_tools = [
        {
            "type": "function", 
            "function": {
                "name": "add_crypto_payment_info", 
                "description": "Add information about cryptocurrency payment acceptance for a restaurant",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "restaurant_id": {"type": "string"},
                        "accepts_crypto": {"type": "boolean"},
                        "crypto_payment_details": {"type": "string"}
                    }
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "add_token_program_info", 
                "description": "Add information about token-based loyalty or membership programs for a restaurant",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "restaurant_id": {"type": "string"},
                        "has_token_program": {"type": "boolean"},
                        "token_program_details": {"type": "string"}
                    }
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "add_nft_rewards_info", 
                "description": "Add information about NFT-based rewards or collectibles for a restaurant",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "restaurant_id": {"type": "string"},
                        "has_nft_rewards": {"type": "boolean"},
                        "nft_rewards_details": {"type": "string"}
                    }
                }
            }
        }
    ]

    # Limit the number of restaurants to process for performance
    if len(restaurants) > max_restaurants:
        logger.info(f"Limiting Web3 information processing to top {max_restaurants} restaurants (out of {len(restaurants)})")
        limited_restaurants = restaurants[:max_restaurants]
    else:
        limited_restaurants = restaurants
        
    # Keep the original list intact with default values
    enriched_restaurants = []
    for restaurant in restaurants:
        enriched_restaurant = restaurant.copy()
        
        # Default Web3 information
        enriched_restaurant.update({
            "accepts_crypto": False,
            "crypto_payment_details": "",
            "has_token_program": False,
            "token_program_details": "",
            "has_nft_rewards": False,
            "nft_rewards_details": ""
        })
        
        enriched_restaurants.append(enriched_restaurant)
    
    # Only process the limited set of restaurants
    for idx, restaurant in enumerate(limited_restaurants):
        restaurant_name = restaurant.get('name', 'Unknown Restaurant')
        restaurant_id = restaurant.get('id', 'unknown')
        
        logger.info(f"Enriching restaurant with Web3 information ({idx+1}/{len(limited_restaurants)}): {restaurant_name}")
        
        # Prepare context for the model with restaurant details
        restaurant_context = (
            f"Restaurant ID: {restaurant_id}\n"
            f"Name: {restaurant_name}\n"
            f"Location: {restaurant.get('formatted_address', restaurant.get('vicinity', 'Unknown location'))}\n"
            f"Restaurant Types: {', '.join(restaurant.get('types', []))}\n\n"
            f"Determine if this restaurant has Web3 features such as cryptocurrency payment acceptance, "
            f"token-based loyalty programs, or NFT rewards, based on available information. "
            f"Provide the most likely assessment with reasonable details."
        )
        
        function_calls = []
        
        # Try with Flock model first if available
        if USE_FLOCK_MODEL and flock_available and flock_model is not None and flock_tokenizer is not None:
            try:
                # Set a timeout for the Flock model generation
                import threading
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                
                def generate_with_flock():
                    tools_json_string = json.dumps(web3_tools, ensure_ascii=False)
                    
                    messages = [
                        {"role": "system", "content": "You are a Web3 restaurant information assistant. Your task is to analyze restaurant data and determine if they likely have Web3 features. Make reasonable inferences based on the restaurant's name, type, and location. Use the following functions to add Web3 information - " + tools_json_string},
                        {"role": "user", "content": restaurant_context}
                    ]
                    
                    text = flock_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    model_inputs = flock_tokenizer([text], return_tensors="pt").to(flock_model.device)
                    
                    generated_ids = flock_model.generate(
                        **model_inputs,
                        max_new_tokens=500,
                        pad_token_id=flock_tokenizer.eos_token_id
                    )
                    
                    # Decode the generated part only
                    input_ids_len = model_inputs.input_ids.shape[1]
                    generated_part_ids = generated_ids[0, input_ids_len:]
                    response_string = flock_tokenizer.decode(generated_part_ids, skip_special_tokens=True)
                    
                    # Clean and parse the response
                    if response_string.strip().startswith("```json"):
                        response_string = response_string.strip()[7:-3].strip()
                    elif response_string.strip().startswith("```"):
                        response_string = response_string.strip()[3:-3].strip()
                        
                    return json.loads(response_string)
                
                # Execute with timeout
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(generate_with_flock)
                    try:
                        function_calls = future.result(timeout=timeout)
                        logger.info(f"Successfully generated Web3 information with Flock model for {restaurant_name}")
                    except TimeoutError:
                        logger.warning(f"Flock model generation timed out after {timeout}s for {restaurant_name}")
                        function_calls = []
                    except Exception as e:
                        logger.error(f"Error in Flock model thread: {e}")
                        function_calls = []
                        
            except Exception as e:
                logger.error(f"Error with Flock model for {restaurant_name}: {e}", exc_info=True)
                function_calls = []
        
        # Use OpenAI GPT-4o-mini by default or as fallback
        if not function_calls:
            logger.info(f"Using OpenAI GPT-4o-mini for Web3 information for restaurant: {restaurant_name}")
            
            # Convert tools to OpenAI format
            openai_tools = []
            for tool in web3_tools:
                if "type" in tool and tool["type"] == "function" and "function" in tool:
                    openai_tools.append({
                        "type": "function",
                        "function": tool["function"]
                    })
            
            messages = [
                {"role": "system", "content": "You are a Web3 restaurant information assistant. Your task is to analyze restaurant data and determine if they likely have Web3 features. Make reasonable inferences based on the restaurant's name, type, and location."},
                {"role": "user", "content": restaurant_context}
            ]
            
            # Set a timeout for OpenAI API call
            import threading
            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            
            def call_openai_with_timeout():
                calls = []
                for tool_idx in range(3):  # Try up to 3 tools
                    try:
                        result = call_openai_for_action(messages, openai_tools)
                        if result:
                            calls.append(result)
                    except Exception as e:
                        logger.error(f"Error calling OpenAI API (tool {tool_idx}): {e}")
                return calls
            
            # Execute with timeout
            with ThreadPoolExecutor() as executor:
                future = executor.submit(call_openai_with_timeout)
                try:
                    function_calls = future.result(timeout=timeout)
                    logger.info(f"Successfully generated Web3 information with OpenAI for {restaurant_name}")
                except TimeoutError:
                    logger.warning(f"OpenAI API call timed out after {timeout}s for {restaurant_name}")
                    function_calls = []
                except Exception as e:
                    logger.error(f"Error in OpenAI thread: {e}")
                    function_calls = []
        
        # Process function calls (either from Flock or OpenAI)
        if isinstance(function_calls, list):
            for call in function_calls:
                if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
                    continue
                    
                func_name = call["name"]
                args = call["arguments"]
                
                # Make sure we're updating the right restaurant
                if args.get("restaurant_id") != restaurant.get("id") and "restaurant_id" in args:
                    args["restaurant_id"] = restaurant.get("id", "unknown")
                
                # Find the correct restaurant in our enriched list by ID
                for enriched in enriched_restaurants:
                    if enriched.get("id") == restaurant.get("id"):    
                        # Update appropriate fields based on function call
                        if func_name == "add_crypto_payment_info":
                            enriched["accepts_crypto"] = args.get("accepts_crypto", False)
                            enriched["crypto_payment_details"] = args.get("crypto_payment_details", "")
                        
                        elif func_name == "add_token_program_info":
                            enriched["has_token_program"] = args.get("has_token_program", False)
                            enriched["token_program_details"] = args.get("token_program_details", "")
                        
                        elif func_name == "add_nft_rewards_info":
                            enriched["has_nft_rewards"] = args.get("has_nft_rewards", False)
                            enriched["nft_rewards_details"] = args.get("nft_rewards_details", "")
                        
                        break
    
    logger.info(f"Added Web3 information to {len(limited_restaurants)} restaurants (out of {len(restaurants)})")
    return enriched_restaurants

def format_web3_section(restaurant: dict) -> str:
    """
    Formats Web3-related information for a restaurant into a presentable Markdown string.
    
    Args:
        restaurant: A dictionary containing restaurant information with Web3-related fields.
    
    Returns:
        A formatted Markdown string with Web3 information, or an empty string if no Web3 features available.
    """
    # Check if any Web3 features are available
    has_web3_features = (
        restaurant.get("accepts_crypto", False) or
        restaurant.get("has_token_program", False) or
        restaurant.get("has_nft_rewards", False)
    )
    
    if not has_web3_features:
        return ""
    
    sections = ["### Web3 Features"]
    
    if restaurant.get("accepts_crypto", False):
        crypto_details = restaurant.get("crypto_payment_details", "").strip()
        sections.append(f"**💰 Accepts Crypto**: Yes{f' - {crypto_details}' if crypto_details else ''}")
    
    if restaurant.get("has_token_program", False):
        token_details = restaurant.get("token_program_details", "").strip()
        sections.append(f"**🪙 Token Program**: Yes{f' - {token_details}' if token_details else ''}")
    
    if restaurant.get("has_nft_rewards", False):
        nft_details = restaurant.get("nft_rewards_details", "").strip()
        sections.append(f"**🖼️ NFT Rewards**: Yes{f' - {nft_details}' if nft_details else ''}")
    
    return "\n".join(sections)

if __name__ == '__main__':
    # Example usage for testing this utility directly
    print("Testing Flock agent utility...")

    # Testing decision-making
    example_tools = [
        {"type": "function", "function": {"name": "analyze_restaurant_reviews", "description": "Analyze detailed user reviews for a list of restaurants to get scores.", "parameters": {"type": "object", "properties": {"restaurant_ids": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "function", "function": {"name": "ask_user_clarification", "description": "Ask the user a question.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "present_basic_list", "description": "Present a simple list of restaurants found.", "parameters": {"type": "object", "properties": {"restaurant_names": {"type": "array", "items": {"type": "string"}}}}}}
    ]

    example_context = "User wants cheap Italian food in Soho. Found restaurants 'Pizza Express', 'Vapiano'. Reviews are available for both. Decide next step."

    print(f"Context:\n{example_context}")
    print(f"\nTools:\n{json.dumps(example_tools, indent=2)}")

    chosen_action = decide_next_action_with_flock(example_context, example_tools)

    print(f"\nChosen Action:\n{chosen_action}")

    if chosen_action:
        print(f"\nSuccessfully decided action: {chosen_action.get('name')}")
    else:
        print("\nFailed to get a decision from Flock model.")
    
    # Testing Web3 information
    example_restaurants = [
        {
            "id": "resto123",
            "name": "Future Bites", 
            "formatted_address": "123 Tech Drive, San Francisco",
            "types": ["restaurant", "food", "tech", "innovative"]
        },
        {
            "id": "resto456",
            "name": "Traditional Pasta",
            "vicinity": "456 Old Town St, Rome",
            "types": ["restaurant", "italian", "traditional"]
        }
    ]
    
    print("\nTesting Web3 information retrieval...")
    enriched_restaurants = get_restaurant_web3_info(example_restaurants)
    
    for resto in enriched_restaurants:
        print(f"\nRestaurant: {resto['name']}")
        web3_section = format_web3_section(resto)
        if web3_section:
            print(web3_section)
        else:
            print("No Web3 features available") 