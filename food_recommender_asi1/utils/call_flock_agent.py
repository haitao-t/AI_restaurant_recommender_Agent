# Utility for calling the Flock Agent model

import logging
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flock Model Configuration ---
MODEL_NAME = "flock-io/Flock_Web3_Agent_Model"
# Global variables for model and tokenizer to avoid reloading on every call
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_flock_model():
    """Loads the Flock model and tokenizer if they haven't been loaded yet."""
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.info(f"Loading Flock model: {MODEL_NAME} to {device}...")
        try:
            # device_map='auto' might require accelerate
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype="auto", # Use torch.float16 if BF16 not supported or for specific GPUs
                device_map="auto" # Use device map for automatic placement
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            logger.info("Flock model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Flock model: {e}", exc_info=True)
            model = None # Ensure it's None if loading fails
            tokenizer = None

def decide_next_action_with_flock(context: str, tools: list[dict]) -> dict | None:
    """
    Uses the Flock Agent model to decide the next action based on context and tools.

    Args:
        context: A string describing the current state and user request.
        tools: A list of dictionaries describing the available tools (functions).

    Returns:
        A dictionary representing the chosen function call (e.g.,
        {"name": "analyze_restaurant_reviews", "arguments": {"restaurant_ids": [...]}})
        or None if the model fails or doesn't produce a valid function call.
    """
    load_flock_model() # Ensure model is loaded

    if model is None or tokenizer is None:
        logger.error("Flock model/tokenizer not loaded. Cannot decide action.")
        return None

    try:
        tools_json_string = json.dumps(tools, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to serialize tools to JSON: {e}")
        return None

    messages = [
        {"role": "system", "content": "You are a helpful assistant for restaurant recommendations with access to the following functions. Use them if required to fulfill the user's request based on the current context. - " + tools_json_string},
        {"role": "user", "content": context}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        logger.info("Generating next action decision with Flock model...")
        # Adjust max_new_tokens as needed, should be enough for function call JSON
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id # Suppress warning
        )

        # Decode the generated part only
        input_ids_len = model_inputs.input_ids.shape[1]
        generated_part_ids = generated_ids[0, input_ids_len:]
        response_string = tokenizer.decode(generated_part_ids, skip_special_tokens=True)

        logger.info(f"Flock model raw output: {response_string}")

        # --- Parse the response --- 
        # Flock model outputs a JSON *string* representing a *list* of function calls.
        # We usually expect one call in this scenario.
        try:
            # Clean potential markdown code fences
            if response_string.strip().startswith("```json"):
                response_string = response_string.strip()[7:-3].strip()
            elif response_string.strip().startswith("```"):
                 response_string = response_string.strip()[3:-3].strip()
                 
            function_calls = json.loads(response_string)
            if isinstance(function_calls, list) and len(function_calls) > 0:
                # Assuming we only care about the first function call instructed
                chosen_call = function_calls[0]
                if isinstance(chosen_call, dict) and "name" in chosen_call and "arguments" in chosen_call:
                    logger.info(f"Flock decided to call function: {chosen_call['name']} with args: {chosen_call['arguments']}")
                    return chosen_call # Return the parsed function call dictionary
                else:
                    logger.warning(f"Flock output list item is not a valid function call format: {chosen_call}")
                    return None
            else:
                logger.warning(f"Flock output was not a non-empty list: {function_calls}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Flock model output JSON: {e}\nRaw output: {response_string}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error parsing Flock function call: {e}\nRaw output: {response_string}", exc_info=True)
            return None

    except Exception as e:
        logger.error(f"Error during Flock model generation or processing: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example usage for testing this utility directly
    print("Testing Flock agent utility...")

    # Define example tools
    example_tools = [
        {"type": "function", "function": {"name": "analyze_restaurant_reviews", "description": "Analyze detailed user reviews for a list of restaurants to get scores.", "parameters": {"type": "object", "properties": {"restaurant_ids": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "function", "function": {"name": "ask_user_clarification", "description": "Ask the user a question.", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "present_basic_list", "description": "Present a simple list of restaurants found.", "parameters": {"type": "object", "properties": {"restaurant_names": {"type": "array", "items": {"type": "string"}}}}}}
    ]

    # Example context
    example_context = "User wants cheap Italian food in Soho. Found restaurants 'Pizza Express', 'Vapiano'. Reviews are available for both. Decide next step."

    print(f"Context:\n{example_context}")
    print(f"\nTools:\n{json.dumps(example_tools, indent=2)}")

    chosen_action = decide_next_action_with_flock(example_context, example_tools)

    print(f"\nChosen Action:\n{chosen_action}")

    if chosen_action:
        print(f"\nSuccessfully decided action: {chosen_action.get('name')}")
    else:
        print("\nFailed to get a decision from Flock model.") 