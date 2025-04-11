"""Conversation nodes for the restaurant recommendation agent."""
import json
import logging
import datetime
import re

from pocketflow import Node
from utils.call_llm import call_llm

logger = logging.getLogger(__name__)


class WelcomeNode(Node):
    """Provides a welcome message to start the conversation."""
    
    def prep(self, shared):
        # Nothing to prepare, just return None
        return None
    
    def exec(self, prep_res):
        # Create a friendly welcome message
        welcome_message = (
            "👋 Hello! I'm your restaurant recommendation assistant. "
            "I can help you find the perfect restaurant based on your "
            "preferences. "
            "\n\nPlease tell me what kind of cuisine you're looking for, "
            "your preferred location, price range, and any dietary "
            "restrictions. "
            "\n\nFor example, you could say: "
            "'I'm looking for an Italian restaurant in Soho, mid-range price, "
            "vegetarian options.'"
        )
        return welcome_message
    
    def post(self, shared, prep_res, exec_res):
        # Store the welcome message as the first system response
        shared["system_response"] = exec_res
        
        # Print the welcome message
        print(f"\nAssistant: {exec_res}")
        
        # Get the first user input
        user_input = input("\nYou: ")
        shared["user_input"] = user_input
        
        # Proceed to the main conversation flow
        return "start_conversation"


class ConversationManagerNode(Node):
    """Node for managing the conversation flow."""

    def __init__(self, max_retries=3):
        """Initialize the conversation manager with retry capability."""
        super().__init__(max_retries=max_retries)
        self.missing_info = []
    
    def prep(self, shared):
        # Prepare the conversation context, starting fresh if this is new
        conversation_history = shared.get("conversation_history", [])
        current_state = shared.get("conversation_state", {})
        user_input = shared.get("user_input", "")
        
        # Build the messages list for the LLM
        system_message = (
            "You are a helpful restaurant recommendation assistant. "
            "Your goal is to recommend restaurants based on user preferences. "
            "You need to gather information about cuisine type, location, "
            "price range, and any dietary restrictions."
        )
        
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add previous conversation history if it exists
        for msg in conversation_history:
            messages.append(msg)
        
        # Add the current user message if available
        if user_input:
            messages.append({"role": "user", "content": user_input})
            
        return {
            "messages": messages,
            "current_state": current_state,
            "user_input": user_input
        }
        
    def exec(self, prep_result):
        messages = prep_result["messages"]
        current_state = prep_result["current_state"]
        user_input = prep_result.get("user_input", "")
        
        # Check what information we still need from the user
        self.missing_info = []
        required_fields = [
            "cuisine", "location", "price_range", 
            "dietary_restrictions"
        ]
        
        for field in required_fields:
            if field not in current_state or not current_state[field]:
                self.missing_info.append(field)
        
        # Add instructions for the AI based on what info we still need
        if self.missing_info:
            missing_fields = ", ".join(self.missing_info)
            system_content = (
                f"Based on the conversation, extract the following "
                f"information from the user: {missing_fields}. Ask for any "
                f"missing details in a friendly, conversational way."
            )
            # Add as a separate instruction message
            messages.append({
                "role": "system", 
                "content": system_content
            })
        
        # Call the LLM with our constructed messages
        try:
            # Format messages for the LLM call
            formatted_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted_prompt += f"System: {content}\n\n"
                elif role == "user":
                    formatted_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    formatted_prompt += f"Assistant: {content}\n\n"
            
            # Now call the LLM with the formatted prompt
            response = call_llm(formatted_prompt)
            
            # Try to extract information from the user's latest message
            if user_input:
                user_text = user_input.lower()
                
                # Extract cuisine
                cuisines = [
                    "chinese", "italian", "indian", "japanese", 
                    "thai", "mexican", "french"
                ]
                for cuisine in cuisines:
                    if cuisine in user_text:
                        current_state["cuisine"] = cuisine.capitalize()
                        logger.info(f"Extracted cuisine: {cuisine}")
                        break
                
                # Extract location
                if "soho" in user_text:
                    current_state["location"] = "Soho"
                    logger.info("Extracted location: Soho")
                elif "london" in user_text:
                    current_state["location"] = "London"
                    logger.info("Extracted location: London")
                
                # Extract price range
                if "mid-range" in user_text or "medium" in user_text:
                    current_state["price_range"] = "Mid-range"
                    logger.info("Extracted price range: Mid-range")
                elif "cheap" in user_text or "budget" in user_text:
                    current_state["price_range"] = "Budget"
                    logger.info("Extracted price range: Budget")
                elif "expensive" in user_text or "high-end" in user_text:
                    current_state["price_range"] = "Expensive"
                    logger.info("Extracted price range: Expensive")
                
                # Extract price per person
                price_match = re.search(
                    r'(\d+)\s*(?:pound|pounds|gbp|£)', 
                    user_text
                )
                if price_match:
                    price = price_match.group(1)
                    if not current_state.get("price_range"):
                        if int(price) <= 15:
                            current_state["price_range"] = "Budget"
                        elif int(price) <= 30:
                            current_state["price_range"] = "Mid-range"
                        else:
                            current_state["price_range"] = "Expensive"
                    logger.info(f"Extracted price: £{price}")
                
                # Extract dietary restrictions
                dietary_keywords = {
                    "vegetarian": "Vegetarian",
                    "vegan": "Vegan",
                    "gluten-free": "Gluten-free",
                    "halal": "Halal",
                    "kosher": "Kosher",
                    "dairy-free": "Dairy-free",
                    "nut-free": "Nut-free"
                }
                
                dietary_restrictions = []
                for keyword, label in dietary_keywords.items():
                    if keyword in user_text:
                        dietary_restrictions.append(label)
                
                if dietary_restrictions:
                    current_state["dietary_restrictions"] = ", ".join(
                        dietary_restrictions
                    )
                    logger.info(
                        f"Extracted dietary restrictions: "
                        f"{dietary_restrictions}"
                    )
                
                # Update the current state in the prep_result
                prep_result["current_state"] = current_state
            
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise
        
    def post(self, shared, prep_res, exec_res):
        # Update conversation history
        user_input = shared.get("user_input", "")
        
        # If we've extracted any information, update the shared state
        if "current_state" in prep_res:
            shared["conversation_state"] = prep_res["current_state"]
            logger.info(
                f"Updated conversation state: {shared['conversation_state']}"
            )
        
        if user_input:
            # Add user message to history
            if "conversation_history" not in shared:
                shared["conversation_history"] = []
            
            # Add user message
            user_msg = {"role": "user", "content": user_input}
            shared["conversation_history"].append(user_msg)
            
            # Add assistant response to history
            asst_msg = {"role": "assistant", "content": exec_res}
            shared["conversation_history"].append(asst_msg)
            
        # Store the system response
        shared["system_response"] = exec_res
        
        # Clear the user input for next iteration
        shared["user_input"] = ""
        
        # For demonstration purposes, after 2 exchanges, we'll assume we have enough info
        # and proceed to recommendations
        if len(shared.get("conversation_history", [])) >= 4:  
            # At least 2 exchanges (4 messages)
            logger.info(
                "Sufficient conversation exchanges, proceeding to "
                "recommendations"
            )
            
            # Extract basic query from conversation
            conversation_text = ""
            for msg in shared.get("conversation_history", []):
                if msg["role"] == "user":
                    conversation_text += msg["content"] + " "
            
            # Build a query from conversation text
            query_text = (
                f"Find restaurants matching this conversation: "
                f"{conversation_text}"
            )
            logger.info(
                f"Generated query for recommendation engine: "
                f"{query_text}"
            )
            
            # Store query for the recommendation flow
            shared["user_query"] = query_text
            
            # Call the recommendation flow
            return "ready_for_recommendations"
        else:
            # Continue conversation until we have enough exchanges
            return "continue_conversation"
            
    def exec_fallback(self, prep_res, exc):
        # Handle errors gracefully
        error_message = (
            "I'm having trouble processing your request. Let's try again."
        )
        logger.error(
            "Error in conversation manager: {}".format(str(exc))
        )
        return error_message


class ConversationContinuationNode(Node):
    """Handles continuing the conversation when not ready for recommendations."""
    
    def prep(self, shared):
        return shared.get("system_response")
        
    def exec(self, system_response):
        # Simply pass through the system response
        return system_response
        
    def post(self, shared, prep_res, exec_res):
        # Check if there's an error in the shared store
        if "error_info" in shared and shared["error_info"]:
            error_msg = shared["error_info"]
            err_fmt = "Error detected in ConversationContinuationNode: {}"
            logger.error(err_fmt.format(error_msg))
            return "error"
            
        # Print the response and prompt for next input
        print("\nAssistant: {}".format(exec_res))
        user_input = input("\nYou: ")
        
        # Store new user input for next iteration
        shared["user_input"] = user_input
        
        # Continue the conversation loop
        return "continue_dialog"


class ConversationCompletionNode(Node):
    """Handles generating the final recommendation."""
    
    def prep(self, shared):
        # Use the generated recommendation if available
        final_response = shared.get("final_response")
        
        if final_response:
            logger.info("Using final response from recommendation flow")
            return {"final_response": final_response}
        
        # If no final response, prepare for generating one with the LLM
        conversation_history = shared.get("conversation_history", [])
        current_state = shared.get("conversation_state", {})
        
        # Create a message list for the recommendation
        system_message = (
            "Based on the conversation, provide a restaurant "
            "recommendation. Format your response with the restaurant "
            "name, cuisine type, price range, and a brief description."
        )
        
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add the conversation history
        for msg in conversation_history:
            messages.append(msg)
            
        return {
            "messages": messages,
            "current_state": current_state
        }
        
    def exec(self, prep_result):
        # If we already have a final response from the recommendation flow, use it
        if isinstance(prep_result, dict) and "final_response" in prep_result:
            return prep_result["final_response"]
        
        # Otherwise generate a response using LLM
        messages = prep_result["messages"]
        
        # Generate the recommendation
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = "Generating recommendation at {}"
            logger.info(message.format(timestamp))
            
            # Format messages for the LLM call
            formatted_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted_prompt += f"System: {content}\n\n"
                elif role == "user":
                    formatted_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    formatted_prompt += f"Assistant: {content}\n\n"
            
            # Now call the LLM with the formatted prompt
            response = call_llm(formatted_prompt)
            return response
        except Exception as e:
            error_msg = "Error generating recommendation: {}"
            logger.error(error_msg.format(str(e)))
            raise
            
    def post(self, shared, prep_res, exec_res):
        # Store the final recommendation
        shared["recommendation"] = exec_res
        
        # Print the recommendation
        print("\nAssistant: {}".format(exec_res))
        
        # Ask if user wants more recommendations
        more_prompt = "\nWould you like more recommendations? (yes/no): "
        user_input = input(more_prompt)
        
        # Store the user's response
        shared["user_input"] = user_input
        
        # Determine next step based on user response
        if user_input.lower() in ["yes", "y", "sure", "yeah"]:
            return "more_recommendations"
        else:
            return "end_conversation"


class DebugNode(Node):
    """A debugging node to print the current state of the conversation."""
    
    def prep(self, shared):
        return shared
        
    def exec(self, shared_data):
        debug_output = json.dumps(shared_data, indent=2, default=str)
        return debug_output
        
    def post(self, shared, prep_res, exec_res):
        print("\nDEBUG STATE:")
        print(exec_res)
        return "continue" 