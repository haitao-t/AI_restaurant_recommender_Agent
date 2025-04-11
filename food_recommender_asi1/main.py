# Entry point for the food recommendation agent

import logging
import sys
import os
from dotenv import load_dotenv # Import dotenv

# Load environment variables from .env file FIRST
# This makes them available for other modules that might import os
load_dotenv()

# Check for essential environment variables early
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not GOOGLE_MAPS_API_KEY:
     missing_vars.append("GOOGLE_MAPS_API_KEY")

if missing_vars:
    print(f"Error: Required environment variable(s) not set: {', '.join(missing_vars)}", file=sys.stderr)
    print("Please create a .env file based on .env.example and fill in the values.", file=sys.stderr)
    sys.exit(1)

# Now that env vars are loaded and checked, import the flows
from conversation_flow import create_conversation_flow

# Configure basic logging AFTER potentially loading logging config from env vars (if any)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Define the logger instance

def run_conversational_recommendation():
    """
    Runs the conversational recommendation flow in an interactive loop.
    """
    logger.info("Starting conversational recommendation flow")

    # 1. Initialize the shared store
    # For conversation flow, we initially only set an empty conversation state
    shared = {
        "user_input": "",  # Start with empty input to trigger welcome message
        "conversation_state": {},  # Will be initialized by ConversationManagerNode
        "system_response": None,
        "error_info": None,  # For error tracking
        # The following will be populated during recommendation generation
        "user_query": None,
        "parsed_query": None,
        "candidate_restaurants": None,
        "reviews_data": None,
        "dimensional_scores_1_10": None,
        "ranked_recommendations": None,
        "final_response": None
    }

    # 2. Create the conversation flow instance
    conversation_flow = create_conversation_flow()

    # 3. Run the flow in a loop
    try:
        while True:
            try:
                # The conversation flow continues until an explicit exit
                conversation_flow.run(shared)
                logger.info("Conversation session completed.")
                # If we get here, the flow ran to completion, ask if user wants to continue
                print("\nWould you like to make another inquiry? (yes/no)")
                continue_response = input("> ").strip().lower()
                if continue_response not in ["yes", "y", "sure", "ok", "okay"]:
                    print("\nThank you for using our service. Goodbye!")
                    break
                
                # Reset certain fields for a new conversation
                shared["user_input"] = ""
                shared["parsed_query"] = None
                shared["candidate_restaurants"] = None
                shared["reviews_data"] = None
                shared["dimensional_scores_1_10"] = None
                shared["ranked_recommendations"] = None
                shared["final_response"] = None
                
                # Don't reset conversation_state completely, just update the stage
                if shared.get("conversation_state"):
                    shared["conversation_state"]["current_stage"] = "information_gathering"
                    
            except Exception as e:
                logger.error(f"An error occurred during the conversation: {e}", exc_info=True)
                # Add error information to shared store
                shared["error_info"] = str(e)
                # Continue with the error flow path
                shared["user_input"] = ""  # Clear user input to avoid loops
                # Here we would normally trigger the error action, but in this implementation,
                # we rely on the DebugNode to handle it in the next iteration
                
    except KeyboardInterrupt:
        print("\nConversation ended by user.")
    except Exception as e:
        logger.error(f"Critical error in conversation loop: {e}", exc_info=True)
        print("\nSorry, I encountered a critical error. Please restart the application.")


def main():
    """Main function to run the conversational agent."""
    print("Conversational Food Recommendation Agent")
    print("---------------------------------------")
    print("(Press Ctrl+C to exit at any time)")
    print()

    # Run the conversational recommendation process
    run_conversational_recommendation()


if __name__ == "__main__":
    # Environment variable checks are now done at the top level after loading .env
    main() 