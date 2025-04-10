import logging
from pocketflow import Flow
from conversation_nodes import (
    ConversationManagerNode,
    ConversationContinuationNode,
    ConversationCompletionNode,
    DebugNode
)
from flow import create_recommendation_flow

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_conversation_flow():
    """Creates and connects the nodes for the conversation flow."""
    logger.info("Creating conversation flow...")

    # 1. Instantiate nodes
    conversation_manager = ConversationManagerNode()
    continue_conversation = ConversationContinuationNode()
    conversation_completion = ConversationCompletionNode()
    debug_node = DebugNode()
    
    # Get the recommendation flow
    recommendation_flow = create_recommendation_flow()

    # 2. Define flow connections
    # Main conversation loop
    conversation_manager - "continue_conversation" >> continue_conversation
    continue_conversation - "continue_dialog" >> conversation_manager
    
    # Path to recommendation generation
    conversation_manager - "recommend" >> recommendation_flow
    
    # Path after recommendation completion
    recommendation_flow >> conversation_completion
    conversation_completion - "continue_dialog" >> conversation_manager
    
    # Error handling path
    conversation_manager - "error" >> debug_node
    continue_conversation - "error" >> debug_node
    recommendation_flow - "error" >> debug_node
    conversation_completion - "error" >> debug_node
    debug_node - "continue_dialog" >> conversation_manager

    # 3. Create the Flow object, specifying the start node
    conversation_flow = Flow(start=conversation_manager)
    logger.info("Conversation flow created successfully.")
    return conversation_flow 