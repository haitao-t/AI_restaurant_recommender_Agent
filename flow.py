import logging
from pocketflow import Flow
from nodes import (
    ParseUserQueryNode, FindRestaurantsNode, FetchReviewsNode,
    DecideActionNode,
    AnalyzeRestaurantReviewsNode, CalculateFitScoreNode, GenerateResponseNode,
    NoCandidatesFoundNode, ReservationNode,
    # Add placeholder nodes for other actions if needed
    # AskClarificationNode, PresentBasicListNode
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_recommendation_flow():
    """Creates and connects the nodes for the recommendation flow."""
    logger.info("Creating recommendation flow...")

    # 1. Instantiate nodes
    parse_query = ParseUserQueryNode()
    find_restaurants = FindRestaurantsNode()
    fetch_reviews = FetchReviewsNode()
    decide_action = DecideActionNode()
    analyze_reviews = AnalyzeRestaurantReviewsNode() # Retries configured in Node class
    calculate_fit = CalculateFitScoreNode()
    generate_response = GenerateResponseNode()
    no_candidates_node = NoCandidatesFoundNode() # Node for handling no results
    make_reservation = ReservationNode() # Node for handling reservations

    # 2. Define flow connections (happy path)
    parse_query >> find_restaurants
    find_restaurants >> fetch_reviews
    fetch_reviews >> decide_action

    # 3. Define branches based on Flock agent's decision
    decide_action - "analyze" >> analyze_reviews
    analyze_reviews >> calculate_fit
    calculate_fit >> generate_response

    # 4. Define reservation path
    generate_response - "make_reservation" >> make_reservation

    # Placeholder for clarification path (needs AskClarificationNode implementation)
    # decide_action - "clarify" >> AskClarificationNode()

    # Placeholder for basic list path (needs PresentBasicListNode implementation)
    # decide_action - "list_only" >> PresentBasicListNode()

    # 5. Define alternative paths (error handling)
    find_restaurants - "no_candidates_found" >> no_candidates_node

    # 6. Create the Flow object, specifying the start node
    recommendation_flow = Flow(start=parse_query)
    logger.info("Recommendation flow created successfully.")
    return recommendation_flow

if __name__ == '__main__':
    # Example of how to create the flow (for testing purposes)
    logger.info("Creating recommendation flow for testing...")
    flow = create_recommendation_flow()
    logger.info(f"Flow created successfully with start node: {flow.start.__class__.__name__}")
    # You would typically run the flow in main.py, not here
    logger.info("Flow verification complete.") 