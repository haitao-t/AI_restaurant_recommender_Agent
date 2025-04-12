import logging
import time
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import the official Aipolabs SDK
from aipolabs import ACI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Aipolabs ACI Configuration ---
AIPOLABS_ACI_API_KEY = os.environ.get("AIPOLABS_ACI_API_KEY")
# For backward compatibility, also check the old env variable name
if not AIPOLABS_ACI_API_KEY:
    AIPOLABS_ACI_API_KEY = os.environ.get("ACI_DEV_API_KEY")
    if AIPOLABS_ACI_API_KEY:
        logger.warning("Using ACI_DEV_API_KEY is deprecated. Please update to AIPOLABS_ACI_API_KEY in your .env file.")

USE_ACI_DEV = os.environ.get("USE_ACI_DEV", "true").lower() == "true"
LINKED_ACCOUNT_OWNER_ID = os.environ.get("LINKED_ACCOUNT_OWNER_ID", "default_user")

# Initialize ACI client
aci_client = None
if AIPOLABS_ACI_API_KEY and USE_ACI_DEV:
    try:
        aci_client = ACI(api_key=AIPOLABS_ACI_API_KEY)
        logger.info("Successfully initialized Aipolabs ACI client")
    except Exception as e:
        logger.error(f"Failed to initialize Aipolabs ACI client: {e}")

# Available services
SUPPORTED_SERVICES = ["COINMARKETCAP", "FIRECRAWL", "TAVILY"]

def get_aci_agent_info(restaurants: List[Dict[str, Any]], 
                       max_retries: int = 3, 
                       delay: int = 2) -> List[Dict[str, Any]]:
    """
    Enriches restaurant data with additional information from Aipolabs ACI services.
    
    Args:
        restaurants: A list of dictionaries containing restaurant information.
                    Each dictionary should have at least 'id' and 'name' keys.
        max_retries: Maximum number of retry attempts.
        delay: Seconds to wait between retries.
    
    Returns:
        A list of dictionaries with the original restaurant information plus ACI-related fields
    """
    if not USE_ACI_DEV:
        logger.info("Aipolabs ACI integration is disabled. Skipping restaurant enrichment.")
        return restaurants
        
    if not aci_client:
        logger.warning("Aipolabs ACI client not initialized. Cannot enrich restaurant data.")
        return restaurants
    
    # If no restaurants to process, return empty list
    if not restaurants:
        return []
    
    enriched_restaurants = []
    
    # Process each restaurant to add ACI information
    for restaurant in restaurants:
        # Create a copy of the restaurant data to enrich
        enriched_restaurant = restaurant.copy()
        restaurant_name = restaurant.get('name', 'Unknown Restaurant')
        restaurant_id = restaurant.get('id', 'unknown')
        
        logger.info(f"Enriching restaurant with Aipolabs ACI services: {restaurant_name} (ID: {restaurant_id})")
        
        # Default ACI information (in case API fails)
        aci_default = {
            "aci_data": {},
            "aci_search_results": [],
            "aci_market_data": {}
        }
        enriched_restaurant.update(aci_default)
        
        # Use TAVILY for web search information about the restaurant
        try:
            search_query = f"{restaurant_name} restaurant reviews"
            search_result = search_with_tavily(search_query)
            if search_result:
                enriched_restaurant["aci_search_results"] = search_result
                logger.info(f"Added TAVILY search results for '{restaurant_name}'")
        except Exception as e:
            logger.error(f"Error using TAVILY for '{restaurant_name}': {e}")
        
        # Add to result list
        enriched_restaurants.append(enriched_restaurant)
    
    return enriched_restaurants

def search_with_tavily(query: str, max_retries: int = 2) -> List[Dict[str, Any]]:
    """Search the web using TAVILY service."""
    if not aci_client:
        return []
    
    for attempt in range(max_retries):
        try:
            # Get the function definition
            function_definition = aci_client.functions.get_definition("TAVILY__SEARCH")
            
            # Execute the search function with proper body parameter
            result = aci_client.functions.execute(
                "TAVILY__SEARCH",
                body={"query": query},  # 使用body参数包装查询
                linked_account_owner_id=LINKED_ACCOUNT_OWNER_ID
            )
            
            if result.success:
                return result.data.get("results", [])
            else:
                logger.warning(f"TAVILY search failed: {result.error}")
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error executing TAVILY search: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    return []

def format_aci_section(restaurant: Dict[str, Any]) -> str:
    """
    Formats the Aipolabs ACI data into a readable section for the restaurant recommendation.
    
    Args:
        restaurant: The restaurant dictionary with ACI data
        
    Returns:
        A formatted string with the ACI section, or empty string if no data
    """
    # Check if we have any ACI data to display
    search_results = restaurant.get('aci_search_results', [])
    
    if not search_results:
        return ""
    
    output = []
    output.append("✨ Aipolabs ACI - Additional Information ✨")
    
    # Add search results if available
    if search_results:
        output.append("\nWeb Search Results:")
        for i, result in enumerate(search_results[:2]):  # Limit to 2 results
            title = result.get("title", "Untitled")
            snippet = result.get("snippet", "No preview available")
            output.append(f"• {title}")
            output.append(f"  {snippet}")
    
    return "\n".join(output) 