import logging
import os
import json
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- ACI.dev Configuration ---
ACI_DEV_API_KEY = os.environ.get("AIPOLABS_ACI_API_KEY", os.environ.get("ACI_DEV_API_KEY"))
USE_ACI_DEV = os.environ.get("USE_ACI_DEV", "true").lower() == "true"
LINKED_ACCOUNT_OWNER_ID = os.environ.get("LINKED_ACCOUNT_OWNER_ID", "default_user")

# Check if ACI.dev should be initialized
if not USE_ACI_DEV:
    logger.info("ACI.dev integration is disabled by configuration")
elif not ACI_DEV_API_KEY:
    logger.warning("ACI.dev API key not found. ACI.dev features will be unavailable.")
else:
    try:
        # Import ACI SDK
        from aipolabs import ACI
        aci_client = ACI(api_key=ACI_DEV_API_KEY)
        logger.info("ACI.dev client initialized successfully")
    except ImportError:
        logger.error("Failed to import ACI.dev SDK. Please install it with 'pip install aipolabs'")
        aci_client = None
    except Exception as e:
        logger.error(f"Failed to initialize ACI.dev client: {e}")
        aci_client = None

def get_additional_restaurant_info(restaurant_name: str, location: str) -> Dict[str, Any]:
    """
    Retrieve additional information about a restaurant using ACI.dev services.
    
    Args:
        restaurant_name: The name of the restaurant.
        location: The location of the restaurant.
        
    Returns:
        A dictionary containing the additional information.
    """
    if not USE_ACI_DEV or not ACI_DEV_API_KEY or 'aci_client' not in globals() or aci_client is None:
        logger.warning("ACI.dev client not available. Skipping additional info retrieval.")
        return {}
    
    try:
        # Prepare the search query
        search_query = f"{restaurant_name} restaurant {location} reviews"
        
        # Call TAVILY search function
        result = aci_client.functions.execute(
            "TAVILY__SEARCH",
            body={"query": search_query},
            linked_account_owner_id=LINKED_ACCOUNT_OWNER_ID
        )
        
        if not result.success:
            logger.warning(f"ACI.dev search failed for {restaurant_name}: {result.error}")
            return {}
        
        # Extract the search results
        search_results = result.data.get("results", [])
        
        # Process the search results to extract useful information
        additional_info = {
            "search_results": search_results,
            "aci_summary": _generate_summary_from_results(search_results, restaurant_name),
            "aci_timestamp": time.time()
        }
        
        logger.info(f"Retrieved additional information for {restaurant_name} using ACI.dev")
        return additional_info
    
    except Exception as e:
        logger.error(f"Error retrieving additional info for {restaurant_name}: {e}")
        return {}

def _generate_summary_from_results(search_results: List[Dict[str, Any]], restaurant_name: str) -> str:
    """
    Generate a concise summary from the search results.
    
    Args:
        search_results: List of search result dictionaries.
        restaurant_name: Name of the restaurant for context.
        
    Returns:
        A string containing the summary.
    """
    if not search_results:
        return "No additional information available."
    
    # Extract snippets from the search results
    snippets = [result.get("snippet", "") for result in search_results[:3] if result.get("snippet")]
    
    if not snippets:
        return "No clear information found in search results."
    
    # Concatenate the snippets
    combined_text = " ".join(snippets)
    
    # For now, just return the combined text (in a real implementation, you might 
    # use an LLM to generate a more coherent summary)
    if len(combined_text) > 500:
        return combined_text[:497] + "..."
    return combined_text

def search_web_for_restaurant_web3(restaurant_name: str) -> Dict[str, Any]:
    """
    Search for restaurants with Web3 features using TAVILY.
    
    Args:
        restaurant_name: The name of the restaurant to search for.
        
    Returns:
        A dictionary containing the search results and processed information.
    """
    if not USE_ACI_DEV or not ACI_DEV_API_KEY or 'aci_client' not in globals() or aci_client is None:
        logger.warning("ACI.dev client not available. Skipping Web3 info search.")
        return {"web3_features": [], "search_results": []}
    
    try:
        # Prepare the search query
        search_query = f"{restaurant_name} cryptocurrency payment NFT loyalty tokens blockchain"
        
        # Call TAVILY search function
        result = aci_client.functions.execute(
            function_name="TAVILY__SEARCH",
            body={"query": search_query},
            linked_account_owner_id=LINKED_ACCOUNT_OWNER_ID
        )
        
        if not result.success:
            logger.warning(f"ACI.dev Web3 search failed for {restaurant_name}: {result.error}")
            return {"web3_features": [], "search_results": []}
        
        # Extract and return the search results
        search_results = result.data.get("results", [])
        
        # Process the results to identify potential Web3 features
        web3_features = _extract_web3_features(search_results)
        
        return {
            "web3_features": web3_features,
            "search_results": search_results
        }
    
    except Exception as e:
        logger.error(f"Error searching for Web3 information for {restaurant_name}: {e}")
        return {"web3_features": [], "search_results": []}

def _extract_web3_features(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract Web3-related features from search results.
    
    Args:
        search_results: List of search result dictionaries.
        
    Returns:
        A list of dictionaries containing identified Web3 features.
    """
    features = []
    
    # Web3 keywords to look for
    web3_keywords = {
        "cryptocurrency": "cryptocurrency_payment",
        "crypto payment": "cryptocurrency_payment",
        "bitcoin": "cryptocurrency_payment",
        "ethereum": "cryptocurrency_payment",
        "token": "loyalty_token",
        "loyalty program": "loyalty_token",
        "nft": "nft_reward",
        "collectible": "nft_reward",
        "blockchain": "blockchain_feature",
        "web3": "web3_integration"
    }
    
    # Check each result for Web3 keywords
    for result in search_results:
        snippet = result.get("snippet", "").lower()
        title = result.get("title", "").lower()
        content = snippet + " " + title
        
        # Check for each keyword
        for keyword, feature_type in web3_keywords.items():
            if keyword in content:
                # Extract a relevant quote if possible
                start_idx = content.find(keyword)
                if start_idx != -1:
                    # Get some context around the keyword (up to 100 chars)
                    start = max(0, start_idx - 50)
                    end = min(len(content), start_idx + len(keyword) + 50)
                    quote = content[start:end]
                    
                    # Add to features if not already present
                    feature = {
                        "type": feature_type,
                        "source": result.get("title", "Unknown source"),
                        "quote": f"...{quote}..." if start > 0 or end < len(content) else quote
                    }
                    
                    # Check if this feature type is already in the list
                    if not any(f["type"] == feature_type for f in features):
                        features.append(feature)
    
    return features 