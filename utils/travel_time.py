"""
Travel time calculation utilities for restaurant recommendations.
Uses Google Maps API to calculate travel time between locations.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

def calculate_travel_time(origin: str, destination: str, mode: str = 'transit') -> Optional[Dict[str, Any]]:
    """
    Calculate travel time between two locations using Google Maps API.
    
    Args:
        origin: Starting location (address or coordinates)
        destination: Destination location (address or coordinates)
        mode: Travel mode (driving, walking, bicycling, transit)
    
    Returns:
        Dictionary containing travel information or None if error
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("Google Maps API key not found in environment variables")
        return None
    
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": mode,
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            logger.error(f"Google Maps API request failed: {response.status_code}")
            return None
        
        data = response.json()
        if data["status"] != "OK":
            logger.warning(f"Google Maps API returned non-OK status: {data['status']}")
            return None
        
        # Extract relevant information
        route = data["routes"][0]
        leg = route["legs"][0]
        
        result = {
            "duration": {
                "text": leg["duration"]["text"],
                "value": leg["duration"]["value"]  # in seconds
            },
            "distance": {
                "text": leg["distance"]["text"],
                "value": leg["distance"]["value"]  # in meters
            },
            "start_address": leg["start_address"],
            "end_address": leg["end_address"],
            "mode": mode
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating travel time: {str(e)}")
        return None

def format_travel_time(travel_info: Dict[str, Any]) -> str:
    """
    Format travel time information into a human-readable string.
    
    Args:
        travel_info: Dictionary returned by calculate_travel_time
        
    Returns:
        Formatted string describing travel time and distance
    """
    if not travel_info:
        return "Travel time information not available"
    
    mode_name = {
        "driving": "driving",
        "walking": "walking",
        "bicycling": "cycling",
        "transit": "public transport"
    }.get(travel_info["mode"], travel_info["mode"])
    
    return f"{travel_info['duration']['text']} by {mode_name} ({travel_info['distance']['text']})"


if __name__ == "__main__":
    # Basic test
    logging.basicConfig(level=logging.INFO)
    origin = "Covent Garden, London"
    destination = "Soho, London"
    
    print(f"Calculating travel time from {origin} to {destination}...")
    travel_info = calculate_travel_time(origin, destination)
    
    if travel_info:
        print(format_travel_time(travel_info))
    else:
        print("Failed to calculate travel time") 