import logging
import time
import random # For simulating API calls
import os
import googlemaps # Import the library
from datetime import datetime
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Google Maps Configuration ---
# Load your API key securely from environment variables
# Ensure GOOGLE_MAPS_API_KEY is set in your .env file
GMAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not GMAPS_API_KEY:
    logging.warning("GOOGLE_MAPS_API_KEY not found in environment variables. Will use simulation mode.")
    gmaps = None
else:
    try:
        gmaps = googlemaps.Client(key=GMAPS_API_KEY)
        logging.info("Successfully initialized Google Maps client.")
    except ValueError as ve:
        logging.warning(f"Invalid Google Maps API key: {ve}. Will use simulation mode for restaurant data.")
        gmaps = None
    except Exception as e:
        logging.warning(f"Failed to initialize Google Maps client: {e}. Will use simulation mode.")
        gmaps = None
# --- End Google Maps Configuration ---


def find_restaurants(location: str, cuisine: list[str], **kwargs) -> list[dict]:
    """
    Finds restaurant candidates using a maps API.

    Args:
        location: The area to search in (e.g., "Soho, London").
        cuisine: A list of preferred cuisines (e.g., ["Italian", "Pizza"]).
        **kwargs: Additional parameters (like budget, radius, etc. - adapt as needed).

    Returns:
        A list of dictionaries, each representing a restaurant candidate,
        e.g., [{"id": "place_id_1", "name": "Restaurant A"}, ...].
        Returns an empty list if no results or on error.
    """
    logging.info(f"Searching for restaurants near '{location}' with cuisine(s): {cuisine}")

    if not gmaps:
        logging.warning("Google Maps client not initialized. Using simulation.")
        # --- Placeholder Implementation ---
        time.sleep(random.uniform(0.3, 0.8)) # Simulate API delay
        candidates = [
            {"id": f"gmap_place_id_{i}", "name": f"Simulated {random.choice(cuisine)} Restaurant {i}"}
            for i in range(1, random.randint(2, 6))
        ]
        if not candidates:
             logging.warning(f"Simulated: No restaurants found for '{location}' and {cuisine}.")
        else:
             logging.info(f"Simulated: Found {len(candidates)} candidates.")
        return candidates
        # --- End Placeholder ---

    # --- Real Implementation ---
    # Construct a query suitable for Google Maps Places API
    # Simple text search: combines cuisine and location
    query = f"{' or '.join(cuisine)} restaurants in {location}"
    logging.info(f"Executing Google Maps query: '{query}'")
    try:
        # Consider using nearby_search for more location-centric results if needed
        # places_result = gmaps.places(query=query, type='restaurant') # Use type parameter
        places_result = gmaps.places(query=query)

        candidates = []
        status = places_result.get("status")

        if status == "OK":
            for place in places_result.get("results", []):
                # Filter by type to ensure it's a relevant place
                place_types = place.get("types", [])
                if "restaurant" in place_types or "food" in place_types:
                    candidates.append({
                        "id": place.get("place_id"),
                        "name": place.get("name"),
                        # Optionally add more details readily available from search:
                        # "address": place.get("formatted_address"),
                        # "rating": place.get("rating"),
                        # "user_ratings_total": place.get("user_ratings_total")
                    })
            logging.info(f"Found {len(candidates)} potential candidates via Google Maps for query: '{query}'.")
        elif status == "ZERO_RESULTS":
             logging.warning(f"Google Maps returned ZERO_RESULTS for query: '{query}'")
        else:
             # Log other statuses like OVER_QUERY_LIMIT, REQUEST_DENIED, INVALID_REQUEST
             logging.error(f"Google Maps API error for query '{query}': {status} - {places_result.get('error_message', '')}")

        return candidates

    except googlemaps.exceptions.ApiError as e:
        # Handles specific Google Maps API errors (e.g., invalid key, quota exceeded)
        logging.error(f"Google Maps API Error during search: {e}", exc_info=True)
        return []
    except Exception as e:
        # Catch other potential issues (network errors, etc.)
        logging.error(f"Unexpected error during Google Maps search: {e}", exc_info=True)
        return []
    # --- End Real Implementation ---


def get_reviews_for_restaurant(restaurant_id: str, max_reviews: int = 20) -> list[str]:
    """
    Fetches recent reviews for a specific restaurant ID using Place Details.

    Args:
        restaurant_id: The unique place ID from the maps API.
        max_reviews: The maximum number of reviews to attempt to fetch.

    Returns:
        A list of review text strings, or an empty list if none found or on error.
    """
    logging.info(f"Fetching reviews for restaurant_id: {restaurant_id} (max: {max_reviews})")

    if not gmaps:
        logging.warning("Google Maps client not initialized. Using simulation.")
        # --- Placeholder Implementation ---
        time.sleep(random.uniform(0.2, 0.5))
        num_reviews = random.randint(0, max_reviews)
        if num_reviews == 0:
            logging.warning(f"Simulated: No reviews found for {restaurant_id}.")
            return []
        simulated_reviews = [
            f"Simulated review {i+1} for {restaurant_id}. " + random.choice([
                "It was fantastic!", "Decent experience.", "Could be better.", "Absolutely amazing ambiance.", "Food was cold."
            ]) for i in range(num_reviews)
        ]
        logging.info(f"Simulated: Fetched {len(simulated_reviews)} reviews for {restaurant_id}.")
        return simulated_reviews
        # --- End Placeholder ---

    # --- Real Implementation (using Place Details) ---
    try:
        # Request 'review' field for Place Details
        # Also requesting 'name' for logging context
        # Ensure your API key is enabled for Places API
        details = gmaps.place(place_id=restaurant_id, fields=['name', 'review'])

        reviews_list = []
        status = details.get("status")

        if status == "OK" and "result" in details:
            reviews_data = details["result"].get("reviews", [])
            if reviews_data:
                # Sort by time (most recent first) if timestamp is available
                # Google reviews often have a 'time' field (Unix timestamp)
                try:
                    reviews_data.sort(key=lambda r: r.get('time', 0), reverse=True)
                except Exception:
                    logging.warning(f"Could not sort reviews by time for {restaurant_id}. Proceeding without sorting.")

                for review in reviews_data[:max_reviews]:
                    if review.get("text"):
                        reviews_list.append(review["text"])
                logging.info(f"Fetched {len(reviews_list)} reviews for {details['result'].get('name', restaurant_id)} via Google Maps Place Details.")
            else:
                 logging.warning(f"No reviews found in Place Details for {details['result'].get('name', restaurant_id)} ({restaurant_id}).")
        elif status == "ZERO_RESULTS":
            # Should not typically happen for a valid place_id, but handle defensively
             logging.warning(f"Google Maps Place Details returned ZERO_RESULTS for {restaurant_id}.")
        else:
             logging.error(f"Google Maps Place Details API error for {restaurant_id}: {status} - {details.get('error_message', '')}")

        return reviews_list

    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Google Maps API Error fetching details for {restaurant_id}: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"Unexpected error fetching reviews for {restaurant_id}: {e}", exc_info=True)
        return []
    # --- End Real Implementation ---


def get_user_geolocation(fallback_location="London") -> dict:
    """
    Attempts to get the user's current location using IP geolocation.
    
    Args:
        fallback_location: Default location to use if geolocation fails.
        
    Returns:
        dict containing location information with keys:
        - 'success': boolean indicating if location was successfully determined
        - 'address': formatted address string (e.g., "Soho, London, UK")
        - 'lat': latitude (float)
        - 'lng': longitude (float)
        - 'neighborhood': local neighborhood name if available
    """
    logging.info("===== ATTEMPTING TO GET USER GEOLOCATION AUTOMATICALLY =====")
    
    # Using ipinfo.io's free API for IP-based geolocation (no API key required for basic usage)
    try:
        response = requests.get('https://ipinfo.io/json', timeout=3)
        if response.status_code == 200:
            data = response.json()
            
            # Extract location data
            loc = data.get('loc', '').split(',')
            city = data.get('city', '')
            region = data.get('region', '')
            country = data.get('country', '')
            
            if loc and len(loc) == 2 and city:
                lat, lng = float(loc[0]), float(loc[1])
                
                # Format the address
                address_parts = [part for part in [city, region, country] if part]
                address = ', '.join(address_parts)
                
                # Try to get neighborhood from reverse geocoding if Google Maps client is available
                neighborhood = None
                if gmaps:
                    try:
                        reverse_geocode = gmaps.reverse_geocode((lat, lng))
                        if reverse_geocode:
                            # Try to extract neighborhood
                            for result in reverse_geocode:
                                for component in result.get('address_components', []):
                                    if 'neighborhood' in component.get('types', []):
                                        neighborhood = component.get('long_name')
                                        break
                                if neighborhood:
                                    break
                    except Exception as e:
                        logging.warning(f"Error during reverse geocoding: {e}")
                
                location_info = {
                    'success': True,
                    'address': address,
                    'lat': lat,
                    'lng': lng,
                    'neighborhood': neighborhood or city
                }
                logging.info(f"===== SUCCESSFULLY DETECTED USER LOCATION: {location_info['address']} ({location_info['neighborhood']}) =====")
                return location_info
    
        # If we reach here, geolocation failed in some way
        logging.warning(f"Failed to get user geolocation. Status code: {response.status_code}")
    except Exception as e:
        logging.warning(f"Exception during geolocation: {e}")
    
    # Return fallback data
    fallback_info = {
        'success': False,
        'address': fallback_location,
        'lat': None,
        'lng': None,
        'neighborhood': fallback_location.split(',')[0] if ',' in fallback_location else fallback_location
    }
    logging.warning(f"===== USING FALLBACK LOCATION: {fallback_info['address']} =====")
    return fallback_info


def get_restaurant_details(place_id: str) -> dict:
    """
    Gets detailed information about a restaurant, including opening hours.
    
    Args:
        place_id: Google Place ID for the restaurant.
        
    Returns:
        Dictionary with restaurant details, including:
        - 'name': Restaurant name
        - 'opening_hours': Opening hours information
        - 'formatted_address': Full address
        - 'formatted_phone_number': Phone number 
        - 'website': Website URL
        - 'price_level': Price level (1-4)
        - 'rating': Google rating
        - 'user_ratings_total': Number of ratings
    """
    logging.info(f"Getting detailed information for restaurant: {place_id}")
    
    if not gmaps:
        logging.warning("Google Maps client not initialized. Using simulation.")
        # Placeholder implementation
        time.sleep(random.uniform(0.2, 0.6))
        
        # Generate random opening hours for today
        now = datetime.now()
        day_of_week = now.weekday()  # 0-6 (Monday is 0)
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        today = days[day_of_week]
        
        # Random opening between 7-11 AM, closing between 8-11 PM
        opening_hour = random.randint(7, 11)
        closing_hour = random.randint(20, 23)
        
        # Calculate if restaurant is open and closing soon
        current_hour = now.hour
        is_open = opening_hour <= current_hour < closing_hour
        closing_soon = is_open and (closing_hour - current_hour) <= 1  # Closing within an hour
        
        simulated_details = {
            'name': f"Simulated Restaurant {place_id[-4:]}",
            'formatted_address': f"123 Simulation St, London",
            'formatted_phone_number': f"+44 20 1234 {random.randint(1000, 9999)}",
            'website': "https://example.com/restaurant",
            'price_level': random.randint(1, 4),
            'rating': round(random.uniform(3.0, 4.9), 1),
            'user_ratings_total': random.randint(50, 500),
            'opening_hours': {
                'periods': [
                    {
                        'open': {'day': day_of_week, 'time': f"{opening_hour:02d}00"},
                        'close': {'day': day_of_week, 'time': f"{closing_hour:02d}00"}
                    }
                ],
                'weekday_text': [
                    f"{today}: {opening_hour}:00 AM - {closing_hour-12 if closing_hour > 12 else closing_hour}:00 {'PM' if closing_hour > 12 else 'AM'}"
                ],
                'is_open': is_open,
                'closing_soon': closing_soon
            }
        }
        
        return simulated_details
    
    # Real implementation using Google Maps API
    try:
        fields = [
            'name', 'opening_hours', 'formatted_address', 
            'formatted_phone_number', 'website', 'price_level',
            'rating', 'user_ratings_total'
        ]
        
        details = gmaps.place(place_id=place_id, fields=fields)
        
        if details.get("status") == "OK" and "result" in details:
            result = details["result"]
            
            # Process opening hours to add is_open and closing_soon flags
            if 'opening_hours' in result:
                # Check if the place is currently open
                result['opening_hours']['is_open'] = result['opening_hours'].get('open_now', False)
                
                # Calculate if it's closing soon (within 1 hour)
                closing_soon = False
                
                # Get current time
                now = datetime.now()
                day_of_week = now.weekday()  # 0-6 (Monday is 0)
                current_hour = now.hour
                current_minute = now.minute
                
                # Check if we have periods data to calculate closing time
                if 'periods' in result['opening_hours']:
                    for period in result['opening_hours']['periods']:
                        # Find today's closing time
                        if period.get('close', {}).get('day') == day_of_week:
                            closing_time = period.get('close', {}).get('time', '')
                            if closing_time and len(closing_time) >= 4:
                                closing_hour = int(closing_time[:2])
                                closing_minute = int(closing_time[2:])
                                
                                # Calculate minutes until closing
                                mins_until_closing = (closing_hour - current_hour) * 60 + (closing_minute - current_minute)
                                
                                # Flag as closing soon if less than 60 minutes until closing
                                if 0 < mins_until_closing <= 60:
                                    closing_soon = True
                                    break
                
                result['opening_hours']['closing_soon'] = closing_soon
            
            return result
        else:
            logging.warning(f"Failed to get restaurant details. Status: {details.get('status')}")
            return {}
            
    except Exception as e:
        logging.error(f"Error getting restaurant details: {e}")
        return {}


def calculate_distance_to_restaurant(user_location, restaurant_location):
    """
    Calculate the distance and estimated travel time between user's location and a restaurant.
    
    Args:
        user_location: Dict with 'lat' and 'lng' keys for user's location
        restaurant_location: Dict with 'lat' and 'lng' keys for restaurant's location
        
    Returns:
        Dictionary with:
        - 'distance_km': Distance in kilometers
        - 'distance_miles': Distance in miles 
        - 'travel_time_driving': Estimated travel time by car in minutes
        - 'travel_time_transit': Estimated travel time by public transport in minutes
        - 'travel_time_walking': Estimated travel time by walking in minutes
    """
    logging.info(f"Calculating distance to restaurant")
    
    if not user_location.get('lat') or not user_location.get('lng'):
        logging.warning("User location coordinates missing, cannot calculate distance")
        return {
            'distance_km': None,
            'distance_miles': None,
            'travel_time_driving': None,
            'travel_time_transit': None,
            'travel_time_walking': None
        }
        
    if not restaurant_location.get('lat') and not restaurant_location.get('lng'):
        logging.warning("Restaurant location coordinates missing, cannot calculate distance")
        return {
            'distance_km': None,
            'distance_miles': None,
            'travel_time_driving': None,
            'travel_time_transit': None,
            'travel_time_walking': None
        }
    
    # If gmaps client is not available, do a rough calculation using Haversine formula
    if not gmaps:
        logging.info("Google Maps client not available, using Haversine formula for rough distance calculation")
        import math
        
        # Earth's radius in kilometers
        R = 6371.0
        
        lat1 = math.radians(user_location['lat'])
        lon1 = math.radians(user_location['lng'])
        lat2 = math.radians(restaurant_location['lat'])
        lon2 = math.radians(restaurant_location['lng'])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_km = R * c
        
        # Very rough time estimates
        walking_speed_km_per_hour = 5.0  # Average walking speed
        transit_speed_km_per_hour = 20.0  # Average transit speed including stops
        driving_speed_km_per_hour = 30.0  # Average urban driving speed
        
        walking_time = (distance_km / walking_speed_km_per_hour) * 60  # Convert to minutes
        transit_time = (distance_km / transit_speed_km_per_hour) * 60
        driving_time = (distance_km / driving_speed_km_per_hour) * 60
        
        return {
            'distance_km': round(distance_km, 2),
            'distance_miles': round(distance_km * 0.621371, 2),
            'travel_time_driving': round(driving_time),
            'travel_time_transit': round(transit_time),
            'travel_time_walking': round(walking_time)
        }
    
    # Use Google Maps Distance Matrix API for more accurate calculation
    try:
        # Convert to tuple of coordinates
        origin = (user_location['lat'], user_location['lng'])
        destination = (restaurant_location['lat'], restaurant_location['lng'])
        
        # Calculate driving distance
        driving_result = gmaps.distance_matrix(
            origins=[origin],
            destinations=[destination],
            mode="driving",
            units="metric"
        )
        
        # Calculate transit distance
        transit_result = gmaps.distance_matrix(
            origins=[origin],
            destinations=[destination],
            mode="transit",
            units="metric"
        )
        
        # Calculate walking distance
        walking_result = gmaps.distance_matrix(
            origins=[origin],
            destinations=[destination],
            mode="walking",
            units="metric"
        )
        
        # Extract driving results
        if driving_result['status'] == 'OK' and driving_result['rows'][0]['elements'][0]['status'] == 'OK':
            driving_element = driving_result['rows'][0]['elements'][0]
            distance_km = driving_element['distance']['value'] / 1000  # Convert meters to km
            driving_time = driving_element['duration']['value'] / 60  # Convert seconds to minutes
        else:
            # Fallback to estimate
            distance_km = None
            driving_time = None
            
        # Extract transit results
        if transit_result['status'] == 'OK' and transit_result['rows'][0]['elements'][0]['status'] == 'OK':
            transit_element = transit_result['rows'][0]['elements'][0]
            transit_time = transit_element['duration']['value'] / 60  # Convert seconds to minutes
        else:
            transit_time = None
            
        # Extract walking results
        if walking_result['status'] == 'OK' and walking_result['rows'][0]['elements'][0]['status'] == 'OK':
            walking_element = walking_result['rows'][0]['elements'][0]
            walking_time = walking_element['duration']['value'] / 60  # Convert seconds to minutes
        else:
            walking_time = None
        
        return {
            'distance_km': round(distance_km, 2) if distance_km else None,
            'distance_miles': round(distance_km * 0.621371, 2) if distance_km else None,
            'travel_time_driving': round(driving_time) if driving_time else None,
            'travel_time_transit': round(transit_time) if transit_time else None,
            'travel_time_walking': round(walking_time) if walking_time else None
        }
        
    except Exception as e:
        logging.error(f"Error calculating distance: {e}")
        return {
            'distance_km': None,
            'distance_miles': None,
            'travel_time_driving': None,
            'travel_time_transit': None,
            'travel_time_walking': None
        }


def make_restaurant_reservation(restaurant_id, reservation_details):
    """
    Makes a reservation at a restaurant.
    
    Args:
        restaurant_id: Google Place ID for the restaurant
        reservation_details: Dictionary containing:
            - date: Date of reservation (YYYY-MM-DD)
            - time: Time of reservation (HH:MM)
            - party_size: Number of people
            - name: Customer name
            - email: Customer email
            - phone: Customer phone number
            - special_requests: Any special requests
            
    Returns:
        Dictionary containing:
            - success: Boolean indicating if reservation was successful
            - reservation_id: Unique ID for the reservation if successful
            - message: Success or error message
            - confirmation_details: Details of the confirmed reservation
    """
    logging.info(f"Making reservation at restaurant: {restaurant_id}")
    
    # Validate required fields
    required_fields = ['date', 'time', 'party_size', 'name', 'phone']
    missing_fields = [field for field in required_fields if field not in reservation_details]
    
    if missing_fields:
        error_msg = f"Missing required reservation details: {', '.join(missing_fields)}"
        logging.error(error_msg)
        return {
            'success': False,
            'message': error_msg,
            'reservation_id': None,
            'confirmation_details': None
        }
    
    # In a real implementation, this would interact with a restaurant reservation system
    # or a third-party API like OpenTable, Resy, etc.
    
    # For this example, we'll simulate a successful reservation
    if not gmaps:
        logging.warning("Google Maps client not initialized. Using simulation for reservation.")
        # Simulate API delay
        time.sleep(random.uniform(0.5, 1.5))
        
        # Simulate success with high probability
        success = random.random() < 0.9
        
        if success:
            # Generate a random reservation ID
            import uuid
            reservation_id = str(uuid.uuid4())[:8].upper()
            
            # Format confirmation details
            confirmation_details = {
                'restaurant_name': f"Simulated Restaurant {restaurant_id[-4:]}",
                'date': reservation_details['date'],
                'time': reservation_details['time'],
                'party_size': reservation_details['party_size'],
                'name': reservation_details['name'],
                'reservation_id': reservation_id
            }
            
            logging.info(f"Simulated successful reservation with ID: {reservation_id}")
            
            return {
                'success': True,
                'reservation_id': reservation_id,
                'message': 'Reservation confirmed',
                'confirmation_details': confirmation_details
            }
        else:
            logging.warning(f"Simulated failed reservation")
            return {
                'success': False,
                'reservation_id': None,
                'message': random.choice([
                    'No availability for the requested time',
                    'Restaurant is fully booked',
                    'Reservation system is temporarily unavailable'
                ]),
                'confirmation_details': None
            }
    
    # For a real implementation, you would integrate with a restaurant reservation API
    try:
        # This is a placeholder for a real API integration
        # For example, with OpenTable, Resy, or a direct integration with the restaurant
        
        # In real implementation:
        # 1. Check availability at the restaurant for the given date, time, and party size
        # 2. If available, create the reservation
        # 3. Return the reservation details
        
        # Mock a successful response for now
        import uuid
        reservation_id = str(uuid.uuid4())[:8].upper()
        
        # Get restaurant name (if we had integrated with a real reservation system)
        restaurant_name = "Unknown Restaurant"
        restaurant_details = get_restaurant_details(restaurant_id)
        if restaurant_details and "name" in restaurant_details:
            restaurant_name = restaurant_details["name"]
        
        confirmation_details = {
            'restaurant_name': restaurant_name,
            'date': reservation_details['date'],
            'time': reservation_details['time'],
            'party_size': reservation_details['party_size'],
            'name': reservation_details['name'],
            'reservation_id': reservation_id
        }
        
        logging.info(f"Created reservation with ID: {reservation_id}")
        
        return {
            'success': True,
            'reservation_id': reservation_id,
            'message': 'Reservation confirmed',
            'confirmation_details': confirmation_details
        }
        
    except Exception as e:
        logging.error(f"Error making reservation: {e}")
        return {
            'success': False,
            'reservation_id': None,
            'message': f"Error making reservation: {str(e)}",
            'confirmation_details': None
        }


if __name__ == '__main__':
    # Example usage for testing this utility directly
    # Ensure GOOGLE_MAPS_API_KEY is set in your .env file and loaded
    load_dotenv() # Load .env file from current directory or parent

    # Re-initialize client here if needed for testing script scope
    test_gmaps_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if test_gmaps_key:
        gmaps = googlemaps.Client(key=test_gmaps_key)
        print("Google Maps client re-initialized for test.")
    else:
        print("GOOGLE_MAPS_API_KEY not set. Tests will likely use simulation.")
        gmaps = None # Ensure it's None if key is missing

    test_location = "Soho, London"
    test_cuisine = ["Italian"]
    print(f"Testing find_restaurants in '{test_location}' for {test_cuisine}...")
    candidates = find_restaurants(test_location, test_cuisine)
    print(f"Found candidates: {candidates}")

    if candidates:
        # Use the first candidate for review fetching test
        test_id_for_reviews = candidates[0]['id']
        print(f"\nTesting get_reviews_for_restaurant for id: {test_id_for_reviews}")
        reviews = get_reviews_for_restaurant(test_id_for_reviews)
        if reviews:
            print(f"Fetched {len(reviews)} reviews:")
            for i, rev in enumerate(reviews[:3]): # Print first few
                print(f"  Review {i+1}: {rev[:80]}...")
        else:
            print("No reviews fetched.")
    else:
        print("\nSkipping review fetching test as no candidates were found.") 