import logging
from pocketflow import Node, BatchNode
import json
import time
import re

# Import utility functions (assuming they are in the utils directory)
from utils.google_maps_api import (
    find_restaurants, 
    get_reviews_for_restaurant, 
    get_user_geolocation, 
    calculate_distance_to_restaurant
)
from utils.call_llm import call_llm, parse_user_query_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParseUserQueryNode(Node):
    """Parses the raw user query into structured preferences (location, cuisine, etc)."""
    def prep(self, shared):
        user_query = shared.get("user_query")
        if not user_query:
            raise ValueError("User query not found in shared store.")
        
        # Get user's current location
        from utils.google_maps_api import get_user_geolocation
        user_location = get_user_geolocation()
        
        # Store user location for later use
        shared["user_geolocation"] = user_location
        
        # Log this clearly to distinguish user's physical location vs. where they want to search
        logger.info(f"===== ATTEMPTING TO GET USER GEOLOCATION AUTOMATICALLY =====")
        if user_location.get('success'):
            logger.info(f"===== SUCCESSFULLY DETECTED USER LOCATION: {user_location.get('address', 'Unknown')} =====")
            logger.info(f"Automatically detected user's physical location for distance calculations: {user_location.get('address', 'Unknown')}")
        else:
            logger.warning("Could not automatically detect user's location. Will use search location only.")
        
        return user_query, user_location

    def exec(self, prep_res):
        user_query, user_location = prep_res
        logger.info(f"Parsing user query: {user_query}")
        
        # Call a general LLM to parse the query
        parsed_data = parse_user_query_llm(user_query)
        
        # If LLM failed completely, use more basic extraction
        if not parsed_data:
            logger.warning("Received empty dictionary from parser. Using fallback extraction methods.")
            parsed_data = self._fallback_extraction(user_query, user_location)
        else:
            # If specific fields are missing, try to extract them with regex
            if not parsed_data.get("location"):
                # Look for "in [location]" pattern
                location_pattern = r'\bin\s+([a-zA-Z\s]+)'
                location_match = re.search(location_pattern, user_query)
                if location_match:
                    location = location_match.group(1).strip()
                    parsed_data["location"] = location
                    logger.info(f"Extracted location using regex: {location}")
                else:
                    # Fallback to user's neighborhood if we can't find a location
                    parsed_data["location"] = user_location.get('neighborhood')
                    logger.info(f"Using user's neighborhood as location: {parsed_data['location']}")
            
        return parsed_data

    def _fallback_extraction(self, user_query, user_location):
        """Fallback method to extract basic info when LLM parsing fails"""
        logger.info("Using fallback extraction for query parsing")
        # Extract basic information from the user query directly - very basic parsing
        words = user_query.lower().split()
        
        # Default location detection - look for "in" followed by a word
        location = None
        import re
        location_pattern = r'\bin\s+([a-zA-Z\s]+?)[,\.]'
        location_match = re.search(location_pattern, user_query)
        if location_match:
            location = location_match.group(1).strip()
        
        # Default cuisine detection - look for common cuisine words
        cuisine = []
        common_cuisines = ["italian", "chinese", "japanese", "mexican", "thai", "indian", "french", "vietnamese", "korean"]
        for cuisine_type in common_cuisines:
            if cuisine_type in user_query.lower():
                cuisine.append(cuisine_type.capitalize())
        
        # Default dietary preferences detection
        dietary_prefs = []
        common_dietary = ["vegetarian", "vegan", "gluten-free", "halal", "kosher", "dairy-free", "nut-free"]
        for diet in common_dietary:
            if diet in user_query.lower():
                dietary_prefs.append(diet.capitalize())
        
        # Default group size detection - look for numbers followed by "people" or similar words
        group_size = None
        group_pattern = r'(\d+)\s*(?:people|persons|friends|guests|group)'
        group_match = re.search(group_pattern, user_query.lower())
        if group_match:
            group_size = int(group_match.group(1))
        
        # Time detection
        time_pattern = r'(\d{1,2})(?::(\d{2}))?\s*(?:am|pm|AM|PM)?'
        time_match = re.search(time_pattern, user_query)
        time = None
        if time_match:
            time = time_match.group(0)
            
        # Budget detection
        budget_pp = None
        budget_pattern = r'(\d+)\s*(?:dollars|pounds|euros|gbp|usd|eur|\$|£|€)'
        budget_match = re.search(budget_pattern, user_query.lower())
        if budget_match:
            budget_pp = int(budget_match.group(1))
        
        # Set defaults - use user's geolocation if no specific location found
        parsed_data = {
            "location": location if location else user_location.get('neighborhood', "Soho"),
            "cuisine": cuisine if cuisine else None,
            "budget_pp": budget_pp,
            "vibe": None,
            "priorities": [],
            "dietary_preferences": dietary_prefs,
            "group_size": group_size,
            "time": time
        }
        
        # Try to detect priorities
        if "ambiance" in user_query.lower() or "atmosphere" in user_query.lower():
            parsed_data["priorities"].append("Ambiance")
        if "service" in user_query.lower():
            parsed_data["priorities"].append("Service")
        if "taste" in user_query.lower() or "food" in user_query.lower():
            parsed_data["priorities"].append("Taste")
        if "value" in user_query.lower() or "price" in user_query.lower() or "budget" in user_query.lower():
            parsed_data["priorities"].append("Value")
            
        return parsed_data

    def post(self, shared, prep_res, exec_res):
        shared["parsed_query"] = exec_res
        
        # Also store individual important fields at the top level for easy access
        if exec_res:
            shared["user_location"] = exec_res.get("location")
            shared["user_cuisine"] = exec_res.get("cuisine")
            shared["user_budget"] = exec_res.get("budget_pp")
            shared["user_group_size"] = exec_res.get("group_size")
            shared["user_time"] = exec_res.get("time")
            shared["user_dietary_preferences"] = exec_res.get("dietary_preferences")
            
        logger.info(f"Stored parsed query: {exec_res}")
        return "default"


class FindRestaurantsNode(Node):
    """Finds restaurant candidates using Google Maps API based on parsed query."""
    def prep(self, shared):
        query = shared.get("parsed_query")
        if not query:
            logger.warning("No parsed query available to find restaurants.")
            return None
        
        # Get user's geolocation for distance calculations
        user_geolocation = shared.get("user_geolocation")
        if not user_geolocation:
            # Check if it's in the conversation state
            conv_state = shared.get("conversation_state", {})
            extracted_info = conv_state.get("extracted_info", {})
            user_geolocation = extracted_info.get("user_geolocation")
            
            # If still not found, fetch it now
            if not user_geolocation:
                logger.info("User geolocation not found, fetching it now.")
                # Fetch user geolocation and store it
                user_geolocation = get_user_geolocation()
                shared["user_geolocation"] = user_geolocation
            else:
                shared["user_geolocation"] = user_geolocation
                
        return query, user_geolocation

    def exec(self, prep_res):
        if prep_res is None:
            return [], [], {}
            
        query, user_geolocation = prep_res
        # Get location and cuisine from the parsed query
        location = query.get("location")
        cuisine = query.get("cuisine", [])
        dietary_preferences = query.get("dietary_preferences", [])

        logger.info(f"Searching for restaurants: {cuisine} in {location}")
        
        # Call Google Maps API to find candidates
        candidates = find_restaurants(location, cuisine)
        
        # If we have user geolocation, calculate distance for each restaurant
        if user_geolocation and user_geolocation.get('success'):
            for candidate in candidates:
                # Extract restaurant coordinates
                if not candidate.get('geometry') or not candidate['geometry'].get('location'):
                    continue
                    
                restaurant_location = {
                    'lat': candidate['geometry']['location'].get('lat'),
                    'lng': candidate['geometry']['location'].get('lng')
                }
                
                # Calculate distance and travel time
                user_coords = {
                    'lat': user_geolocation.get('lat'),
                    'lng': user_geolocation.get('lng')
                }
                
                distance_info = calculate_distance_to_restaurant(
                    user_coords, restaurant_location
                )
                
                # Add distance info to the candidate
                candidate['distance_info'] = distance_info
        
        return candidates, dietary_preferences, user_geolocation

    def post(self, shared, prep_res, exec_res):
        candidates, dietary_preferences, user_geolocation = exec_res
        
        if not candidates:
            logger.warning("No restaurant candidates found.")
            shared["candidate_restaurants"] = []
            # Optionally, return a specific action to handle this (e.g., inform user)
            return "no_candidates_found" # Example action
        else:
            # For each candidate, fetch detailed information including opening hours
            detailed_candidates = []
            for candidate in candidates:
                if candidate.get("id"):
                    details = get_reviews_for_restaurant(candidate["id"], max_reviews=25)
                    if details:
                        # Wrap the reviews list in a dictionary since get_reviews_for_restaurant 
                        # returns a list, not a dictionary
                        candidate["reviews"] = details
                        
                        # If we have dietary preferences, mark whether restaurant is likely to accommodate
                        if dietary_preferences:
                            # Extract business description, categories, reviews etc. to evaluate dietary match
                            text_to_check = ""
                            if "name" in candidate:
                                text_to_check += candidate["name"].lower() + " "
                            if "types" in candidate:
                                text_to_check += " ".join(candidate.get("types", [])).lower() + " "
                                
                            # Check if restaurant potentially accommodates dietary preferences
                            accommodates_diet = False
                            diet_keywords = {
                                "vegetarian": ["vegetarian", "veggie"],
                                "vegan": ["vegan", "plant-based", "plant based"],
                                "gluten-free": ["gluten-free", "gluten free", "gf"],
                                "halal": ["halal"],
                                "kosher": ["kosher"],
                                "dairy-free": ["dairy-free", "dairy free", "no dairy"],
                                "nut-free": ["nut-free", "nut free", "no nuts"]
                            }
                            
                            for diet in dietary_preferences:
                                diet_lower = diet.lower()
                                if diet_lower in diet_keywords:
                                    for keyword in diet_keywords[diet_lower]:
                                        if keyword in text_to_check:
                                            accommodates_diet = True
                                            break
                            
                            candidate["accommodates_dietary_preferences"] = accommodates_diet
                        
                        detailed_candidates.append(candidate)
            
            # If dietary preferences were specified, prioritize restaurants that accommodate them
            if dietary_preferences:
                # Move accommodating restaurants to the front of the list
                detailed_candidates.sort(
                    key=lambda x: (0 if x.get('accommodates_dietary_preferences', False) else 1)
                )
            
            shared["candidate_restaurants"] = detailed_candidates
            shared["user_dietary_preferences"] = dietary_preferences
            shared["user_geolocation"] = user_geolocation
            logger.info(f"Stored {len(detailed_candidates)} candidate restaurants and user geolocation.")
            return "default"


class FetchReviewsNode(Node):
    """Fetches reviews for each candidate restaurant."""
    # This could be a BatchNode if fetching is slow and candidates are many
    # For simplicity, keeping it a regular Node that iterates internally
    def prep(self, shared):
        candidates = shared.get("candidate_restaurants")
        if not candidates:
            logger.warning("No candidates to fetch reviews for. Skipping.")
            return None # Signal to exec to do nothing
        return candidates

    def exec(self, candidates):
        if candidates is None:
            return {}

        reviews_data = {}
        logger.info(f"Fetching reviews for {len(candidates)} candidates...")
        for candidate in candidates:
            resto_id = candidate.get("id")
            if resto_id:
                # Limit the number of reviews per restaurant if needed
                reviews = get_reviews_for_restaurant(resto_id, max_reviews=25)
                reviews_data[resto_id] = reviews
                # Optional: Add a small delay if hitting API limits aggressively
                # time.sleep(0.1)
            else:
                logger.warning(f"Candidate missing id: {candidate.get('name')}")
        logger.info(f"Finished fetching reviews.")
        return reviews_data

    def post(self, shared, prep_res, exec_res):
        shared["reviews_data"] = exec_res
        return "default"


class AnalyzeRestaurantReviewsNode(Node):
    """Analyzes restaurant reviews to extract scores on various dimensions."""
    
    def __init__(self, max_retries=2, wait=5):
        super().__init__(max_retries=max_retries, wait=wait)
    
    def prep(self, shared):
        # Get the list of candidate restaurants and their reviews
        candidates = shared.get("candidate_restaurants", [])
        reviews_data = shared.get("reviews_data", {})
        parsed_query = shared.get("parsed_query", {})
        
        # Identify if there are special dimensions to examine based on user query
        dimensions = ["Taste", "Service", "Ambiance", "Value"]
        
        # Add dietary dimensions if user mentioned dietary preferences
        dietary_preferences = parsed_query.get("dietary_preferences", [])
        if dietary_preferences:
            for pref in dietary_preferences:
                pref_normalized = pref.capitalize()
                if pref_normalized not in dimensions:
                    dimensions.append(pref_normalized)
        
        logger.info(f"Extracted dimensions based on user input: {dimensions}")
        
        # Return the candidates and dimensions to analyze
        return {
            "candidates": candidates,
            "reviews_data": reviews_data,
            "dimensions": dimensions,
            "parsed_query": parsed_query
        }
    
    def exec(self, prep_res):
        candidates = prep_res["candidates"]
        reviews_data = prep_res["reviews_data"]
        dimensions = prep_res["dimensions"]
        parsed_query = prep_res["parsed_query"]
        
        # Prepare for analysis
        dimensional_scores = {}  # Will hold scores for each restaurant by ID
        logger.info(f"Prepared {len(candidates)} restaurants for review analysis with dimensions: {dimensions}")
        
        from utils.call_llm import call_llm
        
        # For each restaurant, analyze its reviews
        for candidate in candidates:
            restaurant_id = candidate.get("id")
            if not restaurant_id or restaurant_id not in reviews_data:
                continue
            
            reviews = reviews_data[restaurant_id]
            if not reviews:
                continue
            
            # Format reviews for analysis
            reviews_text = ""
            for i, review in enumerate(reviews):
                text = review.strip() if isinstance(review, str) else review.get("text", "").strip()
                if text:
                    reviews_text += f"Review {i+1}: {text}\n\n"
            
            if not reviews_text:
                continue
            
            # Prepare prompt for LLM to analyze reviews
            prompt = f"""
Analyze these customer reviews for a restaurant and provide scores (1-10 scale) for the following dimensions:
{', '.join(dimensions)}

For each dimension, a higher score is better:
- 10: Exceptional, world-class
- 8-9: Excellent
- 6-7: Good
- 4-5: Average
- 2-3: Below average
- 1: Poor

Only score dimensions that are explicitly mentioned or strongly implied in the reviews.
If a dimension isn't mentioned at all, use null instead of a score.

REVIEWS:
{reviews_text}

Respond with a JSON object containing only the scores. For example:
{{
  "Taste": 8,
  "Service": 7,
  "Ambiance": 9,
  "Value": 6,
  "Vegetarian": null
}}
"""
            # Using OpenAI (gpt-4o-mini) for this analysis
            logger.info(f"Analyzing reviews for {restaurant_id} using general LLM (attempt {self.cur_retry + 1})...")
            response = call_llm(prompt, model="gpt-4o-mini")
            
            # Parse the response - expect JSON format
            if response:
                try:
                    # First, check if the entire response is a valid JSON
                    try:
                        scores = json.loads(response.strip())
                        json_str = response.strip()
                    except json.JSONDecodeError:
                        # If not, try to extract the JSON part using regex
                        # Extract just the JSON part (in case there's thinking or explanation)
                        json_parts = re.findall(r'```json\n(.*?)\n```|```(.*?)```|^\s*{.*}$', response, re.DOTALL|re.MULTILINE)
                        
                        if json_parts:
                            # Handle both formats of findall results
                            for part in json_parts:
                                if isinstance(part, tuple):
                                    # If it's a tuple, check each element
                                    for subpart in part:
                                        if subpart.strip():
                                            json_str = subpart.strip()
                                            break
                                else:
                                    json_str = part.strip()
                                    break
                        else:
                            # Try to find JSON without code blocks
                            json_match = re.search(r'{[\s\S]*?}', response)
                            if json_match:
                                json_str = json_match.group(0).strip()
                            else:
                                json_str = response.strip()
                        
                        # Parse the JSON
                        scores = json.loads(json_str)
                    
                    dimensional_scores[restaurant_id] = scores
                    logger.info(f"Successfully parsed scores for {restaurant_id} from LLM.")
                    
                    # Also update the candidate directly with the scores
                    # This will allow further nodes to access the scores if needed
                    candidate["dimensional_scores"] = scores
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response from LLM for {restaurant_id}: {response}")
                    logger.error(f"Error during LLM review analysis for {restaurant_id}: LLM did not return valid JSON.")
                except Exception as e:
                    logger.error(f"Error during LLM review analysis for {restaurant_id}: {str(e)}")
        
        # Check if we have any scores
        if not dimensional_scores:
            logger.warning("No dimensional scores were successfully parsed from LLM responses.")
            return {}
            
        return dimensional_scores
    
    def post(self, shared, prep_res, exec_res):
        # Store the dimensional scores in the shared store
        shared["dimensional_scores_1_10"] = exec_res
        
        # Count successful analyses
        score_count = sum(1 for scores in exec_res.values() if scores)
        logger.info(f"Stored dimensional scores (1-10 scale) for {score_count} restaurants.")
        
        return "default"


class CalculateFitScoreNode(Node):
    """Calculates personalized fit scores based on dimensional scores and user priorities."""
    def prep(self, shared):
        user_priorities = shared.get("parsed_query", {}).get("priorities", [])
        # 确保user_priorities不为None
        if user_priorities is None:
            user_priorities = []
            logger.warning("Found None priorities in parsed_query during fit calculation, defaulting to empty list")
            
        dimensional_scores = shared.get("dimensional_scores_1_10", {})
        candidate_restaurants = shared.get("candidate_restaurants", [])
        user_additional_prefs = shared.get("parsed_query", {}).get("additional_preferences", {})
        
        # Only proceed if we have both scores and candidates
        if not dimensional_scores or not candidate_restaurants:
            logger.warning("Missing scores or candidates for fit calculation. Skipping.")
            return None
            
        return user_priorities, dimensional_scores, candidate_restaurants, user_additional_prefs

    def exec(self, prep_res):
        if prep_res is None:
            return []
            
        user_priorities, dimensional_scores, candidates, user_additional_prefs = prep_res
        
        # 确保user_priorities不为None
        if user_priorities is None:
            user_priorities = []
            logger.warning("Found None priorities in prep_res during fit calculation, defaulting to empty list")
            
        # List to store restaurant info with fit scores and explanations
        ranked_restaurants = []
        
        # Standardize priorities to match available dimension keys
        # This handles case-insensitive matching and common variants
        dimension_key_map = {
            "taste": "Taste", 
            "food": "Taste",
            "flavor": "Taste",
            "service": "Service",
            "staff": "Service",
            "waitstaff": "Service",
            "ambiance": "Ambiance",
            "atmosphere": "Ambiance",
            "vibe": "Ambiance",
            "decor": "Ambiance",
            "value": "Value",
            "price": "Value",
            "worth": "Value",
            "wait": "Waiting",
            "waiting": "Waiting",
            "waittime": "Waiting",
            "noise": "Noise",
            "quiet": "Noise",
            "loud": "Noise"
        }
        
        # Map user priorities to dimension keys
        priority_dimensions = []
        for priority in user_priorities:
            priority_lower = priority.lower()
            if priority_lower in dimension_key_map:
                priority_dimensions.append(dimension_key_map[priority_lower])
        
        # Process each candidate restaurant
        for candidate in candidates:
            restaurant_id = candidate.get("id")
            if not restaurant_id or restaurant_id not in dimensional_scores:
                continue
                
            # Get all dimensional scores for this restaurant
            scores = dimensional_scores[restaurant_id]
            if not scores:
                continue
            
            # Calculate base fit score from dimensional scores
            weighted_sum = 0.0
            weight_sum = 0.0
            weighted_counts = {}  # Keep track of weighted contributions
            
            # Define core dimensions that should always be included in score
            core_dimensions = ["Taste", "Service", "Ambiance", "Value"]
            
            # Step 1: Calculate initial weighted average based on user priorities
            for dim, score in scores.items():
                if score is None:
                    continue
                    
                # Apply weight based on whether this dimension is a priority for the user
                weight = 2.0 if dim in priority_dimensions else 1.0
                
                # For "Noise" dimension, invert scoring if "quiet" is explicitly a priority
                # Higher noise score = quieter place, which is better for those prioritizing quiet
                adjusted_score = score
                
                weighted_sum += weight * adjusted_score
                weight_sum += weight
                
                # Track how much each dimension contributed to total
                weighted_counts[dim] = weight * adjusted_score
            
            # Calculate base fit score (0-10)
            base_fit_score = weighted_sum / weight_sum if weight_sum > 0 else 0
            
            # Step 2: Apply modifiers for additional user preferences
            # These are adjustments to the base score based on special requirements
            preference_adjustments = 0
            preference_adjustment_reasons = []
            
            # Noise level preferences
            if "quiet" in user_additional_prefs.get("noise_level", "").lower():
                # If user wants quiet and noise score is good (>7), add bonus
                if "Noise" in scores and scores["Noise"] >= 7:
                    preference_adjustments += 0.5
                    preference_adjustment_reasons.append("Meets quiet atmosphere requirement")
                # If user wants quiet but place is noisy, penalize
                elif "Noise" in scores and scores["Noise"] <= 4:
                    preference_adjustments -= 1.0
                    preference_adjustment_reasons.append("Place might be too noisy for your preference")
            
            # Outdoor seating preference
            if "outdoor" in user_additional_prefs.get("seating", "").lower():
                # Check if restaurant has outdoor seating data
                has_outdoor = False
                for feature in candidate.get("types", []):
                    if "outdoor_seating" in feature:
                        has_outdoor = True
                        preference_adjustments += 0.5
                        preference_adjustment_reasons.append("Has outdoor seating")
                        break
                if not has_outdoor:
                    preference_adjustments -= 0.3
                    preference_adjustment_reasons.append("No confirmed outdoor seating")
            
            # Check if restaurant accommodates dietary preferences
            if candidate.get("accommodates_dietary_preferences"):
                preference_adjustments += 0.7
                preference_adjustment_reasons.append("Accommodates your dietary preferences")
                
            # Check distance if user mentioned travel time concerns
            if "distance" in user_additional_prefs or "near" in user_additional_prefs.get("location_attributes", "").lower():
                if "distance_info" in candidate and candidate["distance_info"].get("distance_km", 10) < 3:
                    preference_adjustments += 0.5
                    preference_adjustment_reasons.append("Close to your current location")
                    
            # Group size considerations - use from user_additional_prefs instead of shared
            group_size = user_additional_prefs.get("group_size")
            # If not in additional_prefs, check if it's in the candidate data
            if not group_size and candidate.get("group_size"):
                group_size = candidate.get("group_size")
                
            if group_size and group_size > 5:
                # For large groups, prioritize restaurants that likely have space
                good_for_groups = False
                for feature in candidate.get("types", []):
                    if "large_groups" in feature:
                        good_for_groups = True
                        preference_adjustments += 0.5
                        preference_adjustment_reasons.append(f"Good for your group of {group_size}")
                        break
                        
            # Apply preference adjustments to base fit score
            final_fit_score = min(10, max(0, base_fit_score + preference_adjustments))
            
            # Construct the ranked restaurant entry
            ranked_entry = {
                "id": restaurant_id,
                "name": candidate.get("name", "Unknown Restaurant"),
                "scores": scores,
                "dimensions_used": list(scores.keys()),
                "fit_score": round(final_fit_score, 1),
                "base_fit_score": round(base_fit_score, 1),
                "preference_adjustments": round(preference_adjustments, 1),
                "adjustment_reasons": preference_adjustment_reasons,
                "review_count": candidate.get("user_ratings_total", 0),
                "formattedAddress": candidate.get("formatted_address"),
                "rating": candidate.get("rating"),
                "price_level": candidate.get("price_level"),
                "user_priorities": priority_dimensions
            }
            
            # Add original candidate data for additional details like location, opening hours
            for key, value in candidate.items():
                if key not in ranked_entry:
                    ranked_entry[key] = value
                    
            ranked_restaurants.append(ranked_entry)
        
        # Sort by fit score
        ranked_restaurants.sort(key=lambda x: x["fit_score"], reverse=True)
        
        return ranked_restaurants

    def post(self, shared, prep_res, exec_res):
        shared["ranked_recommendations"] = exec_res
        logger.info(f"Ranked {len(exec_res)} restaurants based on fit scoring.")
        return "default"


class GenerateResponseNode(Node):
    """Generates the final response to present to the user."""
    def prep(self, shared):
        """Collect all the data needed for the final response."""
        ranked_recommendations = shared.get("ranked_recommendations", [])
        user_query = shared.get("user_query", "")
        user_geolocation = shared.get("user_geolocation", {})
        user_location = user_geolocation.get("address", "Unknown location")
        
        return user_query, ranked_recommendations, user_location
        
    def exec(self, prep_res):
        """Generate the final response using call_llm or templates."""
        user_query, ranked_recommendations, user_location = prep_res
        
        if not ranked_recommendations:
            logger.warning("No recommendations to present in final response")
            return "I couldn't find any restaurants that match your criteria. Would you like to try different preferences?"
        
        try:
            from utils.call_llm import call_llm
            
            # 增强处理：确保每个餐厅记录都包含评分和用户优先事项
            for i, resto in enumerate(ranked_recommendations):
                # 确保评分使用星号表示
                if "scores" in resto:
                    # 标准化评分，添加简短描述
                    refined_scores = {}
                    for dim, score in resto["scores"].items():
                        if score is not None:
                            # 添加描述性文本
                            description = ""
                            if dim == "Taste" and score >= 8.5:
                                description = "(Exceptional)"
                            elif dim == "Ambiance" and score >= 8.0:
                                description = resto.get("ambiance_description", 
                                                      "(Excellent Atmosphere)")
                            elif dim == "Service" and score >= 8.0:
                                description = "(Attentive)"
                            elif dim == "Value" and score <= 6.0:
                                description = "(Higher end)"
                            elif dim == "Value" and score >= 8.0:
                                description = "(Great value)"
                                
                            refined_scores[dim] = {
                                "score": score,
                                "description": description
                            }
                    resto["refined_scores"] = refined_scores
                
                # 确保包含明确的匹配指标
                if "user_priorities" in resto and resto["user_priorities"]:
                    matched_priorities = []
                    for priority in resto["user_priorities"]:
                        if priority in resto.get("scores", {}) and resto["scores"][priority] >= 7.5:
                            matched_priorities.append(priority)
                    resto["matched_priorities"] = matched_priorities
                
                # 添加亮点标记
                if "adjustment_reasons" in resto and resto["adjustment_reasons"]:
                    resto["highlights"] = resto["adjustment_reasons"][:3]  # 最多使用前3个亮点
            
            # 首先使用模板生成响应以确保格式正确，不依赖LLM
            template_response = self._generate_template_response(ranked_recommendations[:3], user_query)
            
            # 尝试通过LLM生成更自然的内容，但保持相同的格式结构
            try:
                # 生成最终响应
                llm_response = call_llm.generate_final_response_llm(
                    user_query, 
                    ranked_recommendations[:3],  # 只使用前3个推荐
                    user_location
                )
                
                # 检查LLM是否生成了有效格式的响应
                if llm_response and ("Restaurant Analysis" in llm_response or "|" in llm_response):
                    logger.info("Using LLM response with proper formatting")
                    return llm_response
                else:
                    logger.warning("LLM response missing proper formatting, using template instead")
                    return template_response
            except Exception as e:
                logger.error(f"Error in LLM response generation: {e}")
                return template_response
                
        except Exception as e:
            logger.error(f"Error generating final response: {e}", exc_info=True)
            # Fall back to a simple template
            return self._generate_template_response(ranked_recommendations[:3], user_query)
            
    def _generate_template_response(self, recommendations, user_query):
        """Generate a basic response if the LLM call fails."""
        # 提取用户查询中的关键信息
        location = "Unknown"
        cuisine = "food"
        budget = "medium budget"
        vibe = "Unknown"
        
        # 尝试从查询中提取一些关键信息
        if isinstance(user_query, dict):
            location = user_query.get("location", "the area")
            if "cuisine" in user_query and user_query["cuisine"]:
                if isinstance(user_query["cuisine"], list):
                    cuisine = ", ".join(user_query["cuisine"])
                else:
                    cuisine = user_query["cuisine"]
            budget = f"£{user_query.get('budget_pp', '??')}pp" if user_query.get('budget_pp') else "your budget"
            vibe = user_query.get("vibe", "place")
        elif isinstance(user_query, str):
            # 简单的文本分析提取位置
            if "in " in user_query:
                parts = user_query.split("in ")
                location_part = parts[1].split(" ")[0].strip(",. ")
                if location_part:
                    location = location_part
        
        # 添加强制换行所需的空格
        def pad_text(text, length=25):
            if len(text) >= length:
                return text
            return text + " " * (length - len(text))
        
        # 使用类似图片中的表格结构格式
        response = f"""Okay there, I've analyzed recent feedback for top restaurants in {location} based on your preferences. Here's a structured recommendation to simplify your decision:

### {location.title()} Restaurant Analysis (Based on Your Specific Needs):

"""
        
        # 表格标题行
        if len(recommendations) >= 3:
            resto_names = [pad_text(r.get("name", "Restaurant")) for r in recommendations[:3]]
            response += f"{resto_names[0]}  {resto_names[1]}  {resto_names[2]}\n"
            
            # 表格分数行
            fit_scores = [f"Fit: {r.get('fit_score', 0):.1f}/10" for r in recommendations[:3]]
            fit_scores = [pad_text(score) for score in fit_scores]
            response += f"{fit_scores[0]}  {fit_scores[1]}  {fit_scores[2]}\n\n"
            
            # 各维度分数
            dimensions = ["Taste", "Service", "Ambiance", "Value"]
            for dim in dimensions:
                dim_scores = []
                for r in recommendations[:3]:
                    score = r.get("scores", {}).get(dim)
                    if score is not None:
                        # 添加描述性文本
                        description = ""
                        if dim == "Ambiance":
                            if score >= 8.0:
                                description = " (Great Vibe)"
                            elif score >= 6.0:
                                description = " (Good)"
                        
                        score_text = f"{dim}: {score:.1f} ⭐{description}"
                    else:
                        score_text = f"{dim}: N/A"
                    dim_scores.append(pad_text(score_text))
                response += f"{dim_scores[0]}  {dim_scores[1]}  {dim_scores[2]}\n"
            
            # 添加亮点行
            highlights = []
            for r in recommendations[:3]:
                if r.get("adjustment_reasons"):
                    highlight = "✅ " + r["adjustment_reasons"][0]
                else:
                    highlight = "✅ Good Choice"
                highlights.append(pad_text(highlight))
            response += f"\n{highlights[0]}  {highlights[1]}  {highlights[2]}\n"
            
            # 添加摘要
            response += "\n### Quick Summary:\n\n"
            for i, r in enumerate(recommendations[:3]):
                name = r.get("name", f"Restaurant {i+1}")
                summary = ""
                if i == 0:
                    summary = "Strongest match overall, especially good value."
                elif i == 1:
                    summary = "Great alternative with excellent ambiance."
                else:
                    summary = "Solid option with unique character."
                response += f"**{name}**: {summary}\n"
            
            # 添加最终建议
            response += f"\nBased on this, **{recommendations[0].get('name', 'the first option')}** seems the closest match, with **{recommendations[1].get('name', 'the second option')}** as a strong alternative. What do you think?"
        
        else:
            # 如果推荐不足3个，就用列表格式
            for i, resto in enumerate(recommendations):
                name = resto.get("name", "Restaurant")
                address = resto.get("formatted_address", resto.get("vicinity", "Address unavailable"))
                rating = resto.get("rating", "N/A")
                fit_score = resto.get("fit_score", 0)
                
                response += f"**{i+1}. {name}** (Fit Score: {fit_score:.1f}/10)\n"
                response += f"   - Address: {address}\n"
                response += f"   - Rating: {rating}/5\n"
                
                # Add scores if available
                if "scores" in resto:
                    score_text = ", ".join([f"{k}: {v:.1f}⭐" for k, v in resto["scores"].items() if v is not None])
                    response += f"   - Scores: {score_text}\n"
                
                response += "\n"
        
        return response

    def post(self, shared, prep_res, exec_res):
        if not exec_res:
            exec_res = "I'm sorry, I couldn't generate restaurant recommendations at this time. Please try again with different criteria."
            
        shared["final_response"] = exec_res
        logger.info("Stored final recommendation response.")
        return "default" # End the flow


class NoCandidatesFoundNode(Node):
    """Provides helpful response when no restaurants match initial criteria."""
    def prep(self, shared):
        # Get the parsed query to understand what user was looking for
        parsed_query = shared.get("parsed_query", {})
        location = parsed_query.get("location")
        cuisine = parsed_query.get("cuisine")
        budget = parsed_query.get("budget_pp")
        
        # Get user's geolocation
        user_geolocation = shared.get("user_geolocation", {})
        
        # Try to get alternative restaurants with relaxed criteria
        alternatives = []
        
        # Try to fetch restaurants with just location
        if location:
            try:
                from utils.google_maps_api import find_restaurants
                # First try: same location, any cuisine
                if cuisine:
                    alternatives = find_restaurants(location, [])
                    
                # If still no results, try nearby areas
                if not alternatives and user_geolocation and user_geolocation.get('success'):
                    # Use user's actual location for broader search
                    alternatives = find_restaurants(
                        user_geolocation.get('neighborhood', 'London'), 
                        cuisine if isinstance(cuisine, list) else [cuisine] if cuisine else []
                    )
            except Exception as e:
                logger.error(f"Error fetching alternative restaurants: {e}")
        
        return parsed_query, alternatives

    def exec(self, prep_res):
        parsed_query, alternatives = prep_res
        location = parsed_query.get("location", "your area")
        cuisine = parsed_query.get("cuisine")
        cuisine_str = ""
        if cuisine:
            if isinstance(cuisine, list):
                cuisine_str = ", ".join(cuisine)
            else:
                cuisine_str = cuisine
        
        # If we found alternatives, include them in the response
        if alternatives and len(alternatives) > 0:
            from utils.call_llm import call_llm
            
            # Create a brief description of the top 3 alternatives
            alt_descriptions = []
            for i, resto in enumerate(alternatives[:3]):
                name = resto.get("name", "Unknown Restaurant")
                address = resto.get("vicinity", "Address unavailable")
                rating = resto.get("rating", "No rating")
                price_level = "".join(["$"] * (resto.get("price_level", 2)))
                
                alt_descriptions.append(f"{name} ({price_level}, {rating}/5) - {address}")
            
            # Generate a helpful response with alternatives
            prompt = f"""
I couldn't find restaurants that match all your criteria exactly. However, here are some alternatives I found in {location}:

{alt_descriptions}

Write a helpful, encouraging response that:
1. Acknowledges that I couldn't find exact matches for their criteria
2. Presents these alternatives as potentially good options
3. Explains how they might differ from what the user wanted (e.g., different cuisine, price point, etc.)
4. Asks if they'd like more details about any of these options
5. Suggests they could try a different search with broader criteria

Keep your response friendly and helpful, focusing on what I CAN offer rather than what I couldn't find.
"""
            try:
                response = call_llm.call_llm(prompt)
                if response:
                    return response
            except Exception as e:
                logger.error(f"Error generating alternative suggestions: {e}")
        
        # Fallback default message if no alternatives or LLM fails
        message = f"""
I couldn't find any restaurants that match your criteria exactly. Here are some suggestions:

1. Try broadening your search to a larger area beyond {location}
2. Consider alternative cuisines{f" besides {cuisine_str}" if cuisine_str else ""}
3. Adjust your budget or other requirements
4. If you provide more flexible criteria, I'd be happy to try again!

I'm here to help you find great dining options, so please let me know how you'd like to proceed.
"""
        return message

    def post(self, shared, _, exec_res):
        # Store the message as the final response
        shared["final_response"] = exec_res
        return "default"


class ReservationNode(Node):
    """Makes a reservation at a selected restaurant."""
    def prep(self, shared):
        reservation_details = shared.get("reservation_details", {})
        restaurant_id = reservation_details.get("restaurant_id")
        
        if not restaurant_id:
            logger.warning("No restaurant ID provided for reservation")
            return None
            
        # Get user details for reservation
        customer_name = reservation_details.get("name")
        customer_phone = reservation_details.get("phone")
        party_size = reservation_details.get("party_size")
        reservation_date = reservation_details.get("date")
        reservation_time = reservation_details.get("time")
        special_requests = reservation_details.get("special_requests", "")
        
        # Check for required fields
        missing_fields = []
        if not customer_name:
            missing_fields.append("name")
        if not customer_phone:
            missing_fields.append("phone")
        if not party_size:
            missing_fields.append("party_size")
        if not reservation_date:
            missing_fields.append("date")
        if not reservation_time:
            missing_fields.append("time")
            
        if missing_fields:
            logger.warning(f"Missing required reservation fields: {missing_fields}")
            return restaurant_id, missing_fields, None
            
        # All required fields present
        reservation_data = {
            "name": customer_name,
            "phone": customer_phone,
            "party_size": party_size,
            "date": reservation_date,
            "time": reservation_time,
            "special_requests": special_requests
        }
        
        return restaurant_id, None, reservation_data

    def exec(self, prep_res):
        if prep_res is None:
            return {
                "success": False,
                "message": "No restaurant ID provided for reservation",
                "missing_fields": [],
                "confirmation": None
            }
        
        restaurant_id, missing_fields, reservation_data = prep_res
        
        if missing_fields:
            return {
                "success": False,
                "message": f"Missing required reservation information: {', '.join(missing_fields)}",
                "missing_fields": missing_fields,
                "confirmation": None
            }
            
        # Make the reservation using Google Maps API
        from utils.google_maps_api import make_restaurant_reservation
        logger.info(f"Making reservation at restaurant {restaurant_id}")
        
        reservation_result = make_restaurant_reservation(
            restaurant_id, reservation_data
        )
        
        if reservation_result.get("success"):
            logger.info(f"Reservation successful with ID: {reservation_result.get('reservation_id')}")
            return {
                "success": True,
                "message": "Reservation confirmed!",
                "missing_fields": [],
                "confirmation": reservation_result.get("confirmation_details")
            }
        else:
            logger.warning(f"Reservation failed: {reservation_result.get('message')}")
            return {
                "success": False,
                "message": reservation_result.get("message", "Unable to complete reservation"),
                "missing_fields": [],
                "confirmation": None
            }

    def post(self, shared, prep_res, exec_res):
        shared["reservation_result"] = exec_res
        
        # Format a user-friendly response
        if exec_res.get("success"):
            confirmation = exec_res.get("confirmation", {})
            response = f"""
## Reservation Confirmed!

Your reservation has been confirmed at **{confirmation.get('restaurant_name', 'the restaurant')}**.

**Reservation Details:**
- **Date:** {confirmation.get('date', 'N/A')}
- **Time:** {confirmation.get('time', 'N/A')}
- **Party Size:** {confirmation.get('party_size', 'N/A')}
- **Reservation ID:** {confirmation.get('reservation_id', 'N/A')}

Please save your reservation ID for reference.
"""
        else:
            missing_fields = exec_res.get("missing_fields", [])
            if missing_fields:
                response = f"To complete your reservation, I'll need the following information: {', '.join(missing_fields)}"
            else:
                response = f"I'm sorry, but I couldn't complete your reservation. {exec_res.get('message', '')}"
        
        shared["reservation_response"] = response
        logger.info("Processed reservation request")
        return "complete"


class DecideActionNode(Node):
    """Uses the Flock Agent model to decide the next step based on context."""
    def __init__(self, *args, **kwargs):
        # Define the tools available for the Flock agent
        # These descriptions are crucial for the model to make good decisions
        self.available_tools = [
            {"type": "function", "function": {
                "name": "analyze_restaurant_reviews",
                "description": "Analyze detailed user reviews for a list of restaurants to get scores on Taste, Service, Ambiance, and Value. Use this if candidates and reviews are available and seem relevant to the user query.",
                "parameters": {"type": "object", "properties": {
                    "restaurant_ids": {"type": "array", "items": {"type": "string"}, "description": "List of restaurant IDs to analyze."}
                }, "required": ["restaurant_ids"]}
            }},
            {"type": "function", "function": {
                "name": "ask_user_clarification",
                "description": "Ask the user a clarifying question if the initial request is too vague, ambiguous, or missing essential information (like location, specific cuisine preferences, or budget if applicable).",
                "parameters": {"type": "object", "properties": {
                    "question": {"type": "string", "description": "The specific question to ask the user."}
                }, "required": ["question"]}
            }},
            # Optional: Add more tools like presenting a basic list if needed
            # {"type": "function", "function": {
            #     "name": "present_basic_list",
            #     "description": "Present a simple list of restaurants found if detailed review analysis is not feasible or seems unnecessary.",
            #     "parameters": {"type": "object", "properties": {
            #         "restaurant_names": {"type": "array", "items": {"type": "string"}}
            #     }}
            # }}
        ]
        # Set retry parameters for the Flock agent calls
        super().__init__(max_retries=1, wait=3, *args, **kwargs) # Fewer retries for agent decision

    def prep(self, shared):
        """Prepare context for the Flock agent."""
        user_query = shared.get("user_query", "")
        parsed_query = shared.get("parsed_query", {})
        candidates = shared.get("candidate_restaurants", [])
        reviews_data = shared.get("reviews_data", {})

        # Construct a context string
        context_lines = [f"User Query: {user_query}"]
        context_lines.append(f"Parsed Preferences: {json.dumps(parsed_query)}")
        if candidates:
            context_lines.append(f"Found {len(candidates)} candidate restaurants: {[c.get('name') for c in candidates]}")
            # Check if reviews are available for candidates
            reviews_available_count = sum(1 for c in candidates if reviews_data.get(c.get('id')))
            context_lines.append(f"Reviews available for {reviews_available_count} of them.")
        else:
            context_lines.append("No restaurant candidates found yet.")

        context_lines.append("Based on this information and the user query, decide the most appropriate next action using the available functions.")
        context = "\n".join(context_lines)

        logger.debug(f"Prepared context for Flock agent:\n{context}")
        return context

    def exec(self, context):
        """Call the Flock agent utility to get the next action decision."""
        logger.info("Skipping Flock agent and defaulting to analyze action...")
        
        # Return a hardcoded analyze action instead of calling the Flock agent
        # Structure matching what the Flock agent would return
        chosen_action = {
            "name": "analyze_restaurant_reviews",
            "arguments": {
                "restaurant_ids": []  # Empty list as we're analyzing all restaurants
            }
        }
        
        return chosen_action # Return the dictionary {"name": ..., "arguments": ...}

    def post(self, shared, prep_res, exec_res):
        """Determine the flow action based on Flock's decision."""
        action_name = exec_res.get("name")
        arguments = exec_res.get("arguments", {})

        logger.info(f"Flock agent decided action: {action_name}")

        # Map function names to flow actions
        if action_name == "analyze_restaurant_reviews":
            # Optionally store the specific IDs Flock wants analyzed if needed later
            # shared["ids_to_analyze"] = arguments.get("restaurant_ids", [])
            return "analyze"
        elif action_name == "ask_user_clarification":
            shared["clarification_question"] = arguments.get("question", "Could you please provide more details?")
            return "clarify"
        # elif action_name == "present_basic_list":
        #     return "list_only"
        else:
            logger.warning(f"Unknown action decided by Flock agent: {action_name}. Defaulting to analyze.")
            # Fallback: If Flock hallucinates a function name, maybe try analyzing anyway?
            return "analyze" # Or maybe an error action 