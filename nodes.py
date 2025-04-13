import logging
from pocketflow import Node, BatchNode
import json

# Import utility functions (assuming they are in the utils directory)
from utils import google_maps_api, call_finetuned_analyzer, call_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        parsed_data = call_llm.parse_user_query_llm(user_query)
        
        # If LLM failed completely, use more basic extraction
        if not parsed_data:
            logger.warning("Received empty dictionary from parser. Using fallback extraction methods.")
            parsed_data = self._fallback_extraction(user_query, user_location)
        else:
            # If specific fields are missing, try to extract them with regex
            if not parsed_data.get("location"):
                # Look for "in [location]" pattern
                import re
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
        """Enhanced fallback method to extract information using LLM when primary parsing fails"""
        logger.info("Using LLM fallback extraction for query parsing")
        
        try:
            from utils import call_llm
            
            # Create a comprehensive prompt for the LLM to extract all query details
            prompt = f"""
Extract structured information from this restaurant search query:

USER QUERY: "{user_query}"

USER'S CURRENT LOCATION: {user_location.get('address', 'Unknown')}

Extract each of these aspects if mentioned in the query:
- location: Specific neighborhood or area where the user wants to find restaurants
- cuisine: Type of food or cuisine requested (e.g., Italian, Chinese, Thai)
- budget_pp: Price per person in numeric form only (e.g., 30, 50, 100)
- vibe: Atmosphere or ambiance (e.g., romantic, casual, upscale)
- priorities: Important aspects to the user (Taste, Service, Ambiance, Value, etc.)
- dietary_preferences: Any dietary restrictions mentioned (vegetarian, vegan, gluten-free, etc.)
- group_size: Number of people in the party
- time: Dining time mentioned (in format HH:MM or specifications like "dinner", "lunch")

Return ONLY a valid JSON object with these fields (use null for missing information):
{
  "location": "extracted location or null",
  "cuisine": ["Cuisine1", "Cuisine2"] or "Cuisine" or null,
  "budget_pp": number or null,
  "vibe": "extracted vibe or null",
  "priorities": ["Priority1", "Priority2"] or null,
  "dietary_preferences": ["Preference1", "Preference2"] or null,
  "group_size": number or null,
  "time": "extracted time or null"
}
"""
            # Call LLM to extract the information
            response = call_llm.call_llm(prompt, model="gpt-4o-mini")
            
            if response:
                try:
                    # Extract and parse JSON response
                    start_idx = response.find("{")
                    end_idx = response.rfind("}")
                    
                    if start_idx >= 0 and end_idx >= 0:
                        json_str = response[start_idx:end_idx+1]
                        extracted_data = json.loads(json_str)
                        
                        # Handle potential null values and ensure structure consistency
                        for key in ["location", "cuisine", "budget_pp", "vibe", "priorities", 
                                   "dietary_preferences", "group_size", "time"]:
                            if key not in extracted_data or extracted_data[key] is None:
                                if key in ["priorities", "dietary_preferences"]:
                                    extracted_data[key] = []
                                else:
                                    extracted_data[key] = None
                                    
                        # Ensure location defaults to user's neighborhood if not extracted
                        if not extracted_data["location"] and user_location.get('neighborhood'):
                            extracted_data["location"] = user_location.get('neighborhood')
                            logger.info(f"Using user's neighborhood as location: {extracted_data['location']}")
                            
                        logger.info(f"LLM fallback extraction successful: {extracted_data}")
                        return extracted_data
                        
                except Exception as json_error:
                    logger.error(f"Error parsing LLM fallback response: {json_error}. Response: {response}")
        
        except Exception as e:
            logger.error(f"Error in LLM fallback extraction: {e}", exc_info=True)
            
        # Ultimate fallback with minimal extraction if LLM fails
        logger.warning("LLM fallback failed, using minimal extraction")
        return {
            "location": user_location.get('neighborhood', "Unknown"),
            "cuisine": None,
            "budget_pp": None,
            "vibe": None,
            "priorities": [],
            "dietary_preferences": [],
            "group_size": None,
            "time": None
        }

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
                user_geolocation = google_maps_api.get_user_geolocation()
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
        candidates = google_maps_api.find_restaurants(location, cuisine)
        
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
                
                distance_info = google_maps_api.calculate_distance_to_restaurant(
                    user_coords, restaurant_location
                )
                
                # Add distance info to the candidate
                candidate['distance_info'] = distance_info
        
        # If we have dietary preferences, use LLM to determine if restaurant likely accommodates them
        if dietary_preferences:
            from utils import call_llm
            
            # Gather all relevant information about the restaurant
            restaurant_info = {}
            for key in ["name", "types", "business_status", "formatted_address", "vicinity"]:
                if key in candidate:
                    restaurant_info[key] = candidate[key]
            
            # Include any additional fields that might help determine dietary accommodations
            if "editorial_summary" in candidate:
                restaurant_info["editorial_summary"] = candidate["editorial_summary"]
            
            # Create a prompt for the LLM to analyze if the restaurant accommodates the dietary preferences
            prompt = f"""
Analyze if this restaurant is likely to accommodate these dietary preferences:
{dietary_preferences}

RESTAURANT INFORMATION:
{json.dumps(restaurant_info, indent=2)}

Consider:
1. Restaurant name (may include terms like "vegan", "vegetarian", etc.)
2. Restaurant types/categories
3. Location (some areas have more accommodating restaurants)
4. Any other relevant information in the data

Return ONLY a JSON object with a single boolean field "accommodates":
{{
  "accommodates": true/false,
  "confidence": "high"/"medium"/"low",
  "reasoning": "Brief explanation of your reasoning"
}}
"""
            try:
                response = call_llm.call_llm(prompt, model="gpt-4o-mini")
                
                if response:
                    # Extract and parse the JSON response
                    start_idx = response.find("{")
                    end_idx = response.rfind("}")
                    
                    if start_idx >= 0 and end_idx >= 0:
                        json_str = response[start_idx:end_idx+1]
                        dietary_analysis = json.loads(json_str)
                        
                        if "accommodates" in dietary_analysis:
                            candidate["accommodates_dietary_preferences"] = dietary_analysis["accommodates"]
                            candidate["dietary_confidence"] = dietary_analysis.get("confidence", "medium")
                            candidate["dietary_reasoning"] = dietary_analysis.get("reasoning", "")
                            
                            logger.info(f"LLM dietary analysis for {candidate.get('name')}: {dietary_analysis['accommodates']} ({dietary_analysis.get('confidence', 'medium')})")
                        else:
                            # Default to not accommodating if LLM response doesn't include the field
                            candidate["accommodates_dietary_preferences"] = False
                    else:
                        candidate["accommodates_dietary_preferences"] = False
                else:
                    candidate["accommodates_dietary_preferences"] = False
            except Exception as e:
                logger.error(f"Error in LLM dietary analysis: {e}")
                candidate["accommodates_dietary_preferences"] = False
        
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
                    details = google_maps_api.get_restaurant_details(candidate["id"])
                    if details:
                        # Merge the details with the candidate info
                        candidate.update(details)
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
                reviews = google_maps_api.get_reviews_for_restaurant(resto_id, max_reviews=25)
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


class AnalyzeReviewsBatchNode(BatchNode):
    """Analyzes reviews for each restaurant using the fine-tuned model (Batch processing)."""
    def __init__(self, *args, **kwargs):
        # Set retry parameters for the fine-tuned model calls
        super().__init__(max_retries=2, wait=5, *args, **kwargs)
        self.cur_retry = 0  # Initialize cur_retry attribute
    
    def prep(self, shared):
        reviews_data = shared.get("reviews_data")
        if not reviews_data:
            logger.warning("No reviews data found to analyze. Skipping.")
            return [] # Return empty iterable
            
        # Get user query for contextual dimension extraction
        user_query = shared.get("user_query", "")
        parsed_query = shared.get("parsed_query", {})
        
        # Extract custom dimensions based on user input
        custom_dimensions = self._extract_relevant_dimensions(user_query, parsed_query)
        
        # Store custom dimensions for later use in scoring
        shared["custom_dimensions"] = custom_dimensions
        
        # Prepare items for batch processing: (restaurant_id, list_of_reviews, custom_dimensions)
        items_to_process = []
        for resto_id, reviews in reviews_data.items():
            if reviews: # Only process if there are reviews
                items_to_process.append((resto_id, reviews, custom_dimensions))
            else:
                 logger.info(f"Skipping analysis for {resto_id} due to no reviews.")
        logger.info(f"Prepared {len(items_to_process)} restaurants for review analysis with dimensions: {custom_dimensions}")
        return items_to_process

    def _extract_relevant_dimensions(self, user_query, parsed_query):
        """Extract dimensions that matter to the user based on their query using LLM."""
        # Core dimensions as fallback
        core_dimensions = ["Taste", "Service", "Ambiance"]
        
        # Validate priorities from parsed query
        priorities = parsed_query.get("priorities", [])
        if priorities is None:
            priorities = []
            logger.warning("Found None priorities in parsed_query, defaulting to empty list")
        
        try:
            # Use LLM to extract dimensions directly from user input
            from utils import call_llm
            
            # Construct a prompt that asks the LLM to identify relevant dimensions
            prompt = f"""
Analyze the following user restaurant query to identify which dining aspects (dimensions) the user cares about.

USER QUERY: {user_query}

Core dimensions include:
- Taste: Food quality, flavor, culinary experience
- Service: Staff behavior, attentiveness, professionalism
- Ambiance: Interior atmosphere, decor, vibe, comfort

Additional dimensions could include (but are not limited to):
- Noise: Sound level, quietness, acoustic comfort
- Cleanliness: Hygiene, tidiness, overall cleanliness
- Authenticity: How authentic or traditional the cuisine is
- Portion Size: Food quantity, serving size
- Decor: Interior design, aesthetics, visual appeal
- Accessibility: Wheelchair access, accommodations for disabilities
- Kid-friendly: Suitability for children, family-friendliness
- Parking: Parking availability, convenience
- Vegetarian/Vegan Options: Plant-based menu availability
- Outdoor Seating: Patio, terrace, garden options
- And any other dimension you can identify from the query

Based ONLY on this query, return a JSON array of dimension names that seem relevant to this user.
Include the core dimensions ONLY if they appear to be specifically important to this user.

RETURN ONLY a valid JSON array with dimension names, like:
["Taste", "Ambiance", "Noise"]
"""
            
            # Call LLM to extract dimensions
            response = call_llm.call_llm(prompt, model="gpt-4o-mini")
            
            if response:
                # Parse the response to extract dimensions
                # Handle potential JSON formatting issues
                try:
                    # Extract JSON array from response
                    start_idx = response.find("[")
                    end_idx = response.rfind("]")
                    
                    if start_idx >= 0 and end_idx >= 0:
                        json_str = response[start_idx:end_idx+1]
                        extracted_dimensions = json.loads(json_str)
                        
                        # Validate that we got a list
                        if isinstance(extracted_dimensions, list):
                            # Add any explicitly mentioned priorities that might have been missed
                            for priority in priorities:
                                if priority not in extracted_dimensions:
                                    extracted_dimensions.append(priority)
                                    
                            logger.info(f"LLM extracted dimensions from user query: {extracted_dimensions}")
                            return extracted_dimensions
                except Exception as json_error:
                    logger.error(f"Error parsing LLM dimensions response: {json_error}. Response: {response}")
        
        except Exception as e:
            logger.error(f"Error using LLM to extract dimensions: {e}", exc_info=True)
        
        # Fallback: If LLM extraction failed, use priorities + core dimensions
        fallback_dimensions = core_dimensions.copy()
        for priority in priorities:
            if priority not in fallback_dimensions:
                fallback_dimensions.append(priority)
                
        logger.warning(f"Using fallback dimension extraction: {fallback_dimensions}")
        return fallback_dimensions

    def exec(self, item):
        """Called once per restaurant in the batch."""
        resto_id, reviews, dimensions = item
        logger.info(f"Analyzing reviews for {resto_id} (attempt {self.cur_retry + 1})...")

        # Check if we can use the fine-tuned model (only has Taste, Service, Ambiance)
        finetuned_model_dimensions = ["Taste", "Service", "Ambiance"]
        
        # If the user only needs the three basic dimensions, use the fine-tuned model
        if set(dimensions).issubset(set(finetuned_model_dimensions + ["Value"])):
            try:
                # Import the review analysis functions
                from utils import get_review_scores
                
                logger.info(f"Using fine-tuned model for {resto_id} with requested dimensions: {dimensions}")
                
                # Call the review scores function with the specific dimensions needed
                finetuned_scores = get_review_scores(resto_id, reviews, dimensions)
                
                if finetuned_scores and isinstance(finetuned_scores, dict):
                    # 不再自动计算Value维度
                    logger.info(f"Successfully got scores from fine-tuned model for {resto_id} (10-point scale): {finetuned_scores}")
                    return {"id": resto_id, "scores_1_10": finetuned_scores}
                else:
                    logger.warning(f"Fine-tuned model did not return valid scores for {resto_id}. Falling back to GPT.")
            except Exception as e:
                logger.error(f"Error using fine-tuned model for {resto_id}: {e}. Falling back to GPT.", exc_info=True)
        else:
            logger.info(f"Using GPT for {resto_id} - dimensions {dimensions} require more than fine-tuned model can provide")

        # If we're here, we need to use the general LLM approach (either we need more dimensions or fine-tuned model failed)
        # Construct the prompt for the general LLM
        reviews_list = []
        for r in reviews[:10]:  # Use first 10 reviews for brevity
            if isinstance(r, dict):
                review_text = r.get('text', '')
            elif isinstance(r, str):
                review_text = r
            else:
                review_text = str(r)  # Handle any other type as string
            reviews_list.append(f"- {review_text}")
        
        reviews_text = "\n".join(reviews_list)
        
        # Create a dynamic prompt with the dimensions
        dimensions_prompt = "\n".join([f"- {dim}" for dim in dimensions])
        
        prompt = f"""
Analyze the following user reviews for the restaurant with ID '{resto_id}':
{reviews_text}

Based ONLY on these reviews, provide estimated scores on a scale of 1 to 10 (integers only) for the following dimensions:
{dimensions_prompt}

Output ONLY a valid JSON object containing scores for these dimensions. Example format:
{{
  "Taste": 8,
  "Service": 7,
  "Ambiance": 9,
  "Value": 6
}}

If a dimension cannot be evaluated from the reviews, assign a value of null for that dimension.
"""
        try:
            # Call the general LLM utility (ensure utils.call_llm.call_llm exists and works)
            response_text = call_llm.call_llm(prompt, model="gpt-4o-mini")
            
            # Attempt to parse the JSON response
            try:
                # Check if the response is wrapped in markdown code blocks and extract just the JSON
                if "```json" in response_text and "```" in response_text:
                    # Extract content between markdown json code markers
                    start_idx = response_text.find("```json") + 7
                    end_idx = response_text.find("```", start_idx)
                    if start_idx > 6 and end_idx > start_idx:  # Valid markers found
                        json_str = response_text[start_idx:end_idx].strip()
                        logger.info(f"Extracted JSON from markdown code block for {resto_id}")
                    else:
                        # Fallback to regular JSON search
                        start_idx = response_text.find("{")
                        end_idx = response_text.rfind("}")
                        if start_idx >= 0 and end_idx >= 0:
                            json_str = response_text[start_idx:end_idx+1]
                        else:
                            raise ValueError("Could not locate valid JSON in response")
                else:
                    # Regular JSON extraction
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}")
                    if start_idx >= 0 and end_idx >= 0:
                        json_str = response_text[start_idx:end_idx+1]
                    else:
                        raise ValueError("Could not locate valid JSON in response")
                
                scores_1_10 = json.loads(json_str)
                
                # Basic validation
                if not isinstance(scores_1_10, dict):
                    logger.error(f"LLM response for {resto_id} is not a valid score dictionary: {response_text}")
                    raise ValueError("Invalid format in LLM response")
                    
                # Check for dimension coverage
                missing_dimensions = [dim for dim in dimensions if dim not in scores_1_10]
                if missing_dimensions:
                    logger.warning(f"Missing dimensions in LLM response for {resto_id}: {missing_dimensions}")
                    # Add nulls for missing dimensions
                    for dim in missing_dimensions:
                        scores_1_10[dim] = None
                        
                logger.info(f"Successfully parsed scores for {resto_id} from LLM.")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response from LLM for {resto_id}: {response_text}")
                raise ValueError("LLM did not return valid JSON.")
            except Exception as e:
                logger.error(f"Error parsing LLM response for {resto_id}: {e}")
                raise ValueError(f"Failed to parse LLM response: {e}")
                
            # Return the restaurant ID along with its 1-10 scores
            return {"id": resto_id, "scores_1_10": scores_1_10}
            
        except Exception as e:
            # Catch errors from the LLM call itself or from raising exceptions in the inner try block
            logger.error(f"Error during LLM review analysis for {resto_id}: {e}")
            # Raise an exception to trigger retry or fallback based on Node config
            raise ConnectionError(f"General LLM API failed or returned invalid data for {resto_id}") from e

    def post(self, shared, prep_res, exec_res_list):
        """Collects results from all successful batch executions."""
        dimensional_scores = {}
        successful_analyses = 0
        for result in exec_res_list:
            # Check if the result is valid (not None from fallback, if implemented)
            if result and result.get("scores_1_10"):
                resto_id = result["id"]
                dimensional_scores[resto_id] = result["scores_1_10"]
                successful_analyses += 1
            elif result:
                logger.warning(f"Analysis resulted in no scores for restaurant {result.get('id')}")
            # else: # Handle case where fallback returned None, if needed
                # logger.error("A batch analysis item failed permanently.")

        shared["dimensional_scores_1_10"] = dimensional_scores # Store the 1-10 scores
        logger.info(f"Stored dimensional scores (1-10 scale) for {successful_analyses} restaurants.")
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
        # Store ranked recommendations in shared state
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
            from utils import call_llm
            
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
                            if dim == "Taste":
                                if score >= 7.0:
                                    description = " (Exceptional)"
                                elif score >= 6.0:
                                    description = " (Excellent)"
                                elif score >= 5.0:
                                    description = " (Very Good)"
                                elif score >= 4.0:
                                    description = " (Good)"
                                elif score >= 3.0:
                                    description = " (Average)"
                                else:
                                    description = " (Needs Improvement)"
                            elif dim == "Service":
                                if score >= 7.0:
                                    description = " (Outstanding)"
                                elif score >= 6.0:
                                    description = " (Excellent)"
                                elif score >= 5.0:
                                    description = " (Attentive)"
                                elif score >= 4.0:
                                    description = " (Satisfactory)"
                                elif score >= 3.0:
                                    description = " (Inconsistent)"
                                else:
                                    description = " (Poor)"
                            elif dim == "Ambiance":
                                if score >= 7.0:
                                    description = " (Spectacular)"
                                elif score >= 6.0:
                                    description = " (Great Vibe)"
                                elif score >= 5.0:
                                    description = " (Pleasant)"
                                elif score >= 4.0:
                                    description = " (Acceptable)"
                                elif score >= 3.0:
                                    description = " (Basic)"
                                else:
                                    description = " (Lacking)"
                            elif dim == "Value":
                                if score >= 8.0:
                                    description = " (Excellent)"
                                elif score >= 6.5:
                                    description = " (Worth it)"
                                elif score <= 4.0:
                                    description = " (Overpriced)"
                                
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
                
                # 添加亮点行
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
            dimensions = ["Taste", "Service", "Ambiance"]
            for dim in dimensions:
                dim_scores = []
                for r in recommendations[:3]:
                    score = r.get("scores", {}).get(dim)
                    if score is not None:
                        # 添加描述性文本
                        description = ""
                        if dim == "Taste":
                            if score >= 7.0:
                                description = " (Exceptional)"
                            elif score >= 6.0:
                                description = " (Excellent)"
                            elif score >= 5.0:
                                description = " (Very Good)"
                            elif score >= 4.0:
                                description = " (Good)"
                            elif score >= 3.0:
                                description = " (Average)"
                            else:
                                description = " (Needs Improvement)"
                        elif dim == "Service":
                            if score >= 7.0:
                                description = " (Outstanding)"
                            elif score >= 6.0:
                                description = " (Excellent)"
                            elif score >= 5.0:
                                description = " (Attentive)"
                            elif score >= 4.0:
                                description = " (Satisfactory)"
                            elif score >= 3.0:
                                description = " (Inconsistent)"
                            else:
                                description = " (Poor)"
                        elif dim == "Ambiance":
                            if score >= 7.0:
                                description = " (Spectacular)"
                            elif score >= 6.0:
                                description = " (Great Vibe)"
                            elif score >= 5.0:
                                description = " (Pleasant)"
                            elif score >= 4.0:
                                description = " (Acceptable)"
                            elif score >= 3.0:
                                description = " (Basic)"
                            else:
                                description = " (Lacking)"
                        
                        score_text = f"{dim}: {score:.1f} ⭐{description}"
                    else:
                        score_text = f"{dim}: N/A"
                    dim_scores.append(pad_text(score_text))
                response += f"{dim_scores[0]}  {dim_scores[1]}  {dim_scores[2]}\n"
            
            # 添加亮点行
            highlights = []
            for r in recommendations[:3]:
                if r.get("highlights"):
                    highlight = "✅ " + r["highlights"][0]
                elif r.get("adjustment_reasons"):
                    highlight = "✅ " + r["adjustment_reasons"][0]
                elif "scores" in r:
                    # 根据评分生成特色亮点
                    scores = r.get("scores", {})
                    max_score_dim = None
                    max_score = 0
                    for dim, score in scores.items():
                        if score is not None and score > max_score:
                            max_score = score
                            max_score_dim = dim
                            
                    if max_score_dim == "Taste":
                        highlight = "🔥 Exceptional Taste"
                    elif max_score_dim == "Service":
                        highlight = "👨‍🍳 Outstanding Service"
                    elif max_score_dim == "Ambiance":
                        highlight = "✨ Great Atmosphere"
                    else:
                        highlight = "👍 Solid Overall"
                else:
                    # 随机生成不同的亮点，避免全都是"Good Choice"
                    options = [
                        "👍 Popular Choice", 
                        "⭐ Well Rated", 
                        "🍽️ Good Dining Experience",
                        "🧑‍🤝‍🧑 Customer Favorite"
                    ]
                    import random
                    highlight = random.choice(options)
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
            from utils import call_llm
            
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
        from utils import google_maps_api
        logger.info(f"Making reservation at restaurant {restaurant_id}")
        
        reservation_result = google_maps_api.make_restaurant_reservation(
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
    """Decides the next step based on available data."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prep(self, shared):
        """Prepare context for decision making."""
        user_query = shared.get("user_query", "")
        parsed_query = shared.get("parsed_query", {})
        candidates = shared.get("candidate_restaurants", [])
        reviews_data = shared.get("reviews_data", {})

        # Construct a context string for logging
        context_lines = [f"User Query: {user_query}"]
        context_lines.append(f"Parsed Preferences: {json.dumps(parsed_query)}")
        if candidates:
            context_lines.append(f"Found {len(candidates)} candidate restaurants: {[c.get('name') for c in candidates]}")
            reviews_available_count = sum(1 for c in candidates if reviews_data.get(c.get('id')))
            context_lines.append(f"Reviews available for {reviews_available_count} of them.")
        else:
            context_lines.append("No restaurant candidates found yet.")

        context = "\n".join(context_lines)
        logger.debug(f"Decision context:\n{context}")
        return context

    def exec(self, context):
        """Decide the next action based on the context."""
        logger.info("Defaulting to analyze action...")
        
        return "analyze"

    def post(self, shared, prep_res, exec_res):
        """Determine the flow action based on the decision."""
        logger.info(f"Decided action: {exec_res}")
        return exec_res  # Just return the action directly 