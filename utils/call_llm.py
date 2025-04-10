# Placeholder for general LLM calls (e.g., OpenAI, Claude)
import logging
import time
import random
import os
import ast # Use ast.literal_eval instead of eval for safety
import json # Import the json module
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv # Import dotenv
# import yaml # If parsing YAML output

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 确保定义logger对象
logger = logging.getLogger(__name__)

# --- OpenAI Configuration ---
# Load API key securely from environment variables
# Ensure the OPENAI_API_KEY environment variable is set in your .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Set a default model that can be overridden
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini") # Use environment variable or default
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not found in environment variables. OpenAI calls will fail.")
    client = None
else:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        client = None

def call_llm(prompt: str, model: str = None, max_retries: int = 3, delay: int = 5) -> str | None:
    """
    Calls a general LLM API (e.g., OpenAI) with retries for rate limits.

    Args:
        prompt: The input prompt for the LLM.
        model: Optional model override (defaults to OPENAI_MODEL if not specified).
        max_retries: Maximum number of retry attempts.
        delay: Seconds to wait between retries.

    Returns:
        The LLM response content as a string, or None if it fails after retries.
    """
    if not client:
        logging.error("OpenAI client not initialized. Cannot make LLM calls.")
        # Optionally return a simulated response for testing without keys
        # logging.warning("Returning simulated LLM response due to missing client.")
        # return f"Simulated response to: {prompt[:50]}..."
        return None

    # Use provided model or fall back to default
    model_to_use = model or OPENAI_MODEL
    logging.debug(f"Calling LLM ({model_to_use}) with prompt: {prompt[:100]}...")
    attempt = 0
    while attempt < max_retries:
        try:
            # --- Real OpenAI Implementation ---
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, # Adjust temperature as needed
            )
            content = response.choices[0].message.content
            logging.debug(f"LLM Response received: {content[:100]}...")
            return content
            # --- End Real OpenAI Implementation ---

        except RateLimitError as e:
            attempt += 1
            logging.warning(f"Rate limit error calling LLM (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
            if attempt >= max_retries:
                logging.error("Max retries exceeded for rate limit error.")
                return None
            time.sleep(delay)
        except APIError as e:
            # Handle other API errors (e.g., server errors, invalid requests)
            attempt += 1
            logging.warning(f"API error calling LLM (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s...")
            if attempt >= max_retries:
                logging.error("Max retries exceeded for API error.")
                return None
            time.sleep(delay)
        except Exception as e:
            # Handle unexpected errors (network issues, etc.)
            logging.error(f"Unexpected error calling LLM: {e}", exc_info=True)
            return None # Fail immediately on unexpected errors

    return None # Should technically be unreachable if max_retries > 0

# --- Optional: Functions for specific LLM tasks ---

def parse_user_query_llm(user_query: str) -> dict | None:
    """Calls LLM specifically to parse the user query into structured data."""
    prompt = f"""
Parse the following user query about restaurant recommendations and extract structured information.

USER QUERY: {user_query}

Extract the following fields CAREFULLY:
* location: Where they want to eat (neighborhood, area, city) - be sure to recognize "in [location]" patterns
* cuisine: What type of food they want (Chinese, Italian, Vietnamese, etc.) - recognize "[cuisine] cuisine" patterns
* budget_pp: Budget per person in numeric form (if mentioned)
* vibe: Atmosphere they're looking for (casual, fancy, romantic, etc.)
* priorities: What's most important to them (taste, service, ambiance, value)
* dietary_preferences: Any dietary requirements (vegetarian, gluten-free, etc.)
* group_size: Number of people (if mentioned) - look for patterns like "2 persons" or "for 2 people"
* time: Any time mentioned (if any) - look for patterns like "2pm" or "at 7:30"

Include ALL information mentioned in the query, even if spread across different parts.
For budget, extract numeric value and be aware of different currency symbols (£, $, €, etc.).
For cuisine, be especially attentive to phrases like "Vietnamese cuisine" or "want to eat [cuisine]".
For group size, look specifically for number + "people/persons/pax" patterns.

Output a JSON object with these fields. Example:
{{
  "location": "Soho",
  "cuisine": ["Vietnamese"],
  "budget_pp": 50,
  "vibe": "romantic",
  "priorities": ["taste", "ambiance"],
  "dietary_preferences": ["vegetarian"],
  "group_size": 2,
  "time": "19:30"
}}

Return ONLY the JSON object with no additional text.
"""
    logging.info(f"Parsing user query: {user_query}")

    # Also use regex to extract some key information as backup
    import re
    
    # Try to extract cuisine with regex as backup
    cuisine_pattern = r'(italian|chinese|japanese|mexican|thai|indian|french|vietnamese|korean|turkish|greek|spanish|lebanese|american)\s+(?:cuisine|food|restaurant)'
    cuisine_match = re.search(cuisine_pattern, user_query.lower())
    cuisine_backup = None
    if cuisine_match:
        cuisine_backup = cuisine_match.group(1).capitalize()
        logging.info(f"Extracted cuisine from regex: {cuisine_backup}")
    
    # Try to extract group size with regex as backup
    group_pattern = r'(\d+)\s*(?:person|persons|people|pax|guests|group)'
    group_match = re.search(group_pattern, user_query.lower())
    group_size_backup = None
    if group_match:
        group_size_backup = int(group_match.group(1))
        logging.info(f"Extracted group size from regex: {group_size_backup}")
    
    # Try to extract budget with regex as backup
    budget_pattern = r'(\d+)\s*(?:dollars|pounds|euros|gbp|usd|eur|\$|£|€)'
    budget_match = re.search(budget_pattern, user_query.lower())
    budget_backup = None
    if budget_match:
        budget_backup = int(budget_match.group(1))
        logging.info(f"Extracted budget from input using pattern '{budget_pattern}': {budget_backup}")

    response_str = call_llm(prompt)
    if not response_str:
        return None
    try:
        # --- Robust Parsing Logic ---
        # Find the start and end of the dictionary literal within the response
        start_index = response_str.find('{')
        end_index = response_str.rfind('}')

        if start_index == -1 or end_index == -1 or end_index < start_index:
            logging.error(f"Could not find dictionary literal {{...}} in LLM response: {response_str}")
            return None

        # Extract the potential dictionary string
        dict_str = response_str[start_index : end_index + 1]
        # --- End Robust Parsing Logic ---

        # Use json.loads() to parse the extracted string, which handles null/true/false
        parsed_data = json.loads(dict_str)
        if isinstance(parsed_data, dict):
            # Process the budget if present
            if "budget_pp" in parsed_data and parsed_data["budget_pp"]:
                # If budget is a string with currency symbols, convert to number
                if isinstance(parsed_data["budget_pp"], str):
                    # Remove currency symbols and convert to float
                    budget_str = parsed_data["budget_pp"]
                    # Handle currency symbols
                    for symbol in ['£', '$', '€', '¥', '₹', 'kr', 'руб']:
                        budget_str = budget_str.replace(symbol, '')
                    # Clean up any remaining non-numeric characters except decimal point
                    budget_str = ''.join(c for c in budget_str if c.isdigit() or c == '.')
                    try:
                        parsed_data["budget_pp"] = float(budget_str)
                    except ValueError:
                        logging.warning(f"Could not convert budget string to float: {budget_str}")
            
            # Use regex backup values if LLM failed to extract these fields
            if not parsed_data.get("cuisine") and cuisine_backup:
                parsed_data["cuisine"] = [cuisine_backup]
                
            if not parsed_data.get("group_size") and group_size_backup:
                parsed_data["group_size"] = group_size_backup
                
            if not parsed_data.get("budget_pp") and budget_backup:
                parsed_data["budget_pp"] = budget_backup
            
            logging.info(f"Successfully parsed query: {parsed_data}")
            return parsed_data
        else:
            logging.error(f"LLM did not return a valid dictionary structure after JSON parsing: {response_str}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing LLM response as JSON: {e}\nResponse: {response_str}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing LLM response for query: {e}\nResponse: {response_str}", exc_info=True)
        return None

def generate_final_response_llm(user_query, ranked_recommendations, user_location="Unknown location"):
    """
    Generate a final response markdown that presents the recommendations.
    
    Args:
        user_query: The original user query
        ranked_recommendations: List of restaurant dictionaries with scores
        user_location: User's current location
        
    Returns:
        Markdown formatted response with restaurant recommendations
    """
    try:
        logger.info("Generating final response with LLM...")
        
        # Extract dietary preferences and group size from query if available
        import json
        query_data = {}
        try:
            # Try to parse if it's a JSON string
            if isinstance(user_query, str) and user_query.strip().startswith('{'):
                query_data = json.loads(user_query)
            # For demo purposes, also accept a dictionary directly
            elif isinstance(user_query, dict):
                query_data = user_query
        except:
            # If parsing fails, use the original query as is
            query_str = user_query if isinstance(user_query, str) else str(user_query)
        
        dietary_prefs = query_data.get('dietary_preferences', []) if isinstance(query_data, dict) else []
        group_size = query_data.get('group_size') if isinstance(query_data, dict) else None
        reservation_time = query_data.get('time') if isinstance(query_data, dict) else None
        additional_prefs = query_data.get('additional_preferences', {}) if isinstance(query_data, dict) else {}
        
        query_str = user_query if isinstance(user_query, str) else json.dumps(user_query)
        
        # Prepare the LLM prompt
        prompt = f"""
You are Tate, a gourmet restaurant recommendation assistant. Generate a restaurant recommendation based on the user's query.

USER QUERY: {query_str}

USER'S CURRENT LOCATION: {user_location}

USER'S DIETARY PREFERENCES: {', '.join(dietary_prefs) if dietary_prefs else 'None specified'}

GROUP SIZE: {group_size if group_size else 'Not specified'}

RESERVATION TIME: {reservation_time if reservation_time else 'Not specified'}

ADDITIONAL PREFERENCES: {json.dumps(additional_prefs, indent=2) if additional_prefs else 'None specified'}

TOP RECOMMENDED RESTAURANTS (already ranked):
"""
        
        # Add details for each restaurant
        from datetime import datetime
        current_time = datetime.now()
        
        for i, resto in enumerate(ranked_recommendations[:5]):  # Limit to top 5
            name = resto.get("name", "Unknown Restaurant")
            address = resto.get("formatted_address", resto.get("vicinity", "Address unavailable"))
            rating = resto.get("rating", "No rating")
            price_level = "".join(["$"] * (resto.get("price_level", 2)))
            fit_score = resto.get("fit_score", 0)
            base_fit_score = resto.get("base_fit_score", 0)
            preference_adjustments = resto.get("preference_adjustments", 0)
            adjustment_reasons = resto.get("adjustment_reasons", [])
            review_count = resto.get("review_count", 0) or resto.get("user_ratings_total", 0)
            
            # Get the dimensions used in scoring
            dimensions_used = resto.get("dimensions_used", [])
            dimension_scores = resto.get("scores", {})
            user_priorities = resto.get("user_priorities", [])
            
            # Format dimension scores
            score_details = []
            for dim, score in dimension_scores.items():
                if score is not None:  # Only include dimensions with valid scores
                    priority_indicator = "★" if dim in user_priorities else ""  # Add star for user priorities
                    score_details.append(f"{dim}{priority_indicator}: {score}/10")
            dimension_text = ", ".join(score_details)
            
            # Opening hours information
            hours_info = ""
            if "opening_hours" in resto and resto["opening_hours"].get("periods"):
                # Find the current day's opening hours
                today = current_time.weekday()
                close_time = None
                for period in resto["opening_hours"]["periods"]:
                    if period.get("open", {}).get("day") == today:
                        close_time = period.get("close", {}).get("time")
                        if close_time:
                            close_hour = int(close_time[:2])
                            close_min = int(close_time[2:])
                            current_hour = current_time.hour
                            current_min = current_time.minute
                            
                            # Calculate minutes until closing
                            minutes_until_close = (close_hour - current_hour) * 60 + (close_min - current_min)
                            
                            if minutes_until_close <= 60 and minutes_until_close > 0:
                                hours_info = f"⚠️ CLOSING SOON: Only {minutes_until_close} minutes until closing!"
                            elif minutes_until_close <= 0:
                                hours_info = "⚠️ MAY BE CLOSED NOW based on listed hours"
                            else:
                                formatted_close = f"{close_hour}:{close_min:02d}"
                                hours_info = f"Open until {formatted_close}"
                        break
            
            # Dietary information
            dietary_info = ""
            if dietary_prefs and resto.get("accommodates_dietary_preferences") is not None:
                if resto.get("accommodates_dietary_preferences"):
                    dietary_info = f"✅ Likely accommodates {', '.join(dietary_prefs)}"
                else:
                    dietary_info = f"⚠️ May not accommodate {', '.join(dietary_prefs)}"
            
            # Distance information
            distance_info = ""
            if "distance_info" in resto and resto["distance_info"].get("distance_km"):
                dist = resto["distance_info"]
                distance_info = f"📍 {dist.get('distance_km')} km from your current location ({dist.get('travel_time_driving')} min driving, {dist.get('travel_time_transit')} min transit)"
            
            # Preference adjustment reasons
            adjustment_text = ""
            if adjustment_reasons:
                adjustment_text = "\n   - Highlights: " + ", ".join(adjustment_reasons)
            
            phone = resto.get("formatted_phone_number", "Phone unavailable")
            website = resto.get("website", "Website unavailable")
            
            prompt += f"""
{i+1}. **{name}** (Fit Score: {fit_score:.1f}/10)
   - Address: {address}
   - Rating: {rating}/5
   - Price: {price_level}
   - Hours: {hours_info}
   - Phone: {phone}
   - {dietary_info if dietary_info else ""}
   - {distance_info if distance_info else ""}
   - Scores: {dimension_text}
   - Based on analysis of {review_count} reviews{adjustment_text}
"""

        prompt += f"""
FORMAT YOUR RESPONSE:
Create a visual comparison of restaurants in this EXACT format:

"Okay [name from query], I've analyzed recent feedback for top restaurants in [location] based on your preferences for [occasion]: [cuisine type], [key priorities], [vibe type], [budget range]. Here's a structured recommendation to simplify your decision:

### [Location] Restaurant Analysis (Based on Your Specific Needs):

| [Restaurant 1 Name] | [Restaurant 2 Name] | [Restaurant 3 Name] |
|---------------------|---------------------|---------------------|
| **Fit: 8.8/10**     | **Fit: 8.2/10**     | **Fit: 7.5/10**     |
| Taste: 4.5 ⭐       | Taste: 4.4 ⭐       | Taste: 4.9 ⭐       |
| Service: 4.6 ⭐     | Service: 4.3 ⭐     | Service: 4.2 ⭐     |
| Ambiance: 4.5 ⭐ (Good Chat Vibe) | Ambiance: 4.3 ⭐ (Lively/Buzzy) | Ambiance: 4.0 ⭐ (Unique/Busy) |
| Value: ~Ok (Higher end) | Value: Good     | Value: Good         |
| ✅ Service, ✅ Convo Vibe; ⚠️ Top of Budget | ✅ Lively Vibe, ✅ Good All-Round | 🔥 Top Taste!; Setting unique, queues possible |

### Quick Summary:

**[Restaurant 1]**: Strongest match, especially for service & chat-friendly ambiance. Price is the main consideration.
**[Restaurant 2]**: Good lively option with solid scores. A great fit if the group enjoys the vibe & menu.
**[Restaurant 3]**: Taste is exceptional, but the unique setting might compromise the 'chatty vibe'. Good value.

Based on this, **[Restaurant 1]** seems the closest match, with **[Restaurant 2]** as a strong alternative. What do you think?"

Follow this structure PRECISELY with the actual restaurant data. The middle section MUST be formatted as a Markdown table exactly as shown, with proper column alignment and spacing. Use bold for restaurant names and fit scores. Use the exact emoji indicators (⭐, ✅, ⚠️, 🔥) as shown.

If you can't create a perfect table, at minimum include:
1. Clearly labeled restaurant sections with proper fit scores
2. Star ratings for each dimension with proper emojis
3. The Quick Summary section with the key points for each restaurant
4. A final recommendation comparing top choices
"""

        response_text = call_llm(
            prompt,
            model="gpt-4o-mini",  # 使用mini模型以控制成本
            max_retries=2
        )
        
        logger.info("Successfully generated final response")
        
        # 进行后处理，确保格式正确
        if response_text:
            # 确保响应中包含关键格式元素
            if "Restaurant Analysis" not in response_text:
                logger.warning("Response missing key formatting elements, attempting to fix")
                # 尝试修复格式问题
                enhance_prompt = f"""
The following restaurant recommendation needs to be reformatted to match our required template with proper Markdown.
Please restructure it to match this exact format:

"Okay [name], I've analyzed recent feedback for top restaurants in [location] based on your preferences for [occasion]: [cuisine type], [key priorities], [vibe type], [budget range]. Here's a structured recommendation to simplify your decision:

[Location] Restaurant Analysis (Based on Your Specific Needs):

[Restaurant 1 Name]          [Restaurant 2 Name]          [Restaurant 3 Name]
Fit: [8.8]/10                Fit: [8.2]/10                Fit: [7.5]/10

Taste: [4.5] ⭐              Taste: [4.4] ⭐              Taste: [4.9] ⭐
Service: [4.6] ⭐            Service: [4.3] ⭐            Service: [4.2] ⭐
Ambiance: [4.5] ⭐ (Good)    Ambiance: [4.3] ⭐ (Lively)  Ambiance: [4.0] ⭐ (Unique)
Value: ~Ok (Higher end)      Value: Good                  Value: Good

✅ [Priority1], ✅ [Priority2]  ✅ [Priority1], ✅ [Priority2]  🔥 [Exceptional feature]
⚠️ [Warning if applicable]     ⚠️ [Warning if applicable]     ⚠️ [Warning if applicable]

Quick Summary:

[Restaurant 1]: Strongest match, especially for [key strength]. [Key concern] is the main consideration.
[Restaurant 2]: Good [attribute] option with solid scores. A great fit if the group enjoys the [attribute] & menu.
[Restaurant 3]: [Attribute] is exceptional, but the [attribute] might compromise the '[desired attribute]'. Good value.

Based on this, [Restaurant 1] seems the closest match, with [Restaurant 2] as a strong alternative. What do you think?"

Original content to reformat:
{response_text}

Keep all the restaurant information, scores and details, but ONLY reformat to match the template exactly.
"""
                enhanced_response = call_llm(enhance_prompt, model="gpt-4o") 
                if enhanced_response:
                    response_text = enhanced_response
            
        return response_text
        
    except Exception as e:
        logger.error(f"Error generating final response: {str(e)}")
        return None

if __name__ == '__main__':
    # Example usage for testing this utility directly
    # Ensure OPENAI_API_KEY is set in your .env file and loaded

    if not client:
        print("OpenAI client not initialized (API key missing?). Skipping tests.")
    else:
        test_query = "Find me a great Italian place in Soho for a date night, budget around £60pp, focus on ambiance and good service."
        print("Testing query parsing...")
        parsed = parse_user_query_llm(test_query)
        print(f"Parsed Query: {parsed}")

        print("\nTesting final response generation...")
        # Dummy data for testing
        dummy_recs = [
            {"id": "g1", "name": "Pasta Palace", "scores": {"Taste": 8.8, "Service": 9.2, "Ambiance": 9.5, "Value": 7.0}, "fit_score": 9.3},
            {"id": "g2", "name": "Vino Venue", "scores": {"Taste": 8.5, "Service": 8.8, "Ambiance": 8.0, "Value": 7.5}, "fit_score": 8.1},
        ]
        final_md = generate_final_response_llm(test_query, dummy_recs)
        if final_md:
            print("\n--- Generated Response ---")
            print(final_md)
            print("--- End Generated Response ---")
        else:
            print("\nFailed to generate final response.") 