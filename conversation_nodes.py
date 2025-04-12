"""Conversation nodes for the restaurant recommendation agent."""
import json
import logging
import datetime
import re

from pocketflow import Node
from utils.call_llm import call_llm

logger = logging.getLogger(__name__)


class WelcomeNode(Node):
    """Provides a welcome message to start the conversation."""
    
    def prep(self, shared):
        # Nothing to prepare, just return None
        return None
    
    def exec(self, prep_res):
        # Create a friendly welcome message
        welcome_message = (
            "👋 Hello! I'm your restaurant recommendation assistant. "
            "I can help you find the perfect restaurant based on your "
            "preferences. "
            "\n\nPlease tell me what kind of cuisine you're looking for, "
            "your preferred location, price range, and any dietary "
            "restrictions. "
            "\n\nFor example, you could say: "
            "'I'm looking for an Italian restaurant in Soho, mid-range price, "
            "vegetarian options.'"
        )
        return welcome_message
    
    def post(self, shared, prep_res, exec_res):
        # Store the welcome message as the first system response
        shared["system_response"] = exec_res
        
        # Print the welcome message
        print(f"\nAssistant: {exec_res}")
        
        # Get the first user input
        user_input = input("\nYou: ")
        shared["user_input"] = user_input
        
        # Proceed to the main conversation flow
        return "start_conversation"


class ConversationManagerNode(Node):
    """Node for managing the conversation flow."""

    def __init__(self, max_retries=3):
        """Initialize the conversation manager with retry capability."""
        super().__init__(max_retries=max_retries)
        self.missing_info = []
    
    def prep(self, shared):
        """Prepare context for the conversation manager."""
        # Get user input and current conversation state
        user_input = shared.get("user_input", "")
        conv_state = shared.get("conversation_state", {})
        
        # Initialize conversation state if it doesn't exist
        if not conv_state:
            # Get user's current location right at the beginning
            from utils.google_maps_api import get_user_geolocation
            user_location = get_user_geolocation()
            
            # Store user location for later use
            logger.info(f"===== ATTEMPTING TO GET USER GEOLOCATION AUTOMATICALLY AT CONVERSATION START =====")
            if user_location.get('success'):
                logger.info(f"===== SUCCESSFULLY DETECTED USER LOCATION: {user_location.get('address', 'Unknown')} =====")
                logger.info(f"Automatically detected user's physical location for distance calculations: {user_location.get('address', 'Unknown')}")
                
                # 获取当前时间用作默认值
                current_time = datetime.datetime.now().strftime("%H:%M")
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                conv_state = {
                    "conversation_history": [],
                    "extracted_info": {
                        "location": None,  # Don't set location yet - this is where the user wants restaurants, not their current location
                        "cuisine": None,
                        "budget": None,
                        "occasion": None,
                        "time": current_time,  # 使用当前时间作为默认值
                        "dining_date": current_date,  # 添加就餐日期字段，使用当前日期作为默认值
                        "is_for_now": True,  # 默认假设是立即就餐
                        "priorities": [],
                        "confirmed": False,
                        "group_size": None,
                        "additional_preferences": {},
                        "user_geolocation": user_location,  # Store user's current location separately
                        "current_system_time": {
                            "time": current_time,
                            "date": current_date
                        }
                    },
                    "current_stage": "initialize",
                    "missing_info": ["location", "cuisine", "budget"],  # 不再需要time，因为已有默认值
                    "last_extracted": {}  # Store last successfully extracted info to handle partial info
                }
            else:
                logger.warning("Could not automatically detect user's location. Will use search location only.")
                # 获取当前时间用作默认值（即使无法获取位置）
                current_time = datetime.datetime.now().strftime("%H:%M")
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                conv_state = {
                    "conversation_history": [],
                    "extracted_info": {
                        "location": None,
                        "cuisine": None,
                        "budget": None,
                        "occasion": None,
                        "time": current_time,  # 使用当前时间作为默认值
                        "dining_date": current_date,  # 添加就餐日期字段
                        "is_for_now": True,  # 默认假设是立即就餐
                        "priorities": [],
                        "confirmed": False,
                        "group_size": None,
                        "additional_preferences": {},
                        "current_system_time": {
                            "time": current_time,
                            "date": current_date
                        }
                    },
                    "current_stage": "initialize",
                    "missing_info": ["location", "cuisine", "budget"],  # 不再需要time
                    "last_extracted": {}
                }
            
            # No user input in first turn, just return the state
            if not user_input:
                return conv_state
        
        # Add user message to history if there's input
        if user_input:
            conv_state["conversation_history"].append({
                "role": "user",
                "content": user_input
            })
            
        return user_input, conv_state
        
    def exec(self, prep_result):
        # Handle the first turn (just initialization)
        if isinstance(prep_result, dict) and not isinstance(prep_result, tuple):
            # This is the first turn with just the initialized conversation state
            welcome_message = "Hi there! I'm your restaurant recommendation assistant. I can help you find the perfect place to eat. Just tell me what kind of food you're looking for and where you'd like to search!"
            self.missing_info = prep_result.get("missing_info", [])
            return "initialize", welcome_message, prep_result
            
        # Handle subsequent turns
        try:
            # Unpack the user input and conversation state
            user_input, conv_state = prep_result
            
            # Get the current stage and missing information
            current_stage = conv_state.get("current_stage", "information_gathering")
            self.missing_info = conv_state.get("missing_info", [])
            
            logger.info(f"Current conversation stage: {current_stage}")
            logger.info(f"User input: {user_input}")
            
            # If no user input in this turn, just continue
            if not user_input:
                return current_stage, None, conv_state
                
            # Extract information from user input
            extracted_info = self._extract_entities(user_input, conv_state.get("extracted_info", {}))
            
            # Update the conversation state with extracted information
            if extracted_info:
                # If we don't have an extracted_info dictionary yet, create one
                if "extracted_info" not in conv_state:
                    conv_state["extracted_info"] = {}
                    
                # Update the extracted_info with the new information
                for key, value in extracted_info.items():
                    conv_state["extracted_info"][key] = value
                    
                logger.info(f"Updated extracted info: {conv_state['extracted_info']}")
            
            # Check if we're ready to generate recommendations
            if self._has_sufficient_info(conv_state.get("extracted_info", {})):
                logger.info("Have sufficient information to generate recommendations")
                # Create a structured query from the extracted information
                structured_query = self._generate_structured_query(conv_state.get("extracted_info", {}))
                response = "Great! Let me find some restaurants that match your preferences..."
                return "recommend", response, conv_state
            else:
                # We still need more information
                missing = self.missing_info
                logger.info(f"Still missing information: {missing}")
                response = self._generate_information_request(
                    conv_state.get("extracted_info", {}), 
                    missing
                )
                return "continue_conversation", response, conv_state
        except Exception as e:
            logger.error(f"Error in conversation manager: {str(e)}")
            return "error", f"I'm having trouble processing your request. Let's try again.", conv_state

    def _has_sufficient_info(self, extracted_info):
        """Check if we have enough information to generate recommendations."""
        # At minimum, we need location and either cuisine or some kind of preference
        has_location = extracted_info.get("location") is not None
        has_cuisine = extracted_info.get("cuisine") is not None
        
        return has_location and has_cuisine
        
    def _generate_information_request(self, extracted_info, missing_info):
        """Generate a message asking for specific missing information."""
        if "location" in missing_info:
            return "What area are you looking to dine in?"
        elif "cuisine" in missing_info:
            return "What type of cuisine are you interested in?"
        elif "budget" in missing_info:
            return "What's your budget for this meal?"
        else:
            return "Could you tell me more about what you're looking for in a restaurant?"
            
    def _generate_structured_query(self, extracted_info):
        """Convert extracted information to a structured query for the recommendation system."""
        # This is a simplified version - in a real system, you'd have more sophisticated logic
        query = {
            "location": extracted_info.get("location"),
            "cuisine": extracted_info.get("cuisine"),
            "price_range": extracted_info.get("budget"),
            "time": extracted_info.get("time"),
            "date": extracted_info.get("dining_date"),
            "priorities": extracted_info.get("priorities", []),
            "additional_preferences": extracted_info.get("additional_preferences", {})
        }
        return query

    def _extract_entities(self, user_input, current_info):
        """Extract entities from user input using LLM and regex patterns."""
        import json
        import re
        
        # Initialize extracted values and fallback variables
        extracted_info = {}
        regex_location = None
        regex_cuisine = None
        regex_group_size = None
        regex_time = None
        regex_budget = None
        is_for_now = None
        dining_date = None
        currency_type = "GBP"  # 默认使用英镑作为货币单位
        
        try:
            # 检测货币类型
            currency_patterns = [
                (r'\$', "USD"),  # 美元符号
                (r'£', "GBP"),   # 英镑符号
                (r'€', "EUR"),   # 欧元符号
                (r'\bUSD\b|\bdollars?\b|\bUS\s+dollars?\b', "USD"),   # 美元关键词
                (r'\bGBP\b|\bpounds?\b|\bUK\s+pounds?\b', "GBP"),     # 英镑关键词
                (r'\bEUR\b|\beuros?\b', "EUR")    # 欧元关键词
            ]
            
            for pattern, curr_code in currency_patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    currency_type = curr_code
                    logger.info(f"Detected currency type: {currency_type}")
                    break
            
            # 检测是否是"现在"就餐
            now_patterns = [
                r'\b(?:now|right now|immediately|立刻|现在|马上)\b',
                r'\b(?:tonight|today|this evening|今晚|今天|今天晚上)\b',
                r'\b(?:ASAP|as soon as possible|尽快)\b'
            ]
            
            for pattern in now_patterns:
                if re.search(pattern, user_input.lower()):
                    is_for_now = True
                    logger.info("Detected dining request for current time")
                    break
            
            # 尝试提取具体时间
            # 标准时间格式 HH:MM
            time_pattern = r'(\d{1,2}):(\d{2})(?:\s*(?:am|pm|AM|PM))?'
            # 简单时间格式 X点/X点半/X点Y分
            simple_time_pattern = r'(\d{1,2})\s*(?:点|时|:|：)(?:\s*(\d{1,2}))?(?:\s*(?:分|半))?'
            # 相对时间表达 X小时后
            relative_time_pattern = r'(\d+)\s*(?:小时|hour|hr)[s]?\s*(?:后|later|from now)'
            
            # 先检查标准时间格式
            time_match = re.search(time_pattern, user_input)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                regex_time = f"{hour}:{minute:02d}"
                logger.info(f"Extracted standard time format: {regex_time}")
                is_for_now = False
            
            # 检查简单时间格式
            if not regex_time:
                simple_match = re.search(simple_time_pattern, user_input)
                if simple_match:
                    hour = int(simple_match.group(1))
                    minute = 0
                    if simple_match.group(2):
                        minute = int(simple_match.group(2))
                    elif "半" in user_input[simple_match.start():simple_match.end()+1]:
                        minute = 30
                        
                    regex_time = f"{hour}:{minute:02d}"
                    logger.info(f"Extracted simple time format: {regex_time}")
                    is_for_now = False
            
            # 检查相对时间
            if not regex_time:
                relative_match = re.search(relative_time_pattern, user_input)
                if relative_match:
                    hours_later = int(relative_match.group(1))
                    now = datetime.datetime.now()
                    future_time = now + datetime.timedelta(hours=hours_later)
                    regex_time = future_time.strftime("%H:%M")
                    logger.info(f"Calculated relative time {hours_later} hours from now: {regex_time}")
                    is_for_now = False
                    
            # 尝试提取日期
            date_patterns = [
                # 明天/后天
                r'(明天|tomorrow|后天|day after tomorrow)',
                # 星期几
                r'(周|星期|週|week|星期日|星期天|Sunday|周日|週日|星期一|Monday|周一|週一|星期二|Tuesday|周二|週二|星期三|Wednesday|周三|週三|星期四|Thursday|周四|週四|星期五|Friday|周五|週五|星期六|Saturday|周六|週六)',
                # MM/DD 或 MM-DD
                r'(\d{1,2})[/-](\d{1,2})',
                # MM月DD日
                r'(\d{1,2})月(\d{1,2})日'
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, user_input)
                if date_match:
                    date_text = date_match.group(0)
                    # 这里可以进一步处理日期转换为标准格式
                    # 简单处理示例
                    if "明天" in date_text or "tomorrow" in date_text.lower():
                        tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
                        dining_date = tomorrow.strftime("%Y-%m-%d")
                    elif "后天" in date_text or "day after tomorrow" in date_text.lower():
                        day_after = datetime.datetime.now() + datetime.timedelta(days=2)
                        dining_date = day_after.strftime("%Y-%m-%d")
                    else:
                        # 对于其他日期匹配，至少记录下来
                        dining_date = date_text
                        
                    logger.info(f"Extracted dining date: {dining_date}")
                    break
            
            # 首先使用正则表达式提取关键信息作为备份
            # Location extraction (e.g., "in Soho")
            location_pattern = r'\bin\s+([a-zA-Z\s]+?)(?:[,\.]|$|\s\d)'
            location_match = re.search(location_pattern, user_input)
            if location_match:
                regex_location = location_match.group(1).strip()
                logger.info(f"Extracted location using regex: {regex_location}")
            else:
                # Also check for direct location mentions without "in"
                direct_loc_pattern = r'\b(holborn|soho|covent garden|camden|islington|shoreditch|mayfair|chelsea|kensington|westminster|london)\b'
                direct_match = re.search(direct_loc_pattern, user_input.lower())
                if direct_match:
                    regex_location = direct_match.group(1).capitalize()
                    logger.info(f"Extracted direct location: {regex_location}")
            
            # Cuisine extraction with expanded pattern
            cuisine_pattern = r'(italian|chinese|japanese|mexican|thai|indian|french|vietnamese|korean|turkish|greek|spanish|lebanese|american)\s+(?:cuisine|food|restaurant|place)'
            cuisine_match = re.search(cuisine_pattern, user_input.lower())
            if cuisine_match:
                regex_cuisine = cuisine_match.group(1).capitalize()
                logger.info(f"Extracted cuisine using regex: {regex_cuisine}")
            else:
                # Also try to match just the cuisine name
                simple_cuisine_pattern = r'\b(italian|chinese|japanese|mexican|thai|indian|french|vietnamese|korean|turkish|greek|spanish|lebanese|american)\b'
                simple_match = re.search(simple_cuisine_pattern, user_input.lower())
                if simple_match:
                    regex_cuisine = simple_match.group(1).capitalize()
                    logger.info(f"Extracted cuisine name directly: {regex_cuisine}")
            
            # Group size extraction
            group_pattern = r'(\d+)\s*(?:person|persons|people|pax|guests?|group)'
            group_match = re.search(group_pattern, user_input.lower())
            if group_match:
                regex_group_size = int(group_match.group(1))
                logger.info(f"Extracted group size from regex: {regex_group_size}")
            
            # Budget extraction
            budget_pattern = r'(\d+)\s*(?:dollars|pounds|euros|gbp|usd|eur|\$|£|€)'
            budget_match = re.search(budget_pattern, user_input.lower())
            if budget_match:
                regex_budget = int(budget_match.group(1))
                logger.info(f"Extracted budget from input using pattern '{budget_pattern}': {regex_budget}")
                
                # 同时尝试确定货币类型
                budget_full_match = budget_match.group(0).lower()
                if "$" in budget_full_match or "dollar" in budget_full_match or "usd" in budget_full_match:
                    currency_type = "USD"
                elif "£" in budget_full_match or "pound" in budget_full_match or "gbp" in budget_full_match:
                    currency_type = "GBP"
                elif "€" in budget_full_match or "euro" in budget_full_match or "eur" in budget_full_match:
                    currency_type = "EUR"
                
                logger.info(f"Currency type from budget extraction: {currency_type}")
            
            # 现在调用LLM来提取结构化信息
            try:
                from utils import call_llm
                prompt = f"""
You are an intelligent assistant that extracts structured information from user queries about restaurant recommendations.

USER INPUT: {user_input}

CURRENT TIME INFO: It is currently {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

CURRENT KNOWN INFORMATION:
{json.dumps(current_info, indent=2)}

Extract the following fields:
* location: Where they want to eat (area, neighborhood, city)
* cuisine: Type of cuisine/food they want (e.g., Italian, Chinese, Vietnamese)
* budget: Budget per person (number only)
* occasion: Type of occasion (e.g., date, business, casual)
* time: Time they want to dine (specific time like "19:30" or "8:00PM")
* is_for_now: Boolean - are they looking to dine RIGHT NOW or TODAY? (true/false)
* dining_date: Any specific date mentioned (e.g., "tomorrow", "Saturday")
* priorities: What's most important (taste, service, ambiance, value)
* group_size: Number of people in their party

ADDITIONALLY, extract ANY OTHER preferences or requirements mentioned in the input:
Examples include but are not limited to:
- Noise level preferences (quiet, lively, etc.)
- Outdoor/indoor seating preferences
- Parking requirements
- Kid-friendly needs
- Accessibility requirements
- View/scenery preferences
- Special facilities (private rooms, etc.)
- Entertainment options
- Specific dish requirements
- Reservation policies

Return JSON format:
```json
{{
  "location": "extracted location or null",
  "cuisine": "extracted cuisine or null",
  "budget": extracted number or null,
  "occasion": "extracted occasion or null",
  "time": "extracted time or null",
  "is_for_now": true/false/null,
  "dining_date": "extracted date or null",
  "priorities": ["priority1", "priority2"],
  "group_size": extracted number or null,
  "additional_preferences": {{
    "preference_name": "preference_value",
    "another_preference": "another_value"
  }}
}}
```
"""
                # Call LLM to extract information with a timeout
                response = call_llm.call_llm(prompt)
                
                if response:
                    # Extract JSON from the response
                    try:
                        # Find JSON block in the response
                        start_idx = response.find("{")
                        end_idx = response.rfind("}")
                        if start_idx >= 0 and end_idx >= 0:
                            json_str = response[start_idx:end_idx+1]
                            extracted = json.loads(json_str)
                            
                            # Use regex backups if LLM failed to extract
                            if not extracted.get("location") and regex_location:
                                extracted["location"] = regex_location
                            if not extracted.get("cuisine") and regex_cuisine:
                                extracted["cuisine"] = regex_cuisine
                            if not extracted.get("group_size") and regex_group_size:
                                extracted["group_size"] = regex_group_size
                            if not extracted.get("time") and regex_time:
                                extracted["time"] = regex_time
                            if not extracted.get("budget") and regex_budget:
                                extracted["budget"] = regex_budget
                            if extracted.get("is_for_now") is None and is_for_now is not None:
                                extracted["is_for_now"] = is_for_now
                            if not extracted.get("dining_date") and dining_date:
                                extracted["dining_date"] = dining_date
                            
                            # Ensure additional_preferences exists
                            if not extracted.get("additional_preferences"):
                                extracted["additional_preferences"] = {}
                                
                            # Add currency type if not present
                            if not extracted.get("currency_type"):
                                extracted["currency_type"] = currency_type
                            
                            # Log additional preferences that were captured
                            if extracted.get("additional_preferences") and len(extracted["additional_preferences"]) > 0:
                                logger.info(f"Captured additional preferences: {extracted['additional_preferences']}")
                            
                            # 检查并更新missing_info列表
                            if extracted.get("location") and "location" in self.missing_info:
                                self.missing_info.remove("location")
                                logger.info(f"Removed location from missing_info")
                                
                            if extracted.get("cuisine") and "cuisine" in self.missing_info:
                                self.missing_info.remove("cuisine")
                                logger.info(f"Removed cuisine from missing_info")
                                
                            if extracted.get("budget") and "budget" in self.missing_info:
                                self.missing_info.remove("budget")
                                logger.info(f"Removed budget from missing_info")
                                
                            if (extracted.get("time") or extracted.get("is_for_now") is True) and "time" in self.missing_info:
                                self.missing_info.remove("time")
                                logger.info(f"Removed time from missing_info")
                            
                            return extracted
                        else:
                            logger.warning(f"No JSON found in LLM response: {response}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from LLM: {e}")
                else:
                    logger.warning("Empty response from LLM")
            except Exception as e:
                logger.error(f"Error calling LLM for entity extraction: {e}", exc_info=True)
                
            # 如果LLM提取失败，回退到正则表达式提取结果
            logger.warning("Using regex fallback extraction because LLM extraction failed")
            
            # 处理正则表达式提取的结果
            user_text = user_input.lower()
            
            # Extract cuisine using keyword matching
            cuisines = [
                "chinese", "italian", "indian", "japanese", 
                "thai", "mexican", "french", "american",
                "spanish", "greek", "turkish", "korean",
                "vietnamese", "lebanese", "ethiopian"
            ]
            for cuisine in cuisines:
                if cuisine in user_text:
                    extracted_info["cuisine"] = cuisine.capitalize()
                    logger.info(f"Extracted cuisine: {cuisine}")
                    # Remove from missing info if it was there
                    if "cuisine" in self.missing_info:
                        self.missing_info.remove("cuisine")
                        break
            
            # Extract location using keyword matching
            locations = ["soho", "london", "manhattan", "brooklyn", "queens", "bronx",
                        "chelsea", "tribeca", "harlem", "midtown", "downtown",
                        "central", "north", "south", "east", "west", "holborn"]
            for location in locations:
                if location in user_text:
                    extracted_info["location"] = location.capitalize()
                    logger.info(f"Extracted location: {location}")
                    # Remove from missing info if it was there
                    if "location" in self.missing_info:
                        self.missing_info.remove("location")
                    break
            
            # Extract price range
            if "mid-range" in user_text or "medium" in user_text:
                extracted_info["budget"] = "Mid-range"
                logger.info("Extracted price range: Mid-range")
                # Remove from missing info if it was there
                if "budget" in self.missing_info:
                    self.missing_info.remove("budget")
            elif "cheap" in user_text or "budget" in user_text or "inexpensive" in user_text:
                extracted_info["budget"] = "Budget"
                logger.info("Extracted price range: Budget")
                # Remove from missing info if it was there
                if "budget" in self.missing_info:
                    self.missing_info.remove("budget")
            elif "expensive" in user_text or "high-end" in user_text or "fancy" in user_text:
                extracted_info["budget"] = "Expensive"
                logger.info("Extracted price range: Expensive")
                # Remove from missing info if it was there
                if "budget" in self.missing_info:
                    self.missing_info.remove("budget")
            
            # Use regex budget if available
            if regex_budget:
                budget_category = None
                
                # Convert price to category
                price_num = int(regex_budget)
                if price_num <= 15:
                    budget_category = "Budget"
                elif price_num <= 30:
                    budget_category = "Mid-range"
                else:
                    budget_category = "Expensive"
                
                extracted_info["budget"] = budget_category
                logger.info(f"Using regex extracted budget: {price_num}, category: {budget_category}")
                # Remove from missing info if it was there
                if "budget" in self.missing_info:
                    self.missing_info.remove("budget")
            
            # Add time information
            if regex_time:
                extracted_info["time"] = regex_time
                logger.info(f"Using regex extracted time: {regex_time}")
                if "time" in self.missing_info:
                    self.missing_info.remove("time")
            elif is_for_now:
                extracted_info["is_for_now"] = True
                logger.info("Using detected is_for_now: True")
                if "time" in self.missing_info:
                    self.missing_info.remove("time")
            
            # Add dining date if available
            if dining_date:
                extracted_info["dining_date"] = dining_date
                logger.info(f"Using regex extracted dining date: {dining_date}")
            
            # Add group size if available
            if regex_group_size:
                extracted_info["group_size"] = regex_group_size
                logger.info(f"Using regex extracted group size: {regex_group_size}")
                
                # Extract dietary restrictions
                dietary_keywords = {
                    "vegetarian": "Vegetarian",
                    "vegan": "Vegan",
                    "gluten-free": "Gluten-free",
                    "gluten free": "Gluten-free",
                    "halal": "Halal",
                    "kosher": "Kosher",
                    "dairy-free": "Dairy-free",
                    "dairy free": "Dairy-free",
                    "nut-free": "Nut-free",
                    "nut free": "Nut-free"
                }
                
                dietary_restrictions = []
                for keyword, label in dietary_keywords.items():
                    if keyword in user_text:
                        dietary_restrictions.append(label)
                
                if dietary_restrictions:
                    extracted_info["dietary_preferences"] = dietary_restrictions
                    logger.info(f"Extracted dietary restrictions: {dietary_restrictions}")
            
            # Extract priorities
            priority_keywords = {
                "taste": "taste",
                "delicious": "taste", 
                "flavor": "taste",
                "service": "service",
                "staff": "service",
                "waiters": "service",
                "ambiance": "ambiance",
                "atmosphere": "ambiance",
                "environment": "ambiance",
                "value": "value",
                "affordable": "value",
                "worth": "value"
            }
            
            priorities = []
            for keyword, priority in priority_keywords.items():
                if keyword in user_text and priority not in priorities:
                    priorities.append(priority)
            
            if priorities:
                extracted_info["priorities"] = priorities
                logger.info(f"Extracted priorities: {priorities}")
                
            # Extract Web3 interest
            web3_keywords = ["crypto", "cryptocurrency", "bitcoin", "ethereum", "token", "nft", 
                             "blockchain", "web3", "digital currency", "digital payment"]
            
            web3_interest = any(keyword in user_text for keyword in web3_keywords)
            if web3_interest:
                extracted_info["web3_interest"] = True
                logger.info("Detected interest in Web3 features")
            
            # 添加货币类型信息
            extracted_info["currency_type"] = currency_type
                
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}", exc_info=True)
            
        return extracted_info
        
    def post(self, shared, prep_res, exec_res):
        # Update shared store based on exec_res
        if exec_res is None:
            # No response needed
            return "continue_dialog"
            
        action, response, updated_state = exec_res
        
        # Store the updated conversation state
        if updated_state:
            shared["conversation_state"] = updated_state
        
        # Store the response for display
        if response:
            shared["system_response"] = response
            
            # Also add to conversation history if there is a conversation state
            if "conversation_state" in shared and "conversation_history" in shared["conversation_state"]:
                shared["conversation_state"]["conversation_history"].append({
                    "role": "assistant",
                    "content": response
                })
        
        # Determine next action based on the returned action
        if action == "initialize":
            return "continue_conversation"
        elif action == "continue_conversation":
            return "continue_conversation"
        elif action == "recommend":
            # Format user query for recommendation system
            user_query = shared.get("user_input", "")
            parsed_query = self._generate_structured_query(
                shared["conversation_state"].get("extracted_info", {})
            )
            
            # Store in shared for recommendation flow
            shared["user_query"] = user_query
            shared["parsed_query"] = parsed_query
            
            logger.info(f"Proceeding to recommendations with query: {parsed_query}")
            return "recommend"
        elif action == "error":
            return "error"
        else:
            return "continue_conversation"
            
    def exec_fallback(self, prep_res, exc):
        """Handle execution failures gracefully."""
        logger.error(f"Error in conversation manager: {str(exc)}", exc_info=True)
        
        # Get conversation state if available
        if isinstance(prep_res, tuple) and len(prep_res) == 2:
            _, conv_state = prep_res
        elif isinstance(prep_res, dict):
            conv_state = prep_res
        else:
            conv_state = {}
            
        # Return error action with friendly message
        return "error", "I'm having trouble understanding your request. Could you please rephrase it?", conv_state


class ConversationContinuationNode(Node):
    """Handles the continuation of the conversation after system responses."""
    
    def prep(self, shared):
        # Get the system response to display to the user
        return shared.get("system_response", "")
        
    def exec(self, system_response):
        # Simply pass through the system response
        return system_response
        
    def post(self, shared, prep_res, exec_res):
        # Check if there's an error in the shared store
        if shared.get("error_info"):
            return "error"
            
        # Display the response to the user and get their input
        if exec_res:
            print(f"\n{exec_res}")
            
        user_input = input("\nYou: ")
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("\nThank you for using our service. Goodbye!")
            exit(0)
            
        # Update the shared store with the new user input
        shared["user_input"] = user_input
        
        return "continue_dialog"


class ConversationCompletionNode(Node):
    """Handles the completion of a recommendation flow."""
    
    def prep(self, shared):
        # Get the final response from the recommendation process
        return shared.get("final_response", "")
    
    def exec(self, final_response):
        # Simply pass through the final response
        if not final_response:
            return "I couldn't generate any recommendations. Let's try again with different preferences."
        return final_response
    
    def post(self, shared, prep_res, exec_res):
        # Check if there's an error in the shared store
        if shared.get("error_info"):
            return "error"
            
        # Display the final response to the user
        if exec_res:
            print(f"\n{exec_res}")
        
        # Ask if user needs anything else about these restaurants
        print("\nDo you have any more questions about these restaurants? Or would you like different recommendations?")
        user_input = input("\nYou: ")
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("\nThank you for using our service. Goodbye!")
            exit(0)
            
        # Update the shared store with the new user input
        shared["user_input"] = user_input
        
        # Check if the user wants to start over
        if any(keyword in user_input.lower() for keyword in ["different", "new", "other", "another", "start over"]):
            # Reset recommendation-specific fields
            shared["parsed_query"] = None
            shared["candidate_restaurants"] = None
            shared["reviews_data"] = None
            shared["dimensional_scores_1_10"] = None
            shared["ranked_recommendations"] = None
            shared["final_response"] = None
            
            # Reset conversation state to information gathering
            if shared.get("conversation_state"):
                shared["conversation_state"]["current_stage"] = "information_gathering"
        
        return "continue_dialog"


class DebugNode(Node):
    """Handles error situations and provides debugging information."""
    
    def prep(self, shared):
        # Collect all relevant information from the shared store
        error_info = shared.get("error_info", "Unknown error")
        conversation_state = shared.get("conversation_state", {})
        user_query = shared.get("user_query", "")
        parsed_query = shared.get("parsed_query", {})
        
        debug_info = {
            "error_info": error_info,
            "user_query": user_query,
            "parsed_query": parsed_query,
            "conversation_state": conversation_state
        }
        
        return debug_info
    
    def exec(self, debug_info):
        # Log the debug information
        logger.error("=== DEBUG INFORMATION ===")
        error_info = debug_info.get("error_info", "Unknown error")
        user_query = debug_info.get("user_query", "")
        parsed_query = debug_info.get("parsed_query", {})
        
        logger.error(f"Error: {error_info}")
        logger.error(f"User Query: {user_query}")
        logger.error(f"Parsed Query: {json.dumps(parsed_query, indent=2)}")
        
        # Generate a user-friendly error message
        friendly_message = """I'm sorry, I encountered an error while processing your request. 
This could be due to:
1. A problem connecting to one of our services
2. Difficulty understanding your request
3. An internal system error

Let's try again. Could you please:
1. Check your internet connection
2. Rephrase your request
3. Or just ask something simpler to start with?"""
        
        return friendly_message
            
    def post(self, shared, prep_res, exec_res):
        # Display the error message and prompt for next input
        shared["system_response"] = exec_res
        shared["error_info"] = None  # Clear the error
        
        # Reset certain fields to avoid propagating bad data
        shared["parsed_query"] = None
        shared["candidate_restaurants"] = None
        
        # Get input for the next turn
        print(f"\n{exec_res}")
        user_input = input("\nPlease try again: ")
        shared["user_input"] = user_input
        
        return "continue_dialog" 
