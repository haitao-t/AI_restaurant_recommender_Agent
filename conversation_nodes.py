import logging
import json
from pocketflow import Node, Flow
from utils import call_llm
import datetime  # 添加datetime模块导入

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationManagerNode(Node):
    """Manages multi-turn conversation and extracts relevant information."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(max_retries=2, wait=3, *args, **kwargs)
        
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
                        "time": None,
                        "dining_date": None,  # 添加就餐日期字段
                        "is_for_now": None,  # 是否是立即就餐
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
                    "missing_info": ["location", "cuisine", "budget", "time"],  # 将时间加入必要信息
                    "last_extracted": {}  # Store last successfully extracted info to handle partial info
                }
            else:
                logger.warning("Could not automatically detect user's location. Will use search location only.")
            conv_state = {
                "conversation_history": [],
                "extracted_info": {
                    "location": None,
                    "cuisine": None,
                    "budget": None,
                    "occasion": None,
                    "time": None,
                    "dining_date": None,
                    "is_for_now": None,
                    "priorities": [],
                    "confirmed": False,
                    "group_size": None,
                    "additional_preferences": {}
                },
                "current_stage": "initialize",
                "missing_info": ["location", "cuisine", "budget", "time"],
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
    
    def exec(self, prep_res):
        """Process the conversation and extract information."""
        # Handle first turn (just initialization)
        if isinstance(prep_res, dict):
            # This is the first turn, just initialize
            welcome_message = "Hi there! I'm your restaurant recommendation assistant. I can help you find the perfect place to eat. Just tell me what area you'd like to search in, what cuisine you're interested in, and your budget. I'll automatically calculate distances from your current location to help you decide!"
            self.missing_info = prep_res.get("missing_info", []) # Initialize missing_info
            return "initialize", welcome_message, prep_res
            
        # Normal processing for subsequent turns
        user_input, conv_state = prep_res
        current_stage = conv_state.get("current_stage", "information_gathering")
        
        # Update missing_info
        self.missing_info = conv_state.get("missing_info", [])
        
        # 记录当前状态，用于调试
        logger.info(f"Current conversation stage: {current_stage}")
        logger.info(f"User input: {user_input}")
        
        # 添加日志记录当前确认状态
        is_confirmed = conv_state["extracted_info"].get("confirmed", False)
        logger.info(f"Current confirmation status: {is_confirmed}")
        
        # 特殊处理：如果用户已经确认，且当前在确认阶段，直接进入推荐
        if is_confirmed and current_stage == "confirmation":
            logger.info("User already confirmed, proceeding directly to recommendations")
            next_stage = "generate_recommendations"
            response = "Great! Let me find some restaurants that match your preferences..."
            return next_stage, response, conv_state
        
        # 对于精确的"yes"输入，如果在确认阶段，直接将confirmed设为True并进入推荐
        if current_stage == "confirmation" and user_input.lower().strip() == "yes":
            logger.info("User entered explicit 'yes' in confirmation stage, proceeding directly to recommendations")
            conv_state["extracted_info"]["confirmed"] = True
            next_stage = "generate_recommendations"
            response = "Great! Let me find some restaurants that match your preferences..."
            return next_stage, response, conv_state
        
        # Extract information from user input
        if current_stage in ["initialize", "information_gathering", "clarification", "confirmation"]:
            # 即使在确认阶段，也尝试提取实体信息
            # 允许用户在确认阶段提供额外信息或修改信息
            extracted_info = self._extract_entities(user_input, conv_state["extracted_info"])
            
            # Save this extraction in case it was partial
            conv_state["last_extracted"] = extracted_info
            
            # First, check if there are preferences to remove
            preferences_to_remove = extracted_info.get("preferences_to_remove", [])
            if preferences_to_remove:
                logger.info(f"Processing preferences to remove: {preferences_to_remove}")
                
                # Process each preference to remove
                for pref in preferences_to_remove:
                    if pref == "noise_level" and "additional_preferences" in conv_state["extracted_info"]:
                        if "noise_level" in conv_state["extracted_info"]["additional_preferences"]:
                            # Remove the noise level preference
                            old_value = conv_state["extracted_info"]["additional_preferences"].pop("noise_level")
                            logger.info(f"Removed noise level preference: {old_value}")
                            
                    elif pref == "seating" and "additional_preferences" in conv_state["extracted_info"]:
                        if "seating" in conv_state["extracted_info"]["additional_preferences"]:
                            # Remove the seating preference
                            old_value = conv_state["extracted_info"]["additional_preferences"].pop("seating")
                            logger.info(f"Removed seating preference: {old_value}")
                            
                    elif pref == "cuisine":
                        # Reset cuisine to None
                        old_cuisine = conv_state["extracted_info"].get("cuisine")
                        if old_cuisine:
                            logger.info(f"Removing cuisine preference: {old_cuisine}")
                            conv_state["extracted_info"]["cuisine"] = None
                            # Add cuisine back to missing info if it was removed
                            if "cuisine" not in self.missing_info:
                                self.missing_info.append("cuisine")
                                
                    elif pref == "priorities" and extracted_info.get("priorities") and isinstance(extracted_info["priorities"], list):
                        # Remove specific priorities
                        for priority in extracted_info["priorities"]:
                            if priority in conv_state["extracted_info"].get("priorities", []):
                                conv_state["extracted_info"]["priorities"].remove(priority)
                                logger.info(f"Removed priority: {priority}")
                                
                    elif "additional_preferences" in conv_state["extracted_info"] and pref in conv_state["extracted_info"]["additional_preferences"]:
                        # Handle any other preferences in the additional_preferences dictionary
                        old_value = conv_state["extracted_info"]["additional_preferences"].pop(pref)
                        logger.info(f"Removed additional preference {pref}: {old_value}")
                        
                # After removing preferences, inform the user
                preference_removal_acknowledgment = True
            else:
                preference_removal_acknowledgment = False
            
            # Now, update the conversation state with new information (after handling removals)
            info_updated = False
            for key, value in extracted_info.items():
                # Skip the preferences_to_remove field
                if key == "preferences_to_remove":
                    continue
                    
                if value is not None and value != []:  # Only update if we have a non-empty value
                    # Special handling for budget to avoid overwriting if the user doesn't provide a new value
                    if key == "budget" and value == 0 and conv_state["extracted_info"].get("budget"):
                        continue
                        
                    # Special handling for additional_preferences - merge rather than replace
                    if key == "additional_preferences" and isinstance(value, dict) and value:
                        # Initialize if not exists
                        if "additional_preferences" not in conv_state["extracted_info"]:
                            conv_state["extracted_info"]["additional_preferences"] = {}
                            
                        # Merge dictionaries, update existing preferences
                        for pref_key, pref_value in value.items():
                            if pref_value:  # Only update if non-empty
                                old_value = conv_state["extracted_info"]["additional_preferences"].get(pref_key)
                                conv_state["extracted_info"]["additional_preferences"][pref_key] = pref_value
                                if old_value != pref_value:
                                    info_updated = True
                                    logger.info(f"Updated additional preference {pref_key}: {old_value} -> {pref_value}")
                    else:
                        # Track if we've updated any information
                        if conv_state["extracted_info"].get(key) != value:
                            info_updated = True
                            conv_state["extracted_info"][key] = value
                        
                        # Remove from missing info if it was added
                        if key in conv_state["missing_info"]:
                            conv_state["missing_info"].remove(key)
            
            # Log all extracted information after updates
            if info_updated:
                logger.info(f"Updated extracted info: {conv_state['extracted_info']}")
            
            # Check if we have enough information to generate recommendations
            if self._has_sufficient_info(conv_state["extracted_info"]):
                if conv_state["extracted_info"]["confirmed"]:
                    next_stage = "generate_recommendations"
                    response = "Great! Let me find some restaurants that match your preferences..."
                else:
                    next_stage = "confirmation"
                    response = self._generate_confirmation_prompt(conv_state["extracted_info"])
            else:
                next_stage = "information_gathering"
                
                # If we just acknowledged preference removal, add that to the response
                if preference_removal_acknowledgment:
                    # Generate response with acknowledge of preference removal
                    remove_prompt = f"""
The user has just asked to remove some preferences from their restaurant search criteria. 
Current information state: {json.dumps(conv_state['extracted_info'], indent=2)}
Some preferences were just removed: {json.dumps(preferences_to_remove, indent=2)}

Generate a brief, friendly response that:
1. Acknowledges the preferences that were removed
2. Confirms their updated preferences
3. Asks for the missing information: {self.missing_info if self.missing_info else "nothing, all necessary info provided"}

Keep it very conversational and brief (1-2 sentences).
"""
                    try:
                        acknowledge_response = call_llm.call_llm(remove_prompt)
                        if acknowledge_response:
                            response = acknowledge_response
                        else:
                            # Fallback message if LLM call fails
                            response = f"I've updated your preferences. Is there anything specific you're looking for now?"
                    except Exception as e:
                        logger.error(f"Error generating preference removal acknowledgment: {e}")
                        response = f"I've updated your preferences. Is there anything specific you're looking for now?"
                else:
                    # Regular information request
                    response = self._generate_information_request(conv_state["extracted_info"], 
                                                                conv_state["missing_info"])
                
            # If this is the very first user input and contains multiple pieces of info, 
            # go directly to confirmation instead of asking more questions
            is_first_input = len(conv_state["conversation_history"]) <= 2  # Initial greeting + one user response
            has_multiple_fields = sum(1 for key, val in extracted_info.items() 
                                    if val and key in ["location", "cuisine", "budget", "group_size"] 
                                    and key != "additional_preferences") >= 2
                                    
            if is_first_input and has_multiple_fields and next_stage == "information_gathering":
                logger.info("First user input contains multiple information fields, going to confirmation directly")
                next_stage = "confirmation"
                response = self._generate_confirmation_prompt(conv_state["extracted_info"])
        
        # Handle confirmation stage
        elif current_stage == "confirmation":
            # 首先检查是否是简单的"yes"确认
            if user_input.lower().strip() == "yes":
                logger.info("Explicit 'yes' confirmation detected, setting confirmed=True")
                conv_state["extracted_info"]["confirmed"] = True
                next_stage = "generate_recommendations"
                response = "Great! Let me find some restaurants that match your preferences..."
                logger.info(f"Setting next_stage to {next_stage} after yes confirmation")
            else:
                # 不再单独处理确认阶段的逻辑，因为已经在前面提取了所有信息
                
                # 仍然保留Budget的特殊处理
                budget_info = self._extract_budget_from_input(user_input)
                if budget_info:
                    conv_state["extracted_info"]["budget"] = budget_info
                    logger.info(f"Updated budget information in confirmation stage: {budget_info}")
                    if "budget" in conv_state["missing_info"]:
                        conv_state["missing_info"].remove("budget")
                
                # 首先检查是否有提供新的具体信息
                has_new_info = False
                # 设置一个门槛，检查是否有足够的新信息来判断这不是简单的确认/拒绝
                if (extracted_info.get("location") or 
                    extracted_info.get("cuisine") or 
                    extracted_info.get("budget") or 
                    extracted_info.get("time") or 
                    extracted_info.get("is_for_now") is not None):
                    has_new_info = True
                    logger.info("Detected new information in confirmation response")
                
                # 检查是否是确认回复
                is_confirmation = self._is_confirmation(user_input)
                logger.info(f"Is confirmation check result: {is_confirmation}")
                    
                if not has_new_info and is_confirmation:
                    # 明确设置确认状态为True
                    conv_state["extracted_info"]["confirmed"] = True
                    logger.info("Setting confirmation status to True and proceeding to recommendations")
                    next_stage = "generate_recommendations"
                    response = "Great! Let me find some restaurants that match your preferences..."
                else:
                    # 如果不确认或提供了新信息，回到信息收集阶段
                    # 确保确认状态为False
                    conv_state["extracted_info"]["confirmed"] = False
                    next_stage = "information_gathering"
                    response = "I understand. Let's refine your preferences. " + \
                              self._generate_information_request(conv_state["extracted_info"], 
                                                              conv_state["missing_info"])
        
        # Handle post-recommendation stage
        elif current_stage == "recommendations_presented":
            # Check if user wants more info, refinement, or new search
            intent = self._determine_post_recommendation_intent(user_input)
            
            if intent == "refine":
                next_stage = "information_gathering"
                response = "I'd be happy to refine the search. What would you like to change about your preferences?"
            elif intent == "new_search":
                # Reset extracted info but keep conversation history
                conv_state["extracted_info"] = {
                    "location": None, "cuisine": None, "budget": None,
                    "occasion": None, "time": None, "priorities": [],
                    "confirmed": False,
                    "group_size": None,
                    "additional_preferences": {}
                }
                conv_state["missing_info"] = ["location", "cuisine", "budget", "priorities"]
                next_stage = "information_gathering"
                response = "Let's start a new search. What kind of restaurant are you looking for now?"
            else:  # more_info
                next_stage = "recommendations_presented"
                response = "Is there something specific you'd like to know about these recommendations?"
        else:
            # Default fallback
            next_stage = "information_gathering"
            response = "I'm not sure I understood. Could you tell me what kind of restaurant you're looking for?"
            
        # Add system response to conversation history
        conv_state["conversation_history"].append({
            "role": "assistant",
            "content": response
        })
        
        # Update stage
        conv_state["current_stage"] = next_stage
        
        return next_stage, response, conv_state
    
    def post(self, shared, prep_res, exec_res):
        """Store updated conversation state and determine next action."""
        # Check if there's an error in the shared store
        if "error_info" in shared and shared["error_info"]:
            logger.error(f"Error detected in shared store: {shared['error_info']}")
            return "error"
            
        next_stage, response, updated_conv_state = exec_res
        
        # Update conversation state in shared store
        shared["conversation_state"] = updated_conv_state
        shared["system_response"] = response
        
        # 记录下一阶段和确认状态，用于调试
        logger.info(f"Next stage after processing: {next_stage}")
        is_confirmed = updated_conv_state["extracted_info"].get("confirmed", False)
        logger.info(f"Confirmation status after processing: {is_confirmed}")
        
        # 检查是否已经确认并处于确认阶段，直接进入推荐阶段
        if is_confirmed and (next_stage == "confirmation" or next_stage == "generate_recommendations"):
            logger.info("User has confirmed, proceeding to recommendations immediately")
            # Determine next action based on stage
            # 确保我们真的进入推荐阶段
            logger.info("Proceeding to recommendation generation stage")
            # Transfer extracted info to the format expected by recommendation flow
            parsed_query = {
                "location": updated_conv_state["extracted_info"]["location"],
                "cuisine": updated_conv_state["extracted_info"]["cuisine"] 
                          if isinstance(updated_conv_state["extracted_info"]["cuisine"], list) 
                          else [updated_conv_state["extracted_info"]["cuisine"]],
                "budget_pp": updated_conv_state["extracted_info"]["budget"],
                "vibe": updated_conv_state["extracted_info"]["occasion"],
                "priorities": updated_conv_state["extracted_info"]["priorities"],
                "group_size": updated_conv_state["extracted_info"]["group_size"],
                "additional_preferences": updated_conv_state["extracted_info"]["additional_preferences"]
            }
            
            # Add user's physical location for distance calculations
            if "user_geolocation" in updated_conv_state["extracted_info"]:
                shared["user_geolocation"] = updated_conv_state["extracted_info"]["user_geolocation"]
            
            shared["user_query"] = self._generate_structured_query(updated_conv_state["extracted_info"])
            shared["parsed_query"] = parsed_query
            return "recommend"
        
        # 继续常规处理逻辑
        # Determine next action based on stage
        if next_stage == "generate_recommendations":
            # 确保我们真的进入推荐阶段
            logger.info("Proceeding to recommendation generation stage")
            # Transfer extracted info to the format expected by recommendation flow
            parsed_query = {
                "location": updated_conv_state["extracted_info"]["location"],
                "cuisine": updated_conv_state["extracted_info"]["cuisine"] 
                          if isinstance(updated_conv_state["extracted_info"]["cuisine"], list) 
                          else [updated_conv_state["extracted_info"]["cuisine"]],
                "budget_pp": updated_conv_state["extracted_info"]["budget"],
                "vibe": updated_conv_state["extracted_info"]["occasion"],
                "priorities": updated_conv_state["extracted_info"]["priorities"],
                "group_size": updated_conv_state["extracted_info"]["group_size"],
                "additional_preferences": updated_conv_state["extracted_info"]["additional_preferences"]
            }
            
            # Add user's physical location for distance calculations
            if "user_geolocation" in updated_conv_state["extracted_info"]:
                shared["user_geolocation"] = updated_conv_state["extracted_info"]["user_geolocation"]
            
            shared["user_query"] = self._generate_structured_query(updated_conv_state["extracted_info"])
            shared["parsed_query"] = parsed_query
            return "recommend"
        else:
            return "continue_conversation"
    
    def _extract_entities(self, user_input, current_info):
        """Extract structured information from user input."""
        import json
        import re
        
        # First check for removal requests
        removal_detected = False
        removed_preferences = {}
        
        # Initialize regex extracted values
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
            
            # Check for preference removal patterns
            removal_patterns = [
                # General negations
                r'(?:don\'t|do not|no longer|not|remove|delete|cancel|取消|不要|别)\s+(?:need|want|care about|require)?\s+([a-z\s]+)',
                # Specific for noise level
                r'(?:don\'t|do not|no longer|not|不再|不要)\s+(?:need|want|要求)?\s+(?:a\s+)?quiet\s+(?:place|restaurant|environment|setting|atmosphere)',
                # Direct cancelation
                r'(?:remove|delete|cancel|取消|删除)\s+(?:the\s+)?([a-z\s]+)(?:\s+requirement| preference| setting)?',
                # Chinese patterns
                r'(?:不需要|不要|取消|删除)\s+([^\s]+)'
            ]

            for pattern in removal_patterns:
                matches = re.findall(pattern, user_input.lower())
                if matches:
                    removal_detected = True
                    logger.info(f"Detected preference removal request: {matches}")
                    
                    for match in matches:
                        # Check what kind of preference is being removed
                        if isinstance(match, str):
                            preference = match.strip()
                            
                            # Handle noise level
                            if "quiet" in preference or "noise" in preference or "安静" in preference:
                                if current_info.get("additional_preferences") and "noise_level" in current_info["additional_preferences"]:
                                    removed_preferences["noise_level"] = current_info["additional_preferences"]["noise_level"]
                                    logger.info(f"Removed noise level preference: {removed_preferences['noise_level']}")
                            
                            # Handle outdoor seating
                            if "outdoor" in preference or "patio" in preference or "terrace" in preference or "户外" in preference:
                                if current_info.get("additional_preferences") and "seating" in current_info["additional_preferences"]:
                                    removed_preferences["seating"] = current_info["additional_preferences"]["seating"]
                                    logger.info(f"Removed seating preference: {removed_preferences['seating']}")
                            
                            # Handle cuisines
                            for cuisine in ["italian", "chinese", "japanese", "mexican", "thai", "indian", "french", "vietnamese"]:
                                if cuisine in preference:
                                    if current_info.get("cuisine") == cuisine.capitalize():
                                        removed_preferences["cuisine"] = cuisine.capitalize()
                                        logger.info(f"Removed cuisine preference: {removed_preferences['cuisine']}")
                            
                            # Handle priorities
                            for priority in ["taste", "service", "ambiance", "value"]:
                                if priority in preference:
                                    if current_info.get("priorities") and priority.capitalize() in current_info["priorities"]:
                                        if "priorities" not in removed_preferences:
                                            removed_preferences["priorities"] = []
                                        removed_preferences["priorities"].append(priority.capitalize())
                                        logger.info(f"Removed priority: {priority.capitalize()}")
            
            # Use regex first for key patterns to have backup values
            # First, try regex extraction for common patterns
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
            
            # If we detected preference removal, add this context to the LLM prompt
            removal_context = ""
            if removal_detected:
                removal_context = """
IMPORTANT: The user may be trying to REMOVE or CANCEL some preferences. 
If they are saying they no longer want something (e.g., "don't need quiet place"), 
make sure to set that preference to null or remove it, rather than extracting it as a new preference.
"""

            # Now call LLM to extract structured information and additional preferences
            try:
                from utils import call_llm
                prompt = f"""
You are an intelligent assistant that extracts structured information from user queries about restaurant recommendations.

USER INPUT: {user_input}

{removal_context}

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
  }},
  "preferences_to_remove": ["preference_name1", "preference_name2"]
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
                            
                            # Add preferences_to_remove field if it doesn't exist
                            if not extracted.get("preferences_to_remove"):
                                extracted["preferences_to_remove"] = []
                                
                            # Add currency type if not present
                            if not extracted.get("currency_type"):
                                extracted["currency_type"] = currency_type
                            
                            # If we detected removals via regex, add them to the LLM extraction
                            if removal_detected:
                                # Add the removed preferences to the LLM results
                                for pref_type, value in removed_preferences.items():
                                    if pref_type not in extracted["preferences_to_remove"]:
                                        extracted["preferences_to_remove"].append(pref_type)
                            
                            # Log additional preferences that were captured
                            if extracted.get("additional_preferences") and len(extracted["additional_preferences"]) > 0:
                                logger.info(f"Captured additional preferences: {extracted['additional_preferences']}")
                                
                            # Log preferences to remove
                            if extracted.get("preferences_to_remove") and len(extracted["preferences_to_remove"]) > 0:
                                logger.info(f"Detected preferences to remove: {extracted['preferences_to_remove']}")
                            
                            return extracted
                        else:
                            logger.warning(f"No JSON found in LLM response: {response}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from LLM: {e}")
                else:
                    logger.warning("Empty response from LLM")
            except Exception as e:
                logger.error(f"Error calling LLM for entity extraction: {e}", exc_info=True)
                
            # If LLM extraction failed or response was invalid, use regex results
            logger.warning("Using regex fallback extraction because LLM extraction failed")
            
        except Exception as e:
            logger.error(f"Error in regex extraction: {e}", exc_info=True)
            
        # Create a fallback structure with whatever we could extract
        extracted = {
            "location": regex_location,
            "cuisine": regex_cuisine,
            "budget": regex_budget,
            "occasion": None,
            "time": regex_time,
            "is_for_now": is_for_now,
            "dining_date": dining_date,
            "priorities": [],
            "group_size": regex_group_size,
            "additional_preferences": {},
            "preferences_to_remove": list(removed_preferences.keys()) if removed_preferences else [],
            "currency_type": currency_type
        }
        
        return extracted
    
    def _has_sufficient_info(self, extracted_info):
        """Check if we have enough information to generate recommendations."""
        # Always get user's current geolocation for distance calculations if not already set
        if not extracted_info.get("user_geolocation"):
            # Import geolocation function here to avoid circular import
            from utils.google_maps_api import get_user_geolocation
            
            # Get user's location automatically
            user_geolocation = get_user_geolocation()
            if user_geolocation.get('success'):
                # Store user's current physical location for distance calculations
                extracted_info["user_geolocation"] = user_geolocation
                logger.info(f"Automatically detected user's physical location for distance calculations: {user_geolocation.get('address')}")
        
        # 如果没有指定时间，尝试确定是否是"现在"就餐
        if not extracted_info.get("time") and not extracted_info.get("is_for_now"):
            # 默认假设是现在就餐
            current_time = datetime.datetime.now().strftime("%H:%M")
            extracted_info["time"] = current_time
            extracted_info["is_for_now"] = True
            logger.info(f"No dining time specified, assuming current time: {current_time}")
        
        # Now check if we have the minimum required information
        # We need an explicit restaurant location (not auto-detected) and at least one of cuisine or budget
        if not extracted_info.get("location"):
            return False
        
        # Also good to have at least one of cuisine or budget
        if not extracted_info.get("cuisine") and not extracted_info.get("budget"):
            return False
        
        return True
    
    def _generate_information_request(self, extracted_info, missing_info):
        """Generate a prompt to request missing information."""
        # Keep location in missing info to ensure we ask for it
        filtered_missing = missing_info.copy()  # Work with a copy to avoid modifying the original
        
        # Check what information we already have
        has_location = extracted_info.get("location") is not None
        has_cuisine = extracted_info.get("cuisine") is not None
        has_budget = extracted_info.get("budget") is not None
        has_group_size = extracted_info.get("group_size") is not None
        has_time = extracted_info.get("time") is not None
        is_for_now = extracted_info.get("is_for_now") is True
        
        # Remove from missing if we already have it
        if has_location and "location" in filtered_missing:
            filtered_missing.remove("location")
        if has_cuisine and "cuisine" in filtered_missing:
            filtered_missing.remove("cuisine")
        if has_budget and "budget" in filtered_missing:
            filtered_missing.remove("budget")
        if (has_time or is_for_now) and "time" in filtered_missing:
            filtered_missing.remove("time")
        
        # Log what info is still missing
        logger.info(f"Still missing information: {filtered_missing}")
        
        # If there's nothing to ask for, check if we actually have enough information
        if not filtered_missing:
            # If we have at least location and either cuisine or budget, we can proceed
            if has_location and (has_cuisine or has_budget) and (has_time or is_for_now):
                # We actually have enough information, no need to ask more
                logger.info("We have enough information to proceed, confirming directly")
                return self._generate_confirmation_prompt(extracted_info)
            
            # Otherwise, determine what's most important to ask for
            if not has_location:
                filtered_missing.append("location")
            elif not has_cuisine and not has_budget:
                # Ask for cuisine first, then budget
                filtered_missing.append("cuisine")
            elif not has_cuisine:
                filtered_missing.append("cuisine")
            elif not has_budget:
                filtered_missing.append("budget")
            elif not has_time and not is_for_now:
                filtered_missing.append("time")
        
        # Prioritize the order: location > cuisine > budget > time
        if len(filtered_missing) > 1:
            priority_order = ["location", "cuisine", "budget", "time"]
            filtered_missing.sort(key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        # Create a prompt using LLM to generate natural-sounding request
        prompt = f"""
You are a helpful restaurant recommendation assistant. You need to ask the user for more information to provide better recommendations.

Currently known information:
- Location: {extracted_info.get('location', 'Not specified')}
- Cuisine: {extracted_info.get('cuisine', 'Not specified')}
- Budget: {extracted_info.get('budget', 'Not specified')}
- Group Size: {extracted_info.get('group_size', 'Not specified')}
- Dining Time: {extracted_info.get('time', 'Not specified')}
- Dining Date: {extracted_info.get('dining_date', 'Today/Not specified')}

The most important missing information is: {filtered_missing}

Generate a natural-sounding message that:
1. Acknowledges what you already know (briefly summarize what you understand so far)
2. Asks for ONLY the most important missing information (focus on {filtered_missing[0] if filtered_missing else 'nothing'})
3. Keeps it conversational and friendly, not like a form
4. For location, ALWAYS ask specifically where the user wants to search for restaurants (not where they currently are)
5. For time, ask when they plan to dine (now or a specific time later)
6. IMPORTANT: If the user has already provided multiple pieces of information, express gratitude and focus only on what's still missing
7. Keep your response VERY brief - just 1-2 short sentences

Your message:
"""
        try:
            from utils import call_llm
            response = call_llm.call_llm(prompt)
            if response and len(response.strip()) > 0:
                return response
            else:
                raise ValueError("Empty response from LLM")
        except Exception as e:
            logger.error(f"Error generating information request: {e}")
            
            # Fallback responses if LLM call fails - more specific based on what's missing
            if not has_location:
                return "Which neighborhood or area would you like to find restaurants in?"
            elif not has_cuisine:
                location = extracted_info.get('location', 'the area')
                return f"What type of cuisine are you looking for in {location}?"
            elif not has_budget:
                cuisine = extracted_info.get('cuisine', 'meal')
                return f"Do you have a budget in mind for this {cuisine}?"
            elif not has_group_size:
                return "How many people will be dining?"
            elif not has_time and not is_for_now:
                return "When are you planning to dine? Is it for now or a specific time later?"
            else:
                return "Could you tell me a bit more about what you're looking for in a restaurant?"
    
    def _generate_confirmation_prompt(self, extracted_info):
        """Generate a confirmation message to verify extracted information."""
        # Format each piece of information appropriately
        location = extracted_info.get('location', 'Not specified')
        
        cuisine = extracted_info.get('cuisine', 'Not specified')
        if isinstance(cuisine, list):
            cuisine = ", ".join(cuisine)
        
        # 根据货币类型设置符号
        budget = extracted_info.get('budget', 'Not specified')
        if isinstance(budget, (int, float)):
            currency_symbol = "$"  # 默认美元符号
            currency_type = extracted_info.get('currency_type', 'GBP')  # 默认使用英镑
            
            if currency_type == "GBP":
                currency_symbol = "£"
            elif currency_type == "EUR":
                currency_symbol = "€"
            elif currency_type == "USD":
                currency_symbol = "$"
                
            budget = f"{currency_symbol}{budget} per person"
        
        occasion = extracted_info.get('occasion', 'Not specified')
        
        group_size = extracted_info.get('group_size', 'Not specified')
        if isinstance(group_size, int):
            group_size = f"{group_size} {'person' if group_size == 1 else 'people'}"
        
        # 格式化时间信息
        time_info = "Now"  # 默认为现在
        if extracted_info.get("is_for_now") is True:
            time_info = "Now"
        elif extracted_info.get("time"):
            time_info = extracted_info.get("time")
            if extracted_info.get("dining_date"):
                time_info += f" on {extracted_info.get('dining_date')}"
            else:
                time_info += " today"
        
        priorities = extracted_info.get('priorities', [])
        if priorities:
            priorities_text = ", ".join(priorities)
        else:
            priorities_text = "Not specified"
            
        # Construct the prompt with neatly formatted information
        prompt = f"""
Based on our conversation, I understand you're looking for:
- Location: {location}
- Cuisine: {cuisine}
- Budget: {budget}
- Dining Time: {time_info}
- Occasion: {occasion}
- Group Size: {group_size}
- Priorities: {priorities_text}
"""

        # Add additional preferences if any
        additional_prefs = extracted_info.get('additional_preferences', {})
        if additional_prefs and len(additional_prefs) > 0:
            prompt += "\nAdditional preferences:\n"
            for pref_name, pref_value in additional_prefs.items():
                formatted_name = pref_name.replace('_', ' ').title()
                prompt += f"- {formatted_name}: {pref_value}\n"

        prompt += "\nIs this correct? I can adjust any details before finding restaurants for you."
        return prompt
    
    def _is_confirmation(self, user_input):
        """Check if user input is a confirmation."""
        # 记录原始输入以便调试
        logger.info(f"Checking confirmation for input: '{user_input}'")
        
        # 处理确切的"yes"
        if user_input.lower().strip() == "yes":
            logger.info("Input is exactly 'yes' - definite confirmation")
            return True
        
        # Use simple keyword matching for basic confirmation detection
        positive_keywords = ["correct", "right", "good", "perfect", "fine", "ok", "okay", "sure", "yep", "yeah", "ye", "y", "yup", "👍", "sounds good"]
        negative_keywords = ["no", "not", "incorrect", "wrong", "change", "modify", "adjust", "edit", "different", "nope", "👎", "修改", "错误"]
        
        # 使用\b确保匹配完整的单词，避免将"in"错误识别为"n"
        negative_patterns = [r'\b{}\b'.format(keyword) for keyword in negative_keywords]
        positive_patterns = [r'\b{}\b'.format(keyword) for keyword in positive_keywords]
        
        # Convert to lowercase for case-insensitive matching
        lower_input = user_input.lower()
        
        # 如果输入只包含一个单词，且为"yes"或其变体，直接返回True
        if lower_input.strip() in ["yes", "yeah", "yep", "yup", "ye", "ok", "okay", "sure", "y", "是", "好", "对"]:
            logger.info(f"Simple positive word detected: '{lower_input.strip()}' - definite confirmation")
            return True
        
        # 检查是否包含位置、菜系等信息
        # 如果用户在回复中提供了具体信息，我们不应该将其视为简单的确认/否认
        import re
        location_indicators = [
            r'\bin\s+([a-zA-Z\s]+)',
            r'\b(holborn|soho|covent garden|camden|islington|shoreditch|mayfair|chelsea|kensington|westminster|london)\b'
        ]
        
        cuisine_indicators = [
            r'\b(italian|chinese|japanese|mexican|thai|indian|french|vietnamese|korean|turkish|greek|spanish|lebanese|american)\b'
        ]
        
        # 检查是否有提供具体信息
        has_specific_info = False
        for pattern in location_indicators + cuisine_indicators:
            if re.search(pattern, lower_input):
                logger.info(f"Detected specific information in confirmation response: {pattern}")
                has_specific_info = True
                break
        
        # 如果用户提供了具体信息，这可能不是一个简单的确认或否认
        if has_specific_info:
            logger.info("User provided specific information in confirmation response - treating as refinement")
            return False
        
        # First check for negative keywords (these have priority)
        for pattern in negative_patterns:
            if re.search(pattern, lower_input):
                logger.info(f"Detected negative confirmation keyword with pattern: {pattern}")
                return False
        
        # Check for confirmation keywords
        for pattern in positive_patterns:
            if re.search(pattern, lower_input):
                logger.info(f"Detected positive confirmation keyword with pattern: {pattern}")
                return True
            
        # If no clear signal, analyze the full message using an LLM
        try:
            from utils import call_llm
            prompt = f"""
Determine if the user's message is a confirmation or rejection of the restaurant search parameters.
User message: "{user_input}"

Return ONLY "yes" if it's a confirmation, or "no" if it's a rejection.
"""
            response = call_llm.call_llm(prompt)
            if response and response.strip().lower() in ["yes", "no"]:
                result = response.strip().lower() == "yes"
                logger.info(f"Used LLM to determine confirmation: {result}")
                return result
        except Exception as e:
            logger.error(f"Error using LLM for confirmation analysis: {e}")
        
        # Default to false if we can't determine (safer to ask again)
        logger.info("No clear confirmation detected, defaulting to False")
        return False
    
    def _determine_post_recommendation_intent(self, user_input):
        """Determine user's intent after recommendations are presented."""
        # Try to match clear intent patterns first
        import re
        
        # Check for refinement patterns
        refine_patterns = [
            r'\b(?:change|different|modify|adjust|cheaper|expensive|another|other|refine|update)\b',
            r'\b(?:don\'t like|not interested|something else|other options)\b',
            r'\b(?:budget|price|cuisine|location|area)\b'
        ]
        
        # Check for new search patterns
        new_search_patterns = [
            r'\b(?:new search|start over|start again|completely different|restart|reset)\b',
            r'\b(?:different city|different area|different cuisine)\b',
            r'\b(?:want to search for|looking for something|search for)\b'
        ]
        
        # Check for more info patterns
        more_info_patterns = [
            r'\b(?:more info|tell me more|details|reviews|rating|stars|menu|photos|picture|hours|open)\b',
            r'\b(?:address|phone|number|reservation|book|website|link)\b',
            r'\b(?:which one|recommend|best one|favorite|top choice|what about|how about)\b'
        ]
        
        # Count matches for each category
        refine_count = sum(1 for pattern in refine_patterns if re.search(pattern, user_input.lower()))
        new_search_count = sum(1 for pattern in new_search_patterns if re.search(pattern, user_input.lower()))
        more_info_count = sum(1 for pattern in more_info_patterns if re.search(pattern, user_input.lower()))
        
        logger.info(f"Intent detection counts - refine: {refine_count}, new_search: {new_search_count}, more_info: {more_info_count}")
        
        # If we have a clear winner by pattern matching
        if refine_count > new_search_count and refine_count > more_info_count:
            return "refine"
        elif new_search_count > refine_count and new_search_count > more_info_count:
            return "new_search"
        elif more_info_count > refine_count and more_info_count > new_search_count:
            return "more_info"
        
        # If no clear pattern match, use LLM
        prompt = f"""
Determine the user's intent from their message after restaurant recommendations were presented.
Classify into one of these categories:
- refine: User wants to modify search parameters (different cuisine, budget, etc.)
- more_info: User wants more information about the recommendations
- new_search: User wants to start a completely new search

User message: "{user_input}"

Return ONLY one of these exact words: "refine", "more_info", or "new_search"
"""
        try:
            from utils import call_llm
            response = call_llm.call_llm(prompt).strip().lower()
            if response in ["refine", "more_info", "new_search"]:
                logger.info(f"LLM determined intent: {response}")
                return response
            else:
                logger.warning(f"LLM returned invalid intent: {response}, defaulting to more_info")
                return "more_info"  # Default fallback
        except Exception as e:
            logger.error(f"Error determining post-recommendation intent: {e}")
            return "more_info"  # Default fallback
    
    def _generate_structured_query(self, extracted_info):
        """Generate a structured query string from extracted information."""
        parts = []
        
        # Add cuisine information if available
        if extracted_info.get("cuisine"):
            cuisine = extracted_info["cuisine"]
            if isinstance(cuisine, list):
                cuisine_str = " or ".join(cuisine)
            else:
                cuisine_str = cuisine
            parts.append(cuisine_str)
            
        # Add location information if available
        if extracted_info.get("location"):
            parts.append(f"in {extracted_info['location']}")
            
        # Add budget information if available
        if extracted_info.get("budget"):
            budget = extracted_info["budget"]
            # 使用正确的货币符号
            currency_symbol = "$"  # 默认
            currency_type = extracted_info.get('currency_type', 'GBP')  # 默认英镑
            
            if currency_type == "GBP":
                currency_symbol = "£"
            elif currency_type == "EUR":
                currency_symbol = "€"
            elif currency_type == "USD":
                currency_symbol = "$"
                
            if isinstance(budget, list) and len(budget) == 2:
                parts.append(f"budget around {currency_symbol}{budget[0]}-{currency_symbol}{budget[1]}pp")
            elif isinstance(budget, (int, float)):
                parts.append(f"budget around {currency_symbol}{budget}pp")
                
        # Add occasion information if available
        if extracted_info.get("occasion"):
            parts.append(f"for {extracted_info['occasion']}")
            
        # Add time information if available
        if extracted_info.get("is_for_now") is True:
            parts.append("for dining right now")
        elif extracted_info.get("time"):
            time_str = f"for dining at {extracted_info['time']}"
            if extracted_info.get("dining_date"):
                time_str += f" on {extracted_info['dining_date']}"
            parts.append(time_str)
            
        # Add priorities if available
        if extracted_info.get("priorities") and len(extracted_info["priorities"]) > 0:
            priorities = ", ".join([f"good {p.lower()}" for p in extracted_info["priorities"]])
            parts.append(f"with {priorities}")
            
        # Add group size if available
        if extracted_info.get("group_size"):
            parts.append(f"for {extracted_info['group_size']} people")
            
        # Add additional preferences if relevant
        additional_prefs = extracted_info.get("additional_preferences", {})
        if additional_prefs:
            # Add noise level if specified
            if "noise_level" in additional_prefs:
                parts.append(f"with {additional_prefs['noise_level']} atmosphere")
                
            # Add seating preference if specified
            if "seating" in additional_prefs:
                parts.append(f"with {additional_prefs['seating']} seating")
                
        # Join all parts with commas and log the query
        query = ", ".join(parts)
        logger.info(f"Generated structured query: {query}")
        return query

    def _extract_budget_from_input(self, user_input):
        """Extract budget information from user input."""
        import re
        
        # Match patterns with currency or budget-specific terms
        budget_patterns = [
            # Currency symbols followed by numbers
            r'[\$£€¥]\s*(\d+)',
            
            # Numbers followed by currency symbols or terms
            r'(\d+)\s*[\$£€¥]',
            r'(\d+)\s*(?:dollars|pounds|euros|gbp|usd|eur)',
            
            # Budget-specific phrases
            r'(?:budget|spend|cost|price)[^\d]*(\d+)',
            r'(?:upto|up to|maximum|max)[^\d]*(\d+)',
            
            # Per-person budget
            r'(\d+)(?:\s*(?:per|each|a|per person|per head|pp|p/p))',
            r'(?:per|each|a|per person|per head|pp|p/p)[^\d]*(\d+)'
        ]
        
        # Try each pattern
        for pattern in budget_patterns:
            match = re.search(pattern, user_input.lower(), re.IGNORECASE)
            if match:
                budget_value = int(match.group(1))
                logger.info(f"Extracted budget from input using pattern '{pattern}': {budget_value}")
                return budget_value
                
        # Check for mentions of cost ranges
        range_match = re.search(r'(\d+)[^\d]+to[^\d]+(\d+)', user_input.lower())
        if range_match:
            min_val = int(range_match.group(1))
            max_val = int(range_match.group(2))
            # Use the average as a single value
            budget_value = (min_val + max_val) // 2
            logger.info(f"Extracted budget range from input: {min_val}-{max_val}, using average: {budget_value}")
            return budget_value
            
        return None


class ConversationContinuationNode(Node):
    """Handles continuing the conversation when not ready for recommendations."""
    
    def prep(self, shared):
        return shared.get("system_response")
        
    def exec(self, system_response):
        # Simply pass through the system response
        return system_response
        
    def post(self, shared, prep_res, exec_res):
        # Check if there's an error in the shared store
        if "error_info" in shared and shared["error_info"]:
            logger.error(f"Error detected in ConversationContinuationNode: {shared['error_info']}")
            return "error"
            
        # Print the response and prompt for next input
        print(f"\nAssistant: {exec_res}")
        user_input = input("\nYou: ")
        
        # Store new user input for next iteration
        shared["user_input"] = user_input
        
        # Continue the conversation loop
        return "continue_dialog"


class ConversationCompletionNode(Node):
    """Handles the end of a conversation after recommendations are presented."""
    
    def prep(self, shared):
        return shared.get("final_response"), shared.get("conversation_state")
        
    def exec(self, prep_res):
        final_response, conv_state = prep_res
        
        # Update conversation state to indicate recommendations were presented
        if conv_state:
            conv_state["current_stage"] = "recommendations_presented"
            
        return final_response, conv_state
        
    def post(self, shared, prep_res, exec_res):
        # Check if there's an error in the shared store
        if "error_info" in shared and shared["error_info"]:
            logger.error(f"Error detected in ConversationCompletionNode: {shared['error_info']}")
            return "error"
            
        final_response, updated_conv_state = exec_res
        
        # Update conversation state
        if updated_conv_state:
            shared["conversation_state"] = updated_conv_state
            
        # Print the recommendation
        print(f"\nAssistant: {final_response}")
        
        # Get user feedback/follow-up
        user_input = input("\nYou: ")
        
        # Store for next iteration
        shared["user_input"] = user_input
        
        # Continue conversation
        return "continue_dialog"


class DebugNode(Node):
    """Provides debugging information if there's an error in the conversation flow."""
    
    def prep(self, shared):
        # Collect all relevant information from the shared store
        return {
            "conversation_state": shared.get("conversation_state", {}),
            "user_input": shared.get("user_input", ""),
            "system_response": shared.get("system_response", ""),
            "error": shared.get("error_info", None)
        }
        
    def exec(self, debug_info):
        # Log the debug information
        logger.error("=== DEBUG INFORMATION ===")
        logger.error(f"Last user input: {debug_info['user_input']}")
        logger.error(f"Last system response: {debug_info['system_response']}")
        
        if debug_info.get("error"):
            logger.error(f"Error: {debug_info['error']}")
            
        conv_state = debug_info.get("conversation_state", {})
        logger.error(f"Current stage: {conv_state.get('current_stage', 'unknown')}")
        logger.error(f"Extracted info: {conv_state.get('extracted_info', {})}")
        
        # Format a user-friendly error message
        error_message = """
I'm sorry, but it seems there was an error processing your request. 
This could be due to:
- A temporary issue with one of our services
- An unexpected input format
- A limitation in our current capabilities

Would you like to try again with a different request?
"""
        return error_message
        
    def post(self, shared, prep_res, exec_res):
        # Display the error message and prompt for next input
        print(f"\nAssistant: {exec_res}")
        user_input = input("\nYou: ")
        
        # Store new user input for next iteration
        shared["user_input"] = user_input
        
        # Reset any error information
        if "error_info" in shared:
            del shared["error_info"]
            
        # Continue the conversation loop
        return "continue_dialog" 