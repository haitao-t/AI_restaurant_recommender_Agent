# Restaurant Recommendation System Design Document

This document outlines the design for the Restaurant Recommendation System based on PocketFlow, a minimal 100-line framework for orchestrating LLM workflows as connected nodes with a shared data store.

## 1. Requirements

*   **Goal:** Provide users with structured, helpful restaurant recommendations based on their preferences and AI-driven analysis of user reviews.

*   **Input:** User query (natural language detailing location, cuisine, budget, vibe, priorities, etc.).
    - Example: "Find me a great Italian place in Soho for a date night, budget around £60pp, focus on ambiance and good service."

*   **Output:** A Markdown formatted response including:
    *   Brief intro acknowledging the request.
    *   A comparison table of top candidates showing dimensional scores (1-10), Fit Score (0-10), and key notes.
    *   Concise summary points linking recommendations to user priorities.
    *   A concluding suggestion and a call to action.

*   **Core Logic:**
    1.  **Query Parsing:** Extract structured data (location, cuisine, priorities) from natural language.
    2.  **Restaurant Discovery:** Find candidate restaurants matching basic criteria.
    3.  **Review Collection:** Gather user reviews for each candidate.
    4.  **Agent Decision:** Determine next steps based on context (full analysis, clarification, etc.).
    5.  **Review Analysis:** Analyze reviews using LLM to extract dimensional ratings.
    6.  **Fit Calculation:** Calculate personalized fit scores based on user priorities.
    7.  **Response Generation:** Format results into a helpful Markdown response.

*   **Location-Related Features:**
    1.  **User's Current Location Detection:** Automatically detects the user's physical location using IP geolocation for distance calculations.
    2.  **Restaurant Search Location:** Uses the location explicitly specified by the user to search for restaurants (not the auto-detected location).
    3.  **Distance Information:** Shows travel distance and time estimates from user's current location to each restaurant.
    4.  **Clear Presentation:** Explicitly indicates when distances are from the user's current location rather than the search location.

## 2. Flow Design

The workflow includes a decision step using the Flock Agent model after fetching reviews to determine the most appropriate next action based on the available data and user query.

```mermaid
graph LR
    A[Start: User Query] --> B(ParseUserQueryNode \nLLM);
    B --> C(FindRestaurantsNode \nMaps API);
    C -- default --> D(FetchReviewsNode \nMaps API);
    C -- no_candidates_found --> J(NoCandidatesFoundNode);
    D --> K{DecideActionNode \nFlock Agent Decision};  % Decision Point
    K -- analyze --> E["AnalyzeReviewsBatchNode \nBatch, Fine-tuned LLM (1-10)"];
    K -- clarify --> L[AskClarificationNode \n(Placeholder)]; % Branch for clarification
    E --> F["CalculateFitScoreNode \nPython Logic (0-10 Fit)"];
    F --> G["GenerateResponseNode \nLLM + Formatting"];
    G --> H[End: Formatted Response];
    J --> H; % No candidates also ends
    L --> H; % Clarification path ends (simplified for now)

    %% Styling (Optional)
    style K fill:#lightgrey,stroke:#333,stroke-width:2px
    style E fill:#f9d,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:1px
```

### Flow Explanation

- **Linear Path (Happy Path):** User query → Parse → Find restaurants → Fetch reviews → Agent decision → Analyze reviews → Calculate fit → Generate response
- **No Results Path:** If no restaurants match initial criteria, the flow branches to a specific node that provides a helpful message
- **Decision Point:** After fetching reviews, the Flock Agent decides whether to:
  - **Analyze:** Proceed with full review analysis (common path)
  - **Clarify:** Ask user for clarification if query is ambiguous (currently a placeholder)
  - *(Future: List Only)* Present a basic list without in-depth analysis

## 3. Utilities (`utils/`)

### `google_maps_api.py` - Google Maps API Integration

*   **`find_restaurants(location, cuisine, **kwargs)`**:
    - **Purpose:** Queries Google Maps Places API to find restaurants matching location and cuisine criteria
    - **Input:** 
      - `location` (str): Geographic area (e.g., "Soho, London")
      - `cuisine` (list): List of cuisine types (e.g., ["Italian", "Pizza"])
      - `**kwargs`: Optional parameters like radius, price_level, etc.
    - **Output:** List of restaurant dictionaries, each containing:
      ```python
      {
          "id": "ChIJ...",              # Place ID (string)
          "name": "Pizza Express",      # Restaurant name
          "address": "123 Soho St",     # Street address
          "rating": 4.5,                # Google average rating (0-5)
          "price_level": 2,             # Price level (1-4)
          "user_ratings_total": 450     # Number of ratings
      }
      ```
    - **Implementation:** Uses `googlemaps` Python library with error handling and optional pagination
    - **API Requirements:** Google Maps API key with Places API enabled

*   **`get_user_geolocation(fallback_location="London")`**:
    - **Purpose:** Attempts to automatically determine the user's current physical location using IP geolocation
    - **Input:** 
      - `fallback_location` (str): Default location to use if geolocation fails
    - **Output:** Dictionary containing location information:
      ```python
      {
          "success": True,            # Whether geolocation was successful
          "address": "London, UK",    # Formatted address string
          "lat": 51.5074,             # Latitude
          "lng": -0.1278,             # Longitude
          "neighborhood": "Holborn"   # Local neighborhood name if available
      }
      ```
    - **Implementation:** Uses ipinfo.io's free API for IP-based geolocation and reverse geocoding with Google Maps
    - **Important:** This provides the user's actual physical location, which is separate from where they want to search for restaurants. Used only for distance calculations.

*   **`get_reviews_for_restaurant(restaurant_id, max_reviews=25)`**:
    - **Purpose:** Fetches user reviews for a specific restaurant
    - **Input:** 
      - `restaurant_id` (str): Google Place ID
      - `max_reviews` (int): Maximum number of reviews to fetch (default: 25)
    - **Output:** List of review dictionaries:
      ```python
      [
          {
              "author_name": "John D.",
              "rating": 4,             # Google review rating (1-5)
              "text": "Great food...",  # Review content
              "time": 1622742400      # Unix timestamp
          },
          # More reviews...
      ]
      ```
    - **Implementation:** Uses Place Details API with review filtering and sorting by recency
    - **Limitations:** Google API typically returns max 5 reviews per call

### `call_finetuned_analyzer.py` - Fine-tuned Model Integration

*   **`analyze_reviews_with_finetuned_model(restaurant_id, reviews)`**:
    - **Purpose:** Calls a specialized fine-tuned model to analyze restaurant reviews and extract dimensional scores
    - **Input:** 
      - `restaurant_id` (str): Restaurant identifier
      - `reviews` (list): List of review texts or dictionaries
    - **Output:** Dictionary of dimensional scores (1-10 scale) or None if failed:
      ```python
      {
          "Taste": 8.5,       # Food quality score
          "Service": 7.2,     # Service quality score
          "Ambiance": 9.0,    # Atmosphere score
          "Value": 6.8,       # Value for money score
          "Waiting": 5.5,     # Wait time score (lower is longer wait)
          "Noise": 7.0        # Noise level score (higher is quieter)
      }
      ```
    - **Implementation:** HTTP POST request to deployed model endpoint with error handling and logging
    - **API Requirements:** Fine-tuned model API URL (configurable via environment variable)

### `call_llm.py` - General LLM Utilities

*   **`call_llm(prompt, model=None, max_retries=3, delay=5)`**:
    - **Purpose:** Generic wrapper for LLM API calls with retry logic and error handling
    - **Input:** 
      - `prompt` (str): The text prompt for the LLM
      - `model` (str, optional): Model name override (default: environment variable or "gpt-4o-mini")
      - `max_retries` (int): Maximum retry attempts for rate limits (default: 3)
      - `delay` (int): Seconds between retries (default: 5)
    - **Output:** Model response text (str) or None if all retries fail
    - **Implementation:** Uses OpenAI Python client with exception handling for rate limits and API errors
    - **API Requirements:** OpenAI API key

*   **`parse_user_query_llm(user_query)`**:
    - **Purpose:** Specialized LLM prompt to extract structured data from natural language restaurant request
    - **Input:** `user_query` (str): User's restaurant request
    - **Output:** Dictionary of parsed preferences or None:
      ```python
      {
          "location": "Soho, London",
          "cuisine": ["Italian", "Pizza"],
          "budget_pp": [50, 70],      # Min, max per person
          "vibe": "Romantic",
          "priorities": ["Ambiance", "Service"]
      }
      ```
    - **Implementation:** Crafted prompt instructing LLM to extract specific fields + JSON parsing
    - **Error Handling:** Robust JSON extraction using regex patterns and validation

*   **`generate_final_response_llm(user_query, ranked_recommendations)`**:
    - **Purpose:** Formats final Markdown response with table and personalized recommendations
    - **Input:** 
      - `user_query` (str): Original user request
      - `ranked_recommendations` (list): Sorted list of restaurant dictionaries with scores
    - **Output:** Markdown-formatted response string
    - **Implementation:** Creates a data table, then prompts LLM to generate narrative around the data

### `call_flock_agent.py` - Agent Decision Making

*   **`decide_next_action_with_flock(context, tools)`**:
    - **Purpose:** Uses Flock Agent LLM to make a decision about next steps in the recommendation process
    - **Input:** 
      - `context` (str): Current state description (parsed query, found restaurants, availability of reviews)
      - `tools` (list): List of available function descriptions (analyze, clarify, etc.)
    - **Output:** Dictionary with chosen action:
      ```python
      {
          "name": "analyze_restaurant_reviews",
          "arguments": {
              "restaurant_ids": ["ChIJ...", "ChIJ..."]  # Optional specific IDs
          }
      }
      ```
    - **Implementation:** Uses HuggingFace Transformers to load and run the Flock Agent model locally
    - **Dependencies:** Requires torch, transformers, and ideally accelerate libraries

## 4. Node Design (`nodes.py`)

### Query Understanding & Data Collection

*   **`ParseUserQueryNode(Node)`**:
    - **Purpose:** Converts natural language query into structured data
    - **Methods:**
      - `prep(shared)`: Retrieves user_query from shared store
      - `exec(user_query)`: Calls parse_user_query_llm to get structured preferences
      - `post(shared, prep_res, exec_res)`: Stores parsed_query in shared store
    - **Fallback Logic:** If parsing returns empty/incomplete, extracts basic info through text matching
    - **Output:** Action "default" to proceed to restaurant search

*   **`FindRestaurantsNode(Node)`**:
    - **Purpose:** Searches for restaurants matching user's location and cuisine preferences
    - **Methods:**
      - `prep(shared)`: Extracts location and cuisine from parsed_query
      - `exec(prep_res)`: Calls find_restaurants with location, cuisine parameters
      - `post(shared, prep_res, exec_res)`: Stores candidate_restaurants or returns "no_candidates_found" action
    - **Branching Logic:** Returns "no_candidates_found" action if no matches found

*   **`FetchReviewsNode(Node)`**:
    - **Purpose:** Retrieves user reviews for each candidate restaurant
    - **Methods:**
      - `prep(shared)`: Gets candidate_restaurants list
      - `exec(candidates)`: Iterates through candidates, calling get_reviews_for_restaurant for each
      - `post(shared, prep_res, exec_res)`: Stores reviews_data dictionary (restaurant_id → reviews)
    - **Optimization:** Can limit reviews per restaurant (typically 25 max)

### Decision Making & Analysis

*   **`DecideActionNode(Node)`**:
    - **Purpose:** Uses agent to determine most appropriate next action based on available data
    - **Methods:**
      - `prep(shared)`: Constructs context string with user_query, parsed_query, candidates count, reviews availability
      - `exec(context)`: Calls decide_next_action_with_flock with context and predefined tools
      - `post(shared, prep_res, exec_res)`: Maps function name ("analyze_restaurant_reviews", "ask_user_clarification") to flow action ("analyze", "clarify")
    - **Configuration:** Includes max_retries=1, wait=3 for API reliability
    - **Available Tools:** Defined in __init__ as JSON Schema objects for function calling

*   **`AnalyzeReviewsBatchNode(BatchNode)`**:
    - **Purpose:** Processes restaurant reviews in batches to extract dimensional scores
    - **Methods:**
      - `prep(shared)`: Creates list of (restaurant_id, reviews) tuples for batch processing
      - `exec(item)`: For each tuple, analyzes reviews to get dimensional scores via LLM prompt
      - `post(shared, prep_res, exec_res_list)`: Aggregates scores from all restaurants into dimensional_scores_1_10
    - **BatchNode Benefit:** Manages retries and error handling for each restaurant separately
    - **Configuration:** Includes max_retries=2, wait=5 for API reliability

*   **`CalculateFitScoreNode(Node)`**:
    - **Purpose:** Computes personalized fit scores based on dimension scores and user priorities
    - **Methods:**
      - `prep(shared)`: Gets priorities, dimensional_scores, and candidates
      - `exec(prep_res)`: Implements weighted average calculation with double weight for user priorities
      - `post(shared, prep_res, exec_res)`: Stores sorted ranked_recommendations list
    - **Algorithm:** 
      1. Maps user priorities to dimension keys (case-insensitive)
      2. Assigns weight=2.0 for prioritized dimensions, weight=1.0 for others
      3. Calculates weighted average of core dimensions (Taste, Service, Ambiance, Value)
      4. Creates ranking with all scores and fit_score, sorted by fit_score
    
### Response Generation & Error Handling

*   **`GenerateResponseNode(Node)`**:
    - **Purpose:** Creates final Markdown response with formatted tables and personalized recommendations
    - **Methods:**
      - `prep(shared)`: Gets user_query and ranked_recommendations
      - `exec(prep_res)`: Calls generate_final_response_llm to create formatted Markdown
      - `post(shared, prep_res, exec_res)`: Stores final_response in shared store
    - **Empty Result Handling:** Returns helpful message if no recommendations available

*   **`NoCandidatesFoundNode(Node)`**:
    - **Purpose:** Provides a helpful message when no restaurants match initial criteria
    - **Methods:**
      - `exec(_)`: Returns a fixed message suggesting broader search criteria
      - `post(shared, _, exec_res)`: Stores message as final_response
    - **UX Improvement:** Suggests specific actions user can take (broader area, different cuisine)

## 5. Shared Store Design

The shared store is a dictionary that serves as the communication mechanism between nodes. It evolves during the flow execution, with each node reading from and writing to this shared state.

### Initial State
```python
shared = {
    "user_query": "Find me a great Italian place in Soho for a date night...",  # Raw user input
    "user_geolocation": {                    # User's current physical location (auto-detected)
        "success": True,
        "address": "London, UK",
        "lat": 51.5074,
        "lng": -0.1278,
        "neighborhood": "Holborn"
    },
    # The following fields are initially None/empty and populated during execution
    "parsed_query": None,
    "candidate_restaurants": None,
    "reviews_data": None,
    "dimensional_scores_1_10": None,
    "ranked_recommendations": None,
    "final_response": None
}
```

### State Evolution During Flow Execution

1. **After ParseUserQueryNode:**
   ```python
   shared["parsed_query"] = {
       "location": "Soho",             # Restaurant search location (user-specified)
       "cuisine": ["Italian"],
       "budget_pp": [50, 70],
       "vibe": "date night",
       "priorities": ["Ambiance", "Service"]
   }
   ```

2. **After FindRestaurantsNode:**
   ```python
   shared["candidate_restaurants"] = [
       {
           "id": "ChIJ123...", 
           "name": "Restaurant A",
           "distance_info": {          # Distance from user's physical location (not search location)
               "distance_km": 1.5,
               "distance_miles": 0.93,
               "travel_time_driving": 8,   # minutes
               "travel_time_transit": 15,  # minutes 
               "travel_time_walking": 18   # minutes
           },
           # Other restaurant details...
       },
       # More restaurants...
   ]
   ```

3. **After FetchReviewsNode:**
   ```python
   shared["reviews_data"] = {
       "ChIJ123...": [{"text": "Great pasta...", "rating": 5, ...}, ...],
       "ChIJ456...": [{"text": "Terrible service...", "rating": 2, ...}, ...],
       # More reviews per restaurant...
   }
   ```

4. **After AnalyzeReviewsBatchNode:**
   ```python
   shared["dimensional_scores_1_10"] = {
       "ChIJ123...": {"Taste": 8.5, "Service": 7.2, "Ambiance": 9.0, ...},
       "ChIJ456...": {"Taste": 7.8, "Service": 6.5, "Ambiance": 8.2, ...},
       # More scores per restaurant...
   }
   ```

5. **After CalculateFitScoreNode:**
   ```python
   shared["ranked_recommendations"] = [
       {
           "id": "ChIJ123...",
           "name": "Restaurant A",
           "scores": {"Taste": 8.5, "Service": 7.2, "Ambiance": 9.0, ...},
           "fit_score": 8.7
       },
       # More ranked restaurants...
   ]
   ```

6. **After GenerateResponseNode:**
   ```python
   shared["final_response"] = "# Restaurant Recommendations\n\nBased on your request for...\n\n| Restaurant | Taste | ... |..."
   ```

## 6. Implementation (`flow.py`, `main.py`)

### `flow.py` - Node Connections and Flow Creation

*   **Purpose:** Defines the flow graph by connecting nodes and specifying branching logic
*   **Key Function:** `create_recommendation_flow()`
    - Instantiates all nodes
    - Connects nodes with appropriate transitions (default and named actions)
    - Returns a Flow object with ParseUserQueryNode as the starting point
*   **Implementation Details:**
    - Uses PocketFlow's `>>` operator for default transitions (e.g., `node_a >> node_b`)
    - Uses `-` operator for named transitions (e.g., `node_a - "action_name" >> node_b`)
    - Contains placeholders for future functionality (clarification, basic list)

### `main.py` - Application Entry Point

*   **Purpose:** Initializes application, handles user input, runs flow, and displays results
*   **Key Functions:**
    - `main()`: Entry point, gets user input, calls run_recommendation
    - `run_recommendation(user_query)`: Initializes shared store, creates flow, runs it, returns result
*   **Implementation Details:**
    - Loads environment variables from .env file
    - Checks for required API keys
    - Sets up logging
    - Handles command-line arguments or interactive input
    - Provides error handling for flow execution

## 7. Conversation Handling (`conversation_nodes.py`)

### ConversationManagerNode

*   **Purpose:** Manages multi-turn conversation to collect all necessary information for restaurant recommendations.
*   **Methods:**
    - **`prep(shared)`**: Initializes or retrieves conversation state and user input
    - **`exec(prep_res)`**: Processes conversation, extracts information, and determines next stage
    - **`post(shared, prep_res, exec_res)`**: Updates shared store and transitions to recommendation or continuation

*   **Key Features:**
    - **Location Handling:** 
        - Automatically detects the user's current physical location via geolocation
        - Stores this as `user_geolocation` for distance calculations
        - Explicitly asks the user for the restaurant search location (`location`)
        - Maintains clear separation between these two location concepts
    - **Information Extraction:**
        - Uses LLMs to extract entities from natural language
        - Prioritizes location, cuisine, and budget in information gathering
        - Tracks missing information and requests it in follow-up questions
    - **Conversation Flow:**
        - Moves through stages: initialize → information_gathering → confirmation → generate_recommendations
        - Provides clear confirmation of extracted preferences
        - Generates natural language requests for missing information

*   **Information Storage:**
    - Conversation state contains:
        ```python
        conv_state = {
            "conversation_history": [...],            # All messages
            "extracted_info": {
                "location": "Soho",                   # Restaurant search location
                "user_geolocation": {...},            # Auto-detected physical location
                "cuisine": "Italian",                 # Cuisine preference
                "budget": 60,                         # Budget per person
                # Other preferences...
            },
            "current_stage": "information_gathering",
            "missing_info": ["location", "cuisine"]   # Info still needed
        }
        ```

### ConversationContinuationNode

*   **Purpose:** Handles continuation of conversation after initial information gathering
*   **Methods:**
    - **`prep(shared)`**: Retrieves conversation state and user input
    - **`exec(prep_res)`**: Processes conversation, extracts information, and determines next stage
    - **`post(shared, prep_res, exec_res)`**: Updates shared store and transitions to recommendation or continuation

*   **Key Features:**
    - **Information Extraction:**
        - Uses LLMs to extract additional information from natural language
        - Tracks missing information and requests it in follow-up questions
    - **Conversation Flow:**
        - Moves through stages: information_gathering → confirmation → generate_recommendations
        - Provides clear confirmation of extracted preferences
        - Generates natural language requests for missing information

*   **Information Storage:**
    - Conversation state contains:
        ```python
        conv_state = {
            "conversation_history": [...],            # All messages
            "extracted_info": {
                "location": "Soho",                   # Restaurant search location
                "user_geolocation": {...},            # Auto-detected physical location
                "cuisine": "Italian",                 # Cuisine preference
                "budget": 60,                         # Budget per person
                # Other preferences...
            },
            "current_stage": "information_gathering",
            "missing_info": ["location", "cuisine"]   # Info still needed
        }
        ```

## 8. Optimization / Future Work

### Performance Optimizations

*   **Parallel Processing:** Implement AsyncParallelBatchNode for concurrent review analysis
*   **Caching Strategies:**
    - Cache Google Maps API results for common locations
    - Cache LLM parsing results for similar queries
    - Cache review analyses with TTL (time-to-live)
*   **API Efficiency:**
    - Implement token counting to optimize prompt sizes
    - Use compression for large review sets
    - Batch Google API requests where possible

### Feature Enhancements

*   **Smarter Response Generation:**
    - Include specific restaurant highlights based on reviews
    - Generate personalized tips based on priorities (e.g., "Ask for a table near the window")
    - Add sentiment analysis for "vibe" detection
*   **Enhanced Decision Making:**
    - Expand Flock Agent capabilities to more branching paths
    - Implement follow-up question generation
    - Add more sophisticated handling of edge cases
*   **User Interaction:**
    - Add interactive refinement of recommendations
    - Implement feedback mechanism for recommendations
    - Create conversational history for context

### Architectural Improvements

*   **Data Persistence:**
    - Add database integration for caching and history
    - Implement result storage for analytics
*   **Modular Extensions:**
    - Create pluggable modules for different cuisine specialists
    - Develop location-specific knowledge modules
*   **API Abstraction:**
    - Create provider-agnostic interfaces for LLM and Maps services
    - Support multiple LLM providers (OpenAI, Anthropic, etc.)

## 9. Reliability

### Error Handling & Recovery

*   **Node-Level Retries:**
    - `AnalyzeReviewsBatchNode`: max_retries=2, wait=5 for LLM API calls
    - `DecideActionNode`: max_retries=1, wait=3 for Flock API calls
    - Individual retry counters via `self.cur_retry` property
*   **Graceful Degradation:**
    - `ParseUserQueryNode`: Falls back to text-based parsing if LLM fails
    - `AnalyzeReviewsBatchNode`: Continues with partial results if some analyses fail
    - `NoCandidatesFoundNode`: Provides helpful response when no restaurants match

### Input Validation

*   **Query Validation:**
    - Check for minimum viable input (location, cuisine)
    - Normalize and validate location formats
*   **API Response Validation:**
    - Validate Google Maps responses for expected fields
    - Validate LLM outputs for correct JSON format
    - Score range validation (1-10 for dimensions)

### Comprehensive Logging

*   **Node-Level Logging:**
    - Entry/exit logging for each node
    - Timing information for performance analysis
    - Input/output logging for debugging
*   **Error Tracing:**
    - Detailed exception logs with context
    - API call failure logging with request details
    - Recovery attempt tracking 

## 10. Recent Enhancements

### Location Handling Improvements

*   **User Location vs. Restaurant Location Distinction:**
    - Refactored the system to clearly distinguish between:
      - User's current physical location (automatically detected using IP geolocation)
      - Restaurant search location (explicitly provided by the user)
    - Improved conversation flow to specifically ask for the restaurant search location
    - Enhanced display of distance information with clear labeling "X km from your current location"

*   **Geolocation Implementation:**
    - Enhanced `get_user_geolocation()` with better error handling and logging
    - Improved storage of geolocation data in the shared state
    - Added clear separation between location concepts in all nodes and LLM prompts

### Opening Hours Display

*   **Enhanced Restaurant Hours Information:**
    - Added proper formatting and display of restaurant opening hours
    - Implemented "Closing Soon" warnings for restaurants that will close within 60 minutes
    - Added clear status indicators for restaurants that may already be closed
    - Improved time formatting for better readability in the final response

### Review Information

*   **Review Count Display:**
    - Added explicit tracking of the number of reviews analyzed per restaurant
    - Now showing review count in the final recommendations ("Based on analysis of X reviews")
    - Enhanced LLM prompt to ensure review counts are always displayed
    - Fixed formatting of review-based insights in the final response 

### Scoring System Improvements

*   **Consistent 10-Point Scoring Scale:**
    - Maintained the original 10-point scoring system throughout the application instead of converting to a 5-point scale
    - Added star emojis (⭐) for visual enhancement while preserving the precise numerical scores
    - Improved score descriptions for clearer interpretation (e.g., "Taste: 9.0 ⭐(Exceptional)")
    - Enhanced tabular output format for easier comparison of restaurant scores

*   **Visual Formatting Enhancements:**
    - Implemented consistent spacing in tabular output for better readability
    - Added visual indicators like checkmarks (✅) for highlighted features
    - Ensured proper alignment of scores and descriptions in the final response
    - Created backup template response system to guarantee consistent formatting regardless of LLM capabilities

## 11. Upcoming Features

### Commuting Information

*   **Enhanced Travel Information:**
    - Planned integration of commuting methods (walking, public transport, driving, cycling)
    - Will display estimated travel times for each transportation method
    - Will highlight recommended commuting options based on weather, time of day, and distance
    - Will indicate public transport availability and frequency

*   **Interactive Maps:**
    - Future addition of interactive map links for directions
    - Planning to include nearby parking information when relevant
    - Will suggest optimal routes considering traffic conditions and public transport schedules
    - Will display transportation costs estimates when available 