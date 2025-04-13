# Restaurant Recommendation System

A conversational agent that recommends restaurants based on user preferences using LLM technology. The system processes natural language conversation to understand user preferences and provides personalized restaurant recommendations.

## Features

- Conversational interface for gathering user preferences
- Support for specifying cuisine type, location, price range, and more
- Restaurant search via Google Maps API
- **NEW:** Analysis of reviews using a local fine-tuned model (`c0sm1c9/restaurant-review-analyzer-dutch`) to score restaurants on key dimensions (Taste, Service, Ambiance)
- Personalized fit scoring based on user priorities
- Support for ASI-1 Mini or OpenAI as the LLM provider
- **NEW:** Web3-related information for restaurants (cryptocurrency payments, token-based loyalty programs, NFT rewards)
- **NEW:** Aipolabs API integration for enhanced restaurant information (TAVILY web search)
- **NEW:** Using GPT-4o-mini for all decision logic with Flock model focused on Web3 processing
- **NEW:** Improved conversation flow with intelligent time handling and error recovery
- **NEW:** Enhanced logging system with configurable log levels

## Setup

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(Ensure you have PyTorch installed if using a GPU for local model inference)*

3. Create a `.env` file in the project root with your API keys:
```
# ASI-1 Mini API Key - Get from Fetch.ai
ASI1_API_KEY=your_asi1_api_key_here

# OpenAI API Key (Required for decision making)
OPENAI_API_KEY=your_openai_api_key_here

# Google Maps API Key (for location and restaurant data)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Aipolabs API Key (for enhanced restaurant information)
AIPOLABS_ACI_API_KEY=your_aipolabs_api_key_here

# Aipolabs linked account ID (required for API calls)
LINKED_ACCOUNT_OWNER_ID=your_linked_account_id_here

# API Selection Configuration
# Set to "true" to use ASI-1 Mini, "false" for OpenAI
USE_ASI1=true

# Set to "true" to enable Aipolabs API integration, "false" to disable
USE_ACI_DEV=true

# Set to "true" to use Flock model for Web3 information, "false" for OpenAI
USE_FLOCK_MODEL=false

# Default model to use
# For ASI-1 Mini, use "asi1-mini"
# For OpenAI, use model name like "gpt-4o-mini" or "gpt-4o"
DEFAULT_MODEL=asi1-mini

# Logging configuration
LOG_LEVEL=INFO
LOGS_DIR=logs
```
*(Note: `FINETUNED_MODEL_API_URL` and `FINETUNED_MODEL_API_KEY` are no longer needed as review analysis is done locally.)*

## Running the Application

Start the recommendation system with:

```bash
python main.py
```

When prompted, enter your restaurant preferences in natural language. For example:
- "I'm looking for Italian food in Soho, mid-range price"
- "Chinese food, 2 people, 20 pounds per person, 6pm, good service and taste, in London"
- "Find restaurants in downtown that accept cryptocurrency payments"
- "Find Japanese restaurants in Tribeca with recent reviews and information"

The system will analyze your request and provide restaurant recommendations based on your preferences, including review analysis to match your priorities.

## Conversation Flow Management

The system now features an improved conversation flow that:

- Automatically detects user's current location for distance calculations
- Sets current time and date as defaults for dining plans
- Intelligently extracts information from user messages
- Supports multi-turn conversations with context maintenance
- Handles errors gracefully with user-friendly recovery
- Allows natural follow-up questions about recommended restaurants

## Web3 Features

The system can use either OpenAI's GPT-4o-mini or the Flock Web3 Agent Model to determine if restaurants might offer Web3-related features, such as:

- **Cryptocurrency Payments**: Identifies restaurants that likely accept crypto as a payment method
- **Token-Based Loyalty Programs**: Highlights restaurants with token-based rewards or membership programs
- **NFT Rewards & Collectibles**: Detects restaurants that offer NFT-based collectibles or special access

To request Web3 information in your search, simply include relevant terms in your query:
- "Restaurants that accept crypto in San Francisco"
- "Places with NFT rewards in New York"
- "Italian restaurants with token programs"

The system will automatically determine whether to include Web3 information based on your query and the restaurants found.

## Aipolabs API Integration

The system now integrates with Aipolabs API to provide enhanced information about restaurants:

- **Web Search Results**: Uses TAVILY to find relevant information about restaurants online
- **Future Support**: The system is designed to easily incorporate other Aipolabs services like COINMARKETCAP and FIRECRAWL as needed

To get additional information in your search results, include terms like "information", "reviews", or "details" in your query:
- "Find detailed information about Italian restaurants in Brooklyn"
- "I want the best sushi places in Chicago with recent reviews"
- "Show me family-friendly restaurants in Austin with additional information"

## Technical Implementation

The system is built using:

- PocketFlow for the LLM workflow orchestration
- Google Maps API for restaurant search and reviews
- **NEW:** Local fine-tuned model (`c0sm1c9/restaurant-review-analyzer-dutch`) via `transformers` and `torch` for review analysis
- ASI-1 Mini or OpenAI for natural language processing
- OpenAI's GPT-4o-mini for all decision-making logic
- Flock Web3 Agent Model (optional) for Web3 feature analysis
- Aipolabs API (TAVILY) for enhanced restaurant information
- Advanced conversation management with state tracking and entity extraction
- Comprehensive error handling and logging system
- Advanced JSON parsing for handling diverse API responses

## Logging System

The system implements a robust logging system that:

- Records all operations to both console and log files
- Creates a dedicated logs directory for persistent storage
- Allows configuration of logging level via environment variables
- Captures and documents any errors for easier troubleshooting

## Fallback Mechanisms

If Google Maps API, Aipolabs API, or other external API access is unavailable, the system will use simulated data or continue without those features. This allows testing without valid API keys.

## Switching LLM Providers

You can switch between ASI-1 Mini and OpenAI by changing the environment variables in your `.env` file:

```
# Use ASI-1 Mini
USE_ASI1=true
DEFAULT_MODEL=asi1-mini

# Use OpenAI
USE_ASI1=false
DEFAULT_MODEL=gpt-4o-mini
```

## Recent Improvements

- **NEW:** Integrated local fine-tuned model (`c0sm1c9/restaurant-review-analyzer-dutch`) for review analysis, removing external API dependency.
- **NEW:** Enhanced conversation management with multi-turn dialog support, entity extraction, and state tracking.
- **NEW:** Automatic time and location processing for more intuitive interaction.
- **NEW:** Robust error handling with user-friendly recovery mechanisms.
- **NEW:** Configurable logging system for easier troubleshooting.
- **NEW:** Separated decision logic to use OpenAI GPT-4o-mini for all decision making.
- **NEW:** Flock model now focused exclusively on Web3 information (when enabled).
- Added Aipolabs API integration with TAVILY for enhanced web search information.
- Added Web3 feature detection with dedicated processing node.
- Enhanced JSON parsing to handle different response formats from LLMs.
- Fixed review processing to handle both string and dictionary formats.
- Improved error handling for API failures.
- Better user experience with formatted restaurant recommendations.
- Corrected various indentation issues identified by linters.

# Potential Business Models for AI Restaurant Recommender

The core value proposition of this application lies in providing **highly detailed, personalized, and deeply analyzed restaurant recommendations** that surpass traditional star ratings, addressing user pain points regarding information overload and rating simplicity. Based on this value, several business models can be considered:

## 1. Direct-to-Consumer (B2C) Models

* **Freemium/Subscription Model:**
  * Offer basic recommendation features for free to attract a user base.
  * Charge a recurring fee (monthly/annually) for premium features such as:
    * Advanced personalization (e.g., recommendations based on historical preferences).
    * Unlimited "deep analysis" requests.
    * Highly specific filters (e.g., "quiet corners for business," "allergy-friendly options").
    * Integrated booking capabilities.
    * Ad-free experience.
* **One-Time Purchase (Premium App):**
  * Offer a paid "Pro" version containing all premium features. (Less common for apps requiring continuous data updates).

## 2. Business-to-Business/Business-to-Consumer (B2B/B2B2C) Models - Targeting Restaurants/Merchants

* **Premium Listings & Enhanced Profiles:**
  * Restaurants pay for richer profile pages (e.g., full menus, high-quality photos, direct booking links).
  * Offer sponsored placements in search results for highly relevant user queries (must be clearly labeled as "Sponsored" or "Promoted" to maintain user trust).
* **Data Analytics & Insights Dashboard (SaaS):**
  * Sell restaurants detailed analytical reports based on LLM analysis of their customer reviews. Provide insights into specific aspects like service details, dish feedback, and ambiance perception. This offers valuable business intelligence for operational improvements.
* **Booking Commissions:**
  * Integrate with online booking systems (e.g., OpenTable, Resy, or proprietary systems). Charge restaurants a commission fee for each booking completed through the app.
* **Targeted Promotions & Offers:**
  * Allow restaurants to push customized promotions or coupons specifically to users whose search criteria closely match the restaurant's offerings (e.g., a user searching for "romantic Italian with vegetarian options" might receive a relevant offer).

## 3. Technology Licensing & API Services (B2B)

* **API Licensing:**
  * License the fine-tuned review analysis LLM or the entire agent-based recommendation engine via API to other platforms (e.g., map services, review sites, travel portals, booking platforms). Charge based on API call volume or a recurring license fee.
* **White-Label Solution:**
  * Offer the complete application and technology stack as a white-label product for large restaurant chains, hospitality groups, or regional food platforms to operate under their own brand.

## 4. Advertising & Affiliate Marketing

* **Relevant Advertising:**
  * Display highly relevant and non-intrusive ads within the user journey (e.g., ads for nearby bars after a restaurant recommendation, ride-sharing service ads). Requires careful implementation to avoid degrading the user experience.
* **Affiliate Partnerships:**
  * Partner with complementary services (e.g., food delivery, taxi services). Earn commissions when users book these services via links within the app after finding a restaurant.

## Summary & Recommendation

A **hybrid approach** is likely most effective:
* Use a **freemium model** to build a large user base.
* Monetize through **user subscriptions** for premium features and convenience.
* Focus heavily on **B2B offerings**, particularly the unique **data analytics dashboard** for restaurants and targeted promotions.
* Explore **API licensing** to scale the technology's reach.

The key is to balance monetization with user experience, maintaining transparency, especially regarding sponsored content, to leverage the core competitive advantage: the quality and intelligence of the recommendations.
