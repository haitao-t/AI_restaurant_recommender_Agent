# Restaurant Recommendation System

A conversational agent that recommends restaurants based on user preferences using LLM technology. The system processes natural language conversation to understand user preferences and provides personalized restaurant recommendations.

## Features

- Conversational interface for gathering user preferences
- Support for specifying cuisine type, location, price range, and more
- Restaurant search via Google Maps API
- Analysis of reviews to score restaurants on key dimensions (Taste, Service, Ambiance, Value)
- Personalized fit scoring based on user priorities
- Support for ASI-1 Mini or OpenAI as the LLM provider

## Setup

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
# ASI-1 Mini API Key - Get from Fetch.ai
ASI1_API_KEY=your_asi1_api_key_here

# OpenAI API Key (Optional fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Google Maps API Key (for location and restaurant data)
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# API Selection Configuration
# Set to "true" to use ASI-1 Mini, "false" for OpenAI
USE_ASI1=true

# Default model to use
# For ASI-1 Mini, use "asi1-mini"
# For OpenAI, use model name like "gpt-4o-mini" or "gpt-4o"
DEFAULT_MODEL=asi1-mini
```

## Running the Application

Start the recommendation system with:

```bash
python -m food_recommender_asi1.main
```

When prompted, enter your restaurant preferences in natural language. For example:
- "I'm looking for Italian food in Soho, mid-range price"
- "Chinese food, 2 people, 20 pounds per person, 6pm, good service and taste, in London"

The system will analyze your request and provide restaurant recommendations based on your preferences, including review analysis to match your priorities.

## Technical Implementation

The system is built using:

- PocketFlow for the LLM workflow orchestration
- Google Maps API for restaurant search and reviews
- ASI-1 Mini or OpenAI for natural language processing
- Advanced JSON parsing for handling diverse API responses
- Robust error handling and fallback mechanisms

## Fallback Mechanisms

If Google Maps API access is unavailable, the system will use simulated data to demonstrate functionality. This allows testing without valid API keys.

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

- Enhanced JSON parsing to handle different response formats from LLMs
- Fixed review processing to handle both string and dictionary formats
- Improved error handling for API failures
- Better user experience with formatted restaurant recommendations 