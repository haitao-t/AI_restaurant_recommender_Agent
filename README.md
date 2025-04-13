# AI Restaurant Recommender Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/rr.png" alt="Restaurant Recommender Logo" width="300"/>
</p>

An AI-powered agent designed to provide personalized restaurant recommendations based on user preferences, location, and in-depth analysis of user reviews using a specialized fine-tuned language model.

## Features

*   **Natural Language Understanding:** Parses user requests in plain English to extract key details like location, cuisine, budget, vibe, and priorities.
*   **Google Maps Integration:**
    *   Finds relevant restaurants based on location and cuisine.
    *   Fetches up-to-date user reviews for candidate restaurants.
    *   Detects the user's current physical location (via IP geolocation) for distance/travel time calculations.
    *   Retrieves opening hours and status.
*   **Fine-tuned Review Analysis:** Utilizes a custom fine-tuned model ([`c0sm1c9/restaurant-review-analyzer-dutch`](https://huggingface.co/c0sm1c9/restaurant-review-analyzer-dutch)) to analyze reviews and assign scores (2-7) for key dimensions: **Taste**, **Service**, and **Ambiance**.
*   **Personalized Fit Score:** Calculates a weighted "Fit Score" (0-10) for each restaurant based on how well its dimensional scores align with the user's stated priorities.
*   **Structured Output:** Presents recommendations in a clear Markdown format, including:
    *   A comparison table with dimensional scores, fit scores, distance, price level, and review counts.
    *   Concise summaries linking recommendations to user priorities.
    *   Opening hours information, including "Closing Soon" warnings.
    *   A concluding suggestion.

## Architecture

The agent is designed as a graph of connected nodes, with each node performing a specific task and sharing data via a central store. This approach allows for modular, maintainable, and flexible application design.

```mermaid
graph LR
    A[Start: User Query] --> B[ParseUserQueryNode LLM]
    B --> C[FindRestaurantsNode Maps API]
    C -- default --> D[FetchReviewsNode Maps API]
    C -- no_candidates_found --> J[NoCandidatesFoundNode]
    D --> K{DecideActionNode Decision Logic}
    K -- analyze --> E[AnalyzeReviewsBatchNode Fine-tuned LLM]
    K -- clarify --> L[AskClarificationNode Placeholder]
    E --> F[CalculateFitScoreNode Python Logic]
    F --> G[GenerateResponseNode LLM Formatting]
    G --> H[End: Formatted Response]
    J --> H
    L --> H

    style K fill:#lightgrey,stroke:#333,stroke-width:2px
    style E fill:#f9d,stroke:#333,stroke-width:2px
    style F fill:#ccf,stroke:#333,stroke-width:1px
```

**Core Technologies:**

*   **Language:** Python 3.x
*   **LLMs:**
    *   General Purpose LLM (e.g., OpenAI GPT-4o-mini) for query parsing and response generation (via `utils/call_llm.py`).
*   **Review Analysis:** Local execution of a fine-tuned model using Hugging Face `transformers` and `torch` (via `utils/call_finetuned_analyzer.py`).
*   **APIs:** Google Maps Places API
*   **Libraries:** `transformers`, `torch`, `googlemaps`, `openai`, `requests`, `python-dotenv`, `ipinfo`

## Fine-tuned Review Analysis Model

A key component is the fine-tuned model [`c0sm1c9/restaurant-review-analyzer-dutch`](https://huggingface.co/c0sm1c9/restaurant-review-analyzer-dutch) hosted on Hugging Face.

*   **Base Model:** XLM-RoBERTa-Base
*   **Task:** Multi-head regression trained primarily on Dutch restaurant reviews.
*   **Output:** Predicts scores (2-7) for **Taste**, **Service**, and **Ambiance**.
*   **Scoring System:** The system implements an optimized scoring algorithm that ensures restaurant ratings have meaningful differentiation between dimensions and restaurants, avoiding the common problem of uniformly high scores.
*   **Integration:** The agent loads and runs this model **locally** using the Hugging Face `transformers` library and a custom Python class defined within the project. This requires installing the `transformers` and `torch` libraries.

### Recent Updates

* **Enhanced Scores Differentiation:** Optimized the scoring system to provide more distinctive and realistic ratings
* **Improved Visual Presentation:** Added more meaningful descriptions for different rating levels and distinctive highlights for restaurants
* **Simplified Architecture:** Removed unused dependencies and features to create a more streamlined experience

## Getting Started

### Prerequisites

*   Python 3.8+
*   Git
*   PyTorch (`torch`)
*   Hugging Face Transformers (`transformers`)
*   Access to Google Maps Places API
*   Access to an OpenAI API key (or compatible LLM provider)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/haitao-t/AI_restaurant_recommender_Agent.git
    cd AI_restaurant_recommender_Agent
    ```

2.  **Create and activate a virtual environment:**
    
    **For macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    
    **For Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This will install required libraries including `googlemaps`, `openai`, `requests`, `python-dotenv`, `ipinfo`, `torch`, `transformers`)*

4.  **Set up environment variables:**
    
    Copy the example environment file and edit it with your API keys:
    ```bash
    cp .env.example .env
    ```
    
    Then open `.env` in your preferred text editor and add your API keys:
    ```dotenv
    # .env file
    GOOGLE_MAPS_API_KEY="YOUR_GOOGLE_MAPS_API_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

    # Optional: Override default LLM model for general tasks
    # DEFAULT_LLM_MODEL="gpt-4o-mini"
    ```
    
    **API Keys Setup:**
    * Google Maps API key: Create or access from the [Google Cloud Console](https://console.cloud.google.com/apis/library/places-backend.googleapis.com). Make sure the Places API is enabled for your project.
    * OpenAI API key: Generate from the [OpenAI Platform](https://platform.openai.com/) by creating an account and navigating to the API keys section.

## Usage

Run the main application script:

```bash
python main.py
```

The application will prompt you to enter your restaurant request.

**Example Query:**

```
Find me a great Italian place in Soho for a date night, budget around £60pp, focus on ambiance and good service.
```

**Expected Output:**

The agent will process the request through the defined workflow and output a Markdown formatted response containing:

*   An introductory sentence acknowledging the request.
*   A comparison table of the top recommended restaurants, showing:
    *   Name
    *   Distance/Travel Time (from your current location)
    *   Price Level
    *   Dimensional Scores (Taste, Service, Ambiance) with descriptive ratings
    *   Calculated Fit Score
    *   Review Count Analyzed
    *   Opening Hours / Status
*   A concise summary highlighting why the recommendations fit the user's priorities.
*   A concluding remark.

## Project Structure

```
├── .env.example        # Example environment variables file
├── README.md           # This file
├── docs/
│   └── design.md       # Detailed design document
├── main.py             # Main application entry point
├── flow.py             # Defines the workflow graph
├── nodes.py            # Implementation of individual workflow nodes
├── requirements.txt    # Python dependencies
└── utils/
    ├── __init__.py
    ├── call_llm.py     # Utilities for calling the general LLM (OpenAI)
    ├── call_finetuned_analyzer.py # Utility for loading and running the local fine-tuned model
    └── google_maps_api.py # Utilities for interacting with Google Maps API
```

## Acknowledgements

This project was developed by:
*   Haitao Tao
*   Yihan Shen
*   Zegeng Zhu

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project uses [PocketFlow](https://github.com/The-Pocket/PocketFlow), which is also licensed under the MIT License.
