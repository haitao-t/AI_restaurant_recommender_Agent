# This file makes the utils directory a Python package

# Optionally, you can import key utility functions here for easier access
from .google_maps_api import find_restaurants, get_reviews_for_restaurant
from .call_finetuned_analyzer import analyze_reviews_with_finetuned_model
# Import the call_llm module itself, not specific functions from it
from . import call_llm

# Define a fallback function for decide_next_action_with_flock
def decide_next_action_with_flock(*args, **kwargs):
    """Fallback function that always returns 'analyze' action"""
    import logging
    logging.warning("Using dummy implementation for decide_next_action_with_flock")
    return {"name": "analyze", "arguments": {}}

# Try to import the real function if torch is available, but don't error if it's not
try:
    # Check if torch is available without trying to import it
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        # Only import if torch exists
        from .call_flock_agent import decide_next_action_with_flock
except (ImportError, ModuleNotFoundError):
    # Already defined fallback above, so we can just pass
    pass

# You can also define package-level variables or configurations if needed
# DEFAULT_MAX_REVIEWS = 20 