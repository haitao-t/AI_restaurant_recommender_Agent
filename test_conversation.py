#!/usr/bin/env python3
"""
Test script for conversation entity extraction
"""
import json
import logging
from utils import call_llm
from conversation_nodes import ConversationManagerNode

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock the LLM call to return a consistent response
def mock_llm_call(prompt):
    if "I want Italian in Holborn for 4 people with 20 pounds per person" in prompt:
        return json.dumps({
            "location": "Holborn",
            "cuisine": "Italian",
            "budget": 20,
            "occasion": None,
            "time": None,
            "priorities": [],
            "group_size": 4,
            "dietary_preferences": None
        })
    # Add more mock cases as needed
    return "{}"

# Create test cases
test_cases = [
    {
        "name": "Multiple entities in one message",
        "input": "I want Italian in Holborn for 4 people with 20 pounds per person",
        "expected": {
            "location": "Holborn",
            "cuisine": "Italian", 
            "budget": 20,
            "group_size": 4
        }
    },
    # Add more test cases as needed
]

def run_tests():
    """Run all test cases"""
    # Replace LLM call with mock
    original_llm_call = call_llm.call_llm
    call_llm.call_llm = mock_llm_call
    
    try:
        node = ConversationManagerNode()
        node.missing_info = []
        
        for test_case in test_cases:
            logger.info(f"Running test: {test_case['name']}")
            test_input = test_case["input"]
            expected = test_case["expected"]
            
            current_info = {
                "location": None, 
                "cuisine": None, 
                "budget": None,
                "occasion": None, 
                "time": None, 
                "priorities": [],
                "confirmed": False
            }
            
            # Test extraction
            result = node._extract_entities(test_input, current_info)
            
            # Log results
            logger.info(f"Input: {test_input}")
            logger.info(f"Expected: {json.dumps(expected)}")
            logger.info(f"Actual: {json.dumps(result)}")
            
            # Check if all expected values are in the result
            success = True
            for key, value in expected.items():
                if key not in result or result[key] != value:
                    success = False
                    logger.error(f"❌ Test failed for key '{key}': expected '{value}', got '{result.get(key)}'")
            
            if success:
                logger.info(f"✓ Test passed: {test_case['name']}")
            else:
                logger.error(f"❌ Test failed: {test_case['name']}")
    
    finally:
        # Restore original LLM call
        call_llm.call_llm = original_llm_call

if __name__ == "__main__":
    run_tests() 