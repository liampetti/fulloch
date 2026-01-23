"""
Intent handler using the centralized tool registry.

This module provides intent handling functionality using the new
tool registry system with backward compatibility.
"""

import json
import logging
from typing import Dict, Any, Optional

from tools.tool_registry import tool_registry

logger = logging.getLogger(__name__)


class IntentHandler:
    """Centralized intent handler using the tool registry."""
    
    def __init__(self):
        self.logger = logging.getLogger("IntentHandler")
    
    def get_available_functions(self):
        """Get all available functions in OpenAI format."""
        return tool_registry.to_openai_schema()
    
    def get_function_descriptions(self) -> str:
        """Get human-readable descriptions of all available functions."""
        descriptions = []
        for schema in tool_registry.get_all_schemas():
            desc = f"- {schema.name}: {schema.description}"
            if schema.parameters:
                params = []
                for param in schema.parameters:
                    param_desc = f"{param.name}"
                    if not param.required:
                        param_desc += " (optional)"
                    if param.default is not None:
                        param_desc += f" (default: {param.default})"
                    params.append(param_desc)
                desc += f" - Parameters: {', '.join(params)}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def handle_intent(self, intent_data: Dict[str, Any]) -> str:
        """
        Handle an intent using the tool registry.
        
        Args:
            intent_data: Dictionary containing function call information
            
        Returns:
            Result of the function execution
        """
        try:
            # Handle OpenAI function calling format
            if "function_call" in intent_data:
                func_call = intent_data["function_call"]
                function_name = func_call["name"]
                arguments = json.loads(func_call["arguments"])
                
                # Execute the function
                result = tool_registry.execute_tool(function_name, kwargs=arguments)
                return result
            
            # Handle legacy format
            elif "intent" in intent_data:
                intent_name = intent_data["intent"]
                args = intent_data.get("args", [])
                
                # Execute the function
                result = tool_registry.execute_tool(intent_name, args=args)
                return result
            
            else:
                self.logger.error(f"Invalid intent data format: {intent_data}")
                return ""
                
        except Exception as e:
            self.logger.exception(f"Error handling intent: {e}")
            return ""
    
    def validate_intent(self, intent_data: Dict[str, Any]) -> bool:
        """Validate that an intent can be executed."""
        try:
            if "function_call" in intent_data:
                func_call = intent_data["function_call"]
                function_name = func_call["name"]
                return tool_registry.get_tool(function_name) is not None
            
            elif "intent" in intent_data:
                intent_name = intent_data["intent"]
                return tool_registry.get_tool(intent_name) is not None
            
            return False
            
        except Exception:
            return False


# Global intent handler instance
intent_handler = IntentHandler()


def handle_intent(intent_json):
    """    
    Args:
        intent_json: Intent data in JSON format or dict
        
    Returns:
        Result of the intent execution
    """
    if isinstance(intent_json, str):
        try:
            intent_data = json.loads(intent_json)
        except json.JSONDecodeError:
            logger.error("Invalid JSON input")
            return "Sorry, I was unable to process this request"
    else:
        intent_data = intent_json
    
    return intent_handler.handle_intent(intent_data)


if __name__ == "__main__":
    # Test the intent handler
    print("Testing Intent Handler")
    print("=" * 50)
    
    # Print available functions
    print("\nAvailable functions:")
    print(intent_handler.get_function_descriptions())
    
    # Test function calling
    print("\nTesting function calls:")
    
    # Test with function call format
    test_intent = {
        "function_call": {
            "name": "get_temperature",
            "arguments": '{"location": "upstairs"}'
        }
    }
    
    result = intent_handler.handle_intent(test_intent)
    print(f"Function call result: {result}")
    
    # Test with legacy format
    legacy_intent = {
        "intent": "get_current_time",
        "args": []
    }
    
    result = intent_handler.handle_intent(legacy_intent)
    print(f"Legacy format result: {result}")
