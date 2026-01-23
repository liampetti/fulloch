"""
Centralized tool registry with schema definitions.

This module provides a unified interface for tool registration and schema management
using the model context protocol for function calling.
"""

import inspect
import json
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Parameter types for function schemas."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterSchema:
    """Schema definition for a function parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None


@dataclass
class FunctionSchema:
    """Schema definition for a tool function."""
    name: str
    description: str
    parameters: List[ParameterSchema]
    returns: str


class ToolRegistry:
    """Centralized registry for all available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, FunctionSchema] = {}
        self._aliases: Dict[str, str] = {}
    
    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        aliases: Optional[List[str]] = None
    ) -> Callable:
        """
        Decorator to register a tool function with schema information.
        
        Args:
            func: The function to register
            name: Function name (defaults to func.__name__)
            description: Function description
            aliases: List of alias names for this function
        """
        func_name = name or func.__name__
        
        # Extract parameter information
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Determine parameter type
            param_type = ParameterType.STRING  # Default
            if param.annotation == int:
                param_type = ParameterType.INTEGER
            elif param.annotation == float:
                param_type = ParameterType.FLOAT
            elif param.annotation == bool:
                param_type = ParameterType.BOOLEAN
            elif param.annotation == list:
                param_type = ParameterType.ARRAY
            
            # Get parameter description from docstring
            param_desc = f"Parameter {param_name}"
            
            # Check if parameter has default
            required = param.default == inspect.Parameter.empty
            
            parameters.append(ParameterSchema(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=required,
                default=param.default if not required else None
            ))
        
        # Create function schema
        schema = FunctionSchema(
            name=func_name,
            description=description or func.__doc__ or f"Function {func_name}",
            parameters=parameters,
            returns="string"
        )
        
        # Register the function
        self._tools[func_name] = func
        self._schemas[func_name] = schema
        
        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = func_name
        
        logger.info(f"Registered tool: {func_name}")
        return func
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool function by name (including aliases)."""
        # Check direct name
        if name in self._tools:
            return self._tools[name]
        
        # Check aliases
        if name in self._aliases:
            return self._tools[self._aliases[name]]
        
        return None
    
    def get_schema(self, name: str) -> Optional[FunctionSchema]:
        """Get schema for a tool function."""
        if name in self._schemas:
            return self._schemas[name]
        
        # Check aliases
        if name in self._aliases:
            return self._schemas[self._aliases[name]]
        
        return None
    
    def get_all_schemas(self) -> List[FunctionSchema]:
        """Get all available function schemas."""
        return list(self._schemas.values())
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all available tools (including aliases)."""
        tools = self._tools.copy()
        # Add aliases
        for alias, original_name in self._aliases.items():
            tools[alias] = self._tools[original_name]
        return tools
    
    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI function calling schema format."""
        schemas = []
        
        for schema in self._schemas.values():
            # Convert parameters to OpenAI format
            properties = {}
            required_params = []
            
            for param in schema.parameters:
                properties[param.name] = {
                    "type": param.type.value,
                    "description": param.description
                }
                
                if param.enum:
                    properties[param.name]["enum"] = param.enum
                
                if param.default is not None:
                    properties[param.name]["default"] = param.default
                
                if param.required:
                    required_params.append(param.name)
            
            openai_schema = {
                "name": schema.name,
                "description": schema.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                }
            }
            
            schemas.append(openai_schema)
        
        return schemas
    
    def execute_tool(self, name: str, args: List[Any] = None, kwargs: Dict[str, Any] = None) -> str:
        """Execute a tool function."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        
        args = args or []
        kwargs = kwargs or {}
        
        try:
            result = tool(*args, **kwargs)
            return result if result is not None else ""
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing {name}: {str(e)}"


# Global tool registry instance
tool_registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    aliases: Optional[List[str]] = None
):
    """Decorator to register a tool function."""
    def decorator(func):
        return tool_registry.register_tool(func, name, description, aliases)
    return decorator 