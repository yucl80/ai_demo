from openai import OpenAI
import os
import time

base_url = "http://127.0.0.1:8000/v1/"

client = OpenAI(base_url=base_url,api_key="NOKEY")

model_id = "functionary"

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": 'I want to plan a 7-day trip to Paris with a focus on art and culture'},
    ], 
    tools=[
        {
            "type": "function",
            "function": {
                "name": "plan_trip",
                "description": "Plan a trip based on user's interests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "The destination of the trip",
                        },
                        "duration": {
                            "type": "integer",
                            "description": "The duration of the trip in days",
                        },
                        "interests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The interests based on which the trip will be planned",
                        },
                    },
                    "required": ["destination", "duration", "interests"],
                }
            }
        }    
    ]
)

print(response)

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            "role": "user", 
            "content": 'What is the estimated value of a 3-bedroom house in San Francisco with 2000 sq ft area?'
        },
        {
            "role": "system", 
            "content": "", 
            "tool_calls": [
                {
                    "type": "function", 
                    "function": {
                        "name": "estimate_property_value", 
                        "arguments": '{\n  "property_details": {"location": "San Francisco", "size": 2000, "rooms": 3}\n}'
                    }
                }
            ]
        }
    ], 
    tools=[
        {
            "type": "function",
            "function": {
                "name": "estimate_property_value",
                "description": "Estimate the market value of a property",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "property_details": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The location of the property"
                                },
                                "size": {
                                    "type": "integer",
                                    "description": "The size of the property in square feet"
                                },
                                "rooms": {
                                    "type": "integer",
                                    "description": "The number of rooms in the property"
                                }
                            },
                            "required": ["location", "size", "rooms"]
                        }
                    },
                    "required": ["property_details"]
                }
            }
        }
    ],
    tool_choice="auto"
)

print(response)

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": 'My internet has been disconnecting frequently for the past week'},
    ], 
    tools=[
        {
            "type": "function",
            "function": {
            "name": "parse_customer_complaint",
            "description": "Parse a customer complaint and identify the core issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "complaint": {
                        "type": "object",
                        "properties": {
                            "issue": {
                                "type": "string",
                                "description": "The main problem",
                            },
                            "frequency": {
                                "type": "string",
                                "description": "How often the issue occurs",
                            },
                            "duration": {
                                "type": "string",
                                "description": "How long the issue has been occurring",
                            },
                        },
                        "required": ["issue", "frequency", "duration"],
                    },
                },
                "required": ["complaint"],
            }
        }
     }
    ],
    tool_choice="auto"
)

print(response)

