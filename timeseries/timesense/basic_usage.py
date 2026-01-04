"""
Basic usage examples for TimeSense MCP server
"""
from readline import clear_history

import numpy as np
from datetime import datetime, timedelta
import ts_server as tss
import asyncio
from openai import OpenAI, AsyncOpenAI
import os
import json
import logging


def generate_sample_data():
    """Generate sample time series data for testing"""

    # Example 1: Trending series with spike
    dates = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(100)]
    values = np.linspace(10, 30, 100) + np.random.normal(0, 2, 100)
    values[50] = 50  # Add a spike

    series_1 = {
        "name": "metric_a",
        "timestamps": [d.isoformat() for d in dates],
        "values": values.tolist()
    }

    # Example 2: Series with change point
    values_2 = np.concatenate([
        np.random.normal(10, 1, 40),  # Stable around 10
        np.random.normal(20, 1, 60),  # Jump to 20
    ])

    series_2 = {
        "name": "metric_b",
        "timestamps": [d.isoformat() for d in dates],
        "values": values_2.tolist()
    }

    # Example 3: Cyclical pattern
    t = np.arange(100)
    values_3 = 15 + 5 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.5, 100)

    series_3 = {
        "name": "metric_c",
        "timestamps": [d.isoformat() for d in dates],
        "values": values_3.tolist()
    }

    return series_1, series_2, series_3


# Example MCP tool calls (these would be called via MCP protocol)
EXAMPLE_REQUESTS = {
    "analyze_extreme": {
        "series": [generate_sample_data()[0]],
        "question": "What is the maximum value in this time series and where does it occur?",
        "task_type": "extreme",
        "verify": True
    },

    "analyze_trend": {
        "series": [generate_sample_data()[0]],
        "question": "What is the overall trend of this time series?",
        "task_type": "trend",
        "verify": True
    },

    "detect_spike": {
        "series": [generate_sample_data()[0]],
        "question": "Are there any spikes or anomalies in this time series?",
        "task_type": "spike",
        "verify": True
    },

    "detect_change_point": {
        "series": [generate_sample_data()[1]],
        "question": "Identify any change points where the series behavior changes significantly.",
        "task_type": "change_point",
        "verify": True
    },

    "segment_series": {
        "series": [generate_sample_data()[2]],
        "question": "Segment this time series into distinct phases.",
        "task_type": "segment",
        "verify": True,
        "window_size": 20
    },

    "compare_series": {
        "series" : [generate_sample_data()[0], generate_sample_data()[1]],
        "question": "Compare the behavior of these two time series.",

        "aspect": "trends and volatility"
    },

    "detect_anomalies": {
        "series": [generate_sample_data()[0], generate_sample_data()[1]],
        "interval": [40, 60],
        "anomaly_rules": [
            "Series 1: values above 45 are anomalous",
            "Series 2: sudden jumps > 5 are anomalous"
        ]
    },

    "comprehensive_analysis": {
        "series": [generate_sample_data()[0]],
        "question": "Provide a comprehensive analysis of this time series including trends, anomalies, and key patterns.",
        "task_type": "describe",
        "verify": False
    }
}


def print_example_requests():
    """Print example requests for documentation"""
    import json

    print("# Example MCP Tool Requests\n")

    for name, request in EXAMPLE_REQUESTS.items():
        print(f"## {name}\n")
        print("```json")
        print(json.dumps(request, indent=2))
        print("```\n")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

server = tss.TimeSeriesMCPServer()

client = AsyncOpenAI(api_key=os.getenv("OPEN_ROUTER_KEY"), base_url="https://openrouter.ai/api/v1")

# Define MCP tools in OpenAI function calling format
# Note: We simplify by not requiring series data in parameters - it will be provided from context
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_time_series",
            "description": "Analyze time series data. Use for: finding extremes (max/min), detecting spikes, identifying trends, finding change points, general descriptions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Analysis question or task description"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["auto", "extreme", "spike", "trend", "change_point", "describe"],
                        "description": "Type of analysis to perform"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_segments",
            "description": "Segment a time series into distinct phases and describe each segment with statistics",
            "parameters": {
                "type": "object",
                "properties": {
                    "window_size": {
                        "type": "integer",
                        "description": "Size of each segment window",
                        "default": 50
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect anomalies in time series data with optional custom rules",
            "parameters": {
                "type": "object",
                "properties": {
                    "interval": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [start, end] interval to focus on"
                    },
                    "anomaly_rules": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Custom anomaly detection rules"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_series",
            "description": "Compare two time series and identify similarities and differences",
            "parameters": {
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "description": "What aspect to compare (e.g., 'trends and volatility', 'overall behavior')",
                        "default": "overall behavior"
                    }
                },
                "required": []
            }
        }
    }
]

async def get_llm_response(input_data):
    """Get LLM response using MCP tools via OpenAI function calling"""

    # Extract question if present
    question = input_data.get("question", "Analyze the following time series data")

    # Determine which tool the LLM should use based on the request structure
    # This helps guide the LLM to call the right tool
    tool_hint = ""
    if "interval" in input_data or "anomaly_rules" in input_data:
        tool_hint = "\n\nUse the detect_anomalies tool to find anomalies in the data."
    elif "aspect" in input_data or (isinstance(input_data.get("series"), list) and len(input_data.get("series", [])) > 1):
        if "compare" in question.lower() or "aspect" in input_data:
            tool_hint = "\n\nUse the compare_series tool to compare the time series."
    elif "segment" in question.lower() or "phase" in question.lower():
        tool_hint = f"\n\nUse the describe_segments tool to segment the time series."
    else:
        tool_hint = f"\n\nUse the analyze_time_series tool with appropriate task_type."

    # Build a summary of the data (not full data to avoid large prompts)
    series_summary = ""
    if "series" in input_data:
        series_summary = "\n\nData provided:"
        for s in input_data["series"]:
            series_summary += f"\n- '{s['name']}': {len(s['values'])} data points"

    prompt = f"""{question}{series_summary}{tool_hint}"""

    messages = [{"role": "user", "content": prompt}]

    logger.info(f"Prompt: {prompt[:200]}...")

    # Call LLM with tools
    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Check if LLM wants to call a tool
    if response_message.tool_calls:
        # Add LLM response to messages
        messages.append(response_message)

        # Execute each tool call
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name

            # Parse tool arguments
            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (expected for complex data): {e}")
                function_args = {}

            logger.info(f"✓ LLM called MCP tool: {function_name}")
            logger.info(f"  Arguments: {function_args}")

            # Call the appropriate MCP server tool
            try:
                if function_name == "analyze_time_series":
                    # Build args from input_data
                    tool_args = {
                        "series": input_data.get("series", []),
                        "question": function_args.get("question", question),
                        "task_type": function_args.get("task_type", input_data.get("task_type", "auto")),
                        "verify": input_data.get("verify", True)
                    }
                    logger.info(f"  → Calling server._analyze_time_series")
                    result = await server._analyze_time_series(tool_args)
                    tool_result = result[0].text

                elif function_name == "describe_segments":
                    tool_args = {
                        "series": input_data["series"][0],  # First series
                        "window_size": function_args.get("window_size", input_data.get("window_size", 50))
                    }
                    logger.info(f"  → Calling server._describe_segments")
                    result = await server._describe_segments(tool_args)
                    tool_result = result[0].text

                elif function_name == "detect_anomalies":
                    tool_args = {
                        "series": input_data.get("series", []),
                        "interval": function_args.get("interval", input_data.get("interval")),
                        "anomaly_rules": function_args.get("anomaly_rules", input_data.get("anomaly_rules"))
                    }
                    logger.info(f"  → Calling server._detect_anomalies")
                    result = await server._detect_anomalies(tool_args)
                    tool_result = result[0].text

                elif function_name == "compare_series":
                    if len(input_data.get("series", [])) >= 2:
                        tool_args = {
                            "series_a": input_data["series"][0],
                            "series_b": input_data["series"][1],
                            "aspect": function_args.get("aspect", input_data.get("aspect", "overall behavior"))
                        }
                        logger.info(f"  → Calling server._compare_series")
                        result = await server._compare_series(tool_args)
                        tool_result = result[0].text
                    else:
                        tool_result = "Error: compare_series requires at least 2 series"
                        logger.error(tool_result)

                else:
                    tool_result = f"Unknown tool: {function_name}"
                    logger.error(tool_result)

            except Exception as e:
                tool_result = f"Error calling {function_name}: {str(e)}"
                logger.error(tool_result, exc_info=True)

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Get final response from LLM
        final_response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            messages=messages
        )

        return final_response.choices[0].message.content
    else:
        # No tool calls, return direct response
        return response_message.content

def run_example_requests():
    # Run all examples
    for name, request in EXAMPLE_REQUESTS.items():
        print(f"\n{'='*70}")
        print(f"## {name.upper()}")
        print(f"{'='*70}\n")
        try:
            res = asyncio.run(get_llm_response(request))
            print(res)
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Failed to process {name}: {e}", exc_info=True)
        print()

def get_example_requests():
    """Get example request by name"""
    return EXAMPLE_REQUESTS

if __name__ == "__main__":
    run_example_requests()
