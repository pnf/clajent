"""
Basic usage examples for TimeSense MCP server
"""
from readline import clear_history

import numpy as np
from datetime import datetime, timedelta
import ts_server as tss
import asyncio
from anthropic import Anthropic
from ignore_me_secrets import ANTHROPIC_KEY
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

server = tss.TimeSeriesMCPServer()

client = Anthropic(api_key=ANTHROPIC_KEY)
async def get_llm_response(input):
    # check if input dict contains "question"
    if "question" in input:
        prompt = input["question"]
        del input["question"]
    else:
        prompt = "Analyze the following time series data:"

    prompt = f"{prompt}:\n\n{json.dumps(input, indent=2)}"

    message = await client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        mcp_servers = [server],
        messages=[{"role": "user", "content": prompt}]
    )
    logger.info(f"LLM Prompt (first 200 chars): {prompt[:200]}...")
    return message.choices[0].message.content

def run_example_requests():
    for name, request in EXAMPLE_REQUESTS.items():
        print(f"## {name}\n")
        res = asyncio.run(get_llm_response(request))
        print(res)

def get_example_requests():
    """Get example request by name"""
    return EXAMPLE_REQUESTS

if __name__ == "__main__":
    run_example_requests()
