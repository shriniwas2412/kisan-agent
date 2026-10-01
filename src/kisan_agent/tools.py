"""Tools the agent can call. Thin wrappers over mandi-mcp's library functions plus the leaf classifier."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

from mandi_mcp import crops, mandi, weather

from . import vision

TOOLS = [
    {
        "name": "get_mandi_prices",
        "description": "Today's wholesale (mandi) prices for a commodity, in Rs/quintal. Use when the farmer asks what a crop is selling for.",
        "input_schema": {
            "type": "object",
            "properties": {
                "commodity": {"type": "string", "description": 'English Agmarknet name, e.g. "Wheat", "Soyabean", "Onion"'},
                "state": {"type": "string", "description": 'e.g. "Madhya Pradesh"'},
                "district": {"type": "string", "description": 'e.g. "Sehore"'},
            },
            "required": ["commodity"],
        },
    },
    {
        "name": "best_mandi",
        "description": "Which mandi in a state pays the most for a commodity today, plus the median price. Use for 'where should I sell?'.",
        "input_schema": {
            "type": "object",
            "properties": {"commodity": {"type": "string"}, "state": {"type": "string"}},
            "required": ["commodity", "state"],
        },
    },
    {
        "name": "farm_weather",
        "description": "Daily forecast (temperature, rain, wind) for an Indian town/village with a spray/no-spray advisory per day.",
        "input_schema": {
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "Town or village name in English"},
                "days": {"type": "integer", "minimum": 1, "maximum": 16},
            },
            "required": ["place"],
        },
    },
    {
        "name": "crop_calendar",
        "description": "Season and sowing/harvest months for a crop (English or Hindi name).",
        "input_schema": {"type": "object", "properties": {"crop": {"type": "string"}}, "required": ["crop"]},
    },
    {
        "name": "what_to_sow",
        "description": "Crops typically sown and harvested in a month (1-12). Omit month for the current month.",
        "input_schema": {"type": "object", "properties": {"month": {"type": "integer", "minimum": 1, "maximum": 12}}},
    },
    {
        "name": "diagnose_leaf",
        "description": "Classify a crop-leaf photo the farmer sent. Returns the top disease guesses with confidence. Only call when an image is attached.",
        "input_schema": {
            "type": "object",
            "properties": {"image_id": {"type": "string", "description": "id of the uploaded image"}},
            "required": ["image_id"],
        },
    },
]


async def run_tool(name: str, args: dict, images: dict[str, bytes] | None = None) -> str:
    """Execute a tool and return a JSON string for the tool_result block."""
    try:
        if name == "get_mandi_prices":
            prices = await mandi.fetch_prices(args["commodity"], args.get("state"), args.get("district"), limit=20)
            result = {"unit": "Rs/quintal", "prices": [asdict(p) for p in prices]}
        elif name == "best_mandi":
            result = mandi.summarize(await mandi.fetch_prices(args["commodity"], args["state"], limit=200))
        elif name == "farm_weather":
            result = await weather.forecast(args["place"], args.get("days", 5))
        elif name == "crop_calendar":
            result = crops.crop_info(args["crop"])
        elif name == "what_to_sow":
            result = crops.crops_for_month(args.get("month") or date.today().month)
        elif name == "diagnose_leaf":
            image = (images or {}).get(args["image_id"])
            if image is None:
                result = {"error": f"No image with id {args['image_id']!r}"}
            else:
                result = vision.classify(image)
        else:
            result = {"error": f"Unknown tool {name!r}"}
    except (mandi.MandiError, weather.WeatherError, KeyError) as e:
        result = {"error": str(e)}
    return json.dumps(result, ensure_ascii=False)
