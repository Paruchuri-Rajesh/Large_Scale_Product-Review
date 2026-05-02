"""Agentic review auditor using Ollama tool-calling directly.

Usage:
    python -m src.agents.review_auditor --product-id P2012
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import requests
from ollama import Client

sys.path.append(str(Path(__file__).resolve().parents[2]))

BASE_URL = "http://127.0.0.1:8000"
ollama = Client()

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_aggregate",
            "description": "Fetch aggregate stats (review_count, avg_rating, fraud_rate, pct_5star) for a product_id.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_reviews",
            "description": "Fetch recent streaming-scored reviews for a product_id.",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_fraud_reviewers",
            "description": "Fetch top suspicious reviewers ranked by fraud_rate.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 10}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_review",
            "description": "Score a single review text for sentiment and fraud probability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "review_body": {"type": "string"},
                    "star_rating": {"type": "integer", "default": 3},
                },
                "required": ["review_body"],
            },
        },
    },
]

# ── Tool implementations ──────────────────────────────────────────────────────

def get_product_aggregate(product_id: str) -> str:
    try:
        resp = requests.get(f"{BASE_URL}/aggregates/products?limit=500", timeout=10)
        match = next((p for p in resp.json() if str(p.get("product_id", "")) == product_id), None)
        return json.dumps(match, indent=2) if match else f"No aggregate found for {product_id}."
    except Exception as e:
        return f"Error: {e}"

def get_product_reviews(product_id: str) -> str:
    try:
        resp = requests.get(f"{BASE_URL}/stream/recent?limit=200", timeout=10)
        rows = [r for r in resp.json() if str(r.get("product_id", "")) == product_id]
        if not rows:
            return f"No streaming reviews found for {product_id}. Run make kafka-produce first."
        return json.dumps(rows[:20], indent=2)
    except Exception as e:
        return f"Error: {e}"

def get_top_fraud_reviewers(limit: int = 10) -> str:
    try:
        resp = requests.get(f"{BASE_URL}/aggregates/fraud-reviewers?limit={limit}", timeout=10)
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Error: {e}"

def score_review(review_body: str, star_rating: int = 3) -> str:
    try:
        resp = requests.post(
            f"{BASE_URL}/predict",
            json={"review_body": review_body, "star_rating": star_rating},
            timeout=15,
        )
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Error: {e}"

TOOL_FN = {
    "get_product_aggregate": get_product_aggregate,
    "get_product_reviews": get_product_reviews,
    "get_top_fraud_reviewers": get_top_fraud_reviewers,
    "score_review": score_review,
}

# ── Agentic loop ──────────────────────────────────────────────────────────────

SYSTEM = """You are a product review fraud auditor. Investigate a product and write a full audit report.

Use the tools in this order:
1. get_product_aggregate — get overall stats including product_category
2. get_product_reviews — get individual reviews
3. get_top_fraud_reviewers — check for suspicious reviewer overlap
4. score_review — score any suspicious review texts

After gathering data, write a report with these exact sections:

PRODUCT SUMMARY
Include the product_id, product_category from the aggregate data, review count, avg rating, and fraud rate.

FRAUD SIGNALS
Describe specific patterns found. List exactly 5 suspicious reviewers maximum — no more.

RISK VERDICT
Start this section with exactly: "RISK VERDICT for [product_id] is HIGH/MEDIUM/LOW because..." all on one line.
Never split across lines.

RECOMMENDATIONS
Give exactly 3 concrete actions.

Important: Always include the product category in the Product Summary. Never list more than 5 reviewers."""

def run_auditor(product_id: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Audit product_id={product_id}. Use tools to gather data, then write the full report."},
    ]

    for iteration in range(8):
        response = ollama.chat(
            model="llama3.2",
            messages=messages,
            tools=TOOLS,
            options={"temperature": 0.2},
        )
        msg = response.message
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            # No more tool calls — final answer
            return msg.content

        # Execute each tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments or {}
            print(f"[agent] calling {fn_name}({fn_args})")
            result = TOOL_FN[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "content": result,
            })

    return "Max iterations reached."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-id", required=True)
    args = parser.parse_args()
    print(f"\n{'='*60}\nAUDIT REPORT — Product: {args.product_id}\n{'='*60}")
    print(run_auditor(args.product_id))
    print('='*60)

if __name__ == "__main__":
    main()
