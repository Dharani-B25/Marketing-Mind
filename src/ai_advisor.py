"""OpenAI Responses API helpers for dashboard analysis and chat."""

import requests


RESPONSES_URL = "https://api.openai.com/v1/responses"


def ask_openai(api_key, messages, instructions, model="gpt-5.6-luna", max_output_tokens=1400):
    """Send a stateless text request; conversation history is supplied by the app."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured in app secrets.")

    payload = {
        "model": model,
        "instructions": instructions,
        "input": messages,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    response = requests.post(
        RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=(15, 120),
    )
    if response.status_code >= 400:
        try:
            message = response.json().get("error", {}).get("message", "")
        except ValueError:
            message = ""
        raise RuntimeError(message or f"OpenAI API returned HTTP {response.status_code}.")

    result = response.json()
    output_text = []
    for item in result.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                output_text.append(content["text"])
    answer = "\n".join(output_text).strip()
    if not answer:
        raise RuntimeError("OpenAI returned no text. Please try again.")
    return answer
