"""Google PageSpeed Insights API helpers for mobile and desktop audits."""

import requests


API_URL = "https://pagespeedonline.googleapis.com/pagespeedonline/v5/runPagespeed"


def run_pagespeed(url, strategy, api_key=""):
    """Run Lighthouse categories for one URL and device strategy."""
    params = {
        "url": url,
        "strategy": strategy.lower(),
        "category": ["performance", "accessibility", "best-practices", "seo"],
    }
    headers = {}
    if api_key:
        # Send credentials in a header rather than the query string so the key
        # is less likely to appear in request URLs or URL-based logs.
        headers["x-goog-api-key"] = api_key

    response = requests.get(API_URL, params=params, headers=headers, timeout=120)
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except ValueError:
            detail = ""
        raise RuntimeError(detail or f"PageSpeed API returned HTTP {response.status_code}.")

    payload = response.json()
    lighthouse = payload.get("lighthouseResult", {})
    categories = lighthouse.get("categories", {})
    audits = lighthouse.get("audits", {})

    scores = {
        key: round(category["score"] * 100)
        for key, category in categories.items()
        if category.get("score") is not None
    }

    metric_ids = ["largest-contentful-paint", "cumulative-layout-shift", "interaction-to-next-paint"]
    metrics = {}
    for audit_id in metric_ids:
        item = audits.get(audit_id, {})
        if item.get("displayValue"):
            metrics[audit_id] = item["displayValue"]
    if "interaction-to-next-paint" not in metrics:
        inp = audits.get("experimental-interaction-to-next-paint", {})
        if inp.get("displayValue"):
            metrics["interaction-to-next-paint"] = inp["displayValue"]

    opportunities = []
    for audit_id, item in audits.items():
        if item.get("scoreDisplayMode") in {"manual", "notApplicable", "informative"}:
            continue
        if item.get("score") is not None and item["score"] < 0.9 and item.get("title"):
            opportunities.append({
                "title": item["title"],
                "description": item.get("description", "").split("[Learn more]")[0].strip(),
                "display_value": item.get("displayValue", ""),
                "score": item["score"],
            })
    opportunities.sort(key=lambda item: (item["score"], item["title"]))

    mobile_checks = []
    if strategy.lower() == "mobile":
        for audit_id in ("viewport", "font-size", "tap-targets", "content-width"):
            item = audits.get(audit_id, {})
            if item.get("score") is not None:
                mobile_checks.append({
                    "Check": item.get("title", audit_id),
                    "Result": "Pass" if item["score"] >= 0.9 else "Needs attention",
                    "Details": item.get("displayValue", ""),
                })

    return {
        "strategy": strategy.lower(),
        "url": lighthouse.get("finalUrl", url),
        "scores": scores,
        "metrics": metrics,
        "mobile_checks": mobile_checks,
        "opportunities": opportunities[:8],
        "fetch_time": lighthouse.get("fetchTime", ""),
        "field_data": payload.get("loadingExperience", {}).get("metrics", {}),
    }
