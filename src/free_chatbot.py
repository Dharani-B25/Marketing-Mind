"""Rule-based, no-API answers for common MarketMind dashboard questions."""


def _number(value, digits=2):
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _currency(value):
    try:
        return f"₹{float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def answer_dashboard_question(question, context):
    """Answer supported KPI, campaign, channel, forecast, and SEO questions."""
    q = (question or "").casefold().strip()
    kpis = context.get("overall_kpis") or {}
    campaigns = context.get("campaigns") or []
    channels = context.get("channels") or []
    selected = context.get("selected_campaign") or {}
    seo = context.get("latest_on_page_technical_seo_audit")
    speed_results = context.get("latest_pagespeed_mobile_desktop_results") or {}

    def answer(direct, evidence, next_step):
        return (
            f"**Direct answer**\n{direct}\n\n"
            f"**Dashboard evidence**\n{evidence}\n\n"
            f"**Next step**\n{next_step}"
        )

    asks_seo = any(word in q for word in ("seo", "website", "technical", "mobile", "desktop", "speed", "page speed", "pagespeed", "lcp", "cls", "inp"))
    asks_forecast = any(word in q for word in ("forecast", "future", "predict", "next month", "scenario", "budget change"))
    asks_channel = any(word in q for word in ("channel", "facebook", "instagram", "google ads", "email", "social"))
    asks_campaign = any(word in q for word in ("campaign", "campaigns", "ad", "ads"))
    asks_best = any(word in q for word in ("best", "top", "highest", "strongest", "better", "nalla", "sirandha"))
    asks_worst = any(word in q for word in ("worst", "lowest", "weakest", "poor", "low", "weak", "mosam"))
    asks_action = any(word in q for word in ("recommend", "suggest", "action", "improve", "optimi", "what should", "next step", "enna seiya", "epdi improve"))

    if asks_seo:
        if not seo and not speed_results:
            return answer(
                "There are no SEO or PageSpeed results in the current dashboard yet.",
                "No completed website audit or mobile/desktop speed report is attached to this dashboard context.",
                "Run the On-page/Technical SEO audit and the mobile or desktop PageSpeed check, then ask again.",
            )
        facts = []
        if seo:
            facts.append(f"Audited URL: {seo.get('url', 'N/A')}")
            facts.append(f"Page response time: {_number(seo.get('response_seconds'))} seconds")
            facts.append(f"Page size: {_number(seo.get('page_kb'))} KB")
            facts.append(f"Internal links: {seo.get('internal_links', 'N/A')}")
            facts.append(f"Images missing alt text: {seo.get('images_missing_alt', 'N/A')}")
        for device, result in speed_results.items():
            scores = result.get("scores") or {}
            facts.append(f"{device.title()} PageSpeed scores: " + ", ".join(f"{name} {_number(score, 0)}/100" for name, score in scores.items()))
            metrics = result.get("metrics") or {}
            if metrics:
                facts.append(f"{device.title()} loading metrics: " + ", ".join(f"{name} {value}" for name, value in metrics.items()))
            opportunities = result.get("opportunities") or []
            if opportunities:
                facts.append(f"{device.title()} improvement: {opportunities[0].get('title', 'Review PageSpeed opportunities')}")
        return answer(
            "Review the audit and device-specific speed results below; address the largest reported issue first.",
            "\n".join(f"- {fact}" for fact in facts),
            "Fix the highest-impact reported issue, rerun the same check, and compare the new result.",
        )

    if asks_forecast:
        forecast = context.get("ml_revenue_forecast")
        scenario = context.get("budget_scenario") or {}
        evidence = [f"Forecast revenue: {_currency(forecast)}" if forecast is not None else "A revenue forecast is not available for this dataset."]
        if scenario:
            evidence.append(
                f"Budget scenario ({scenario.get('budget_change_percent', 'N/A')}% change): "
                f"estimated revenue {_currency(scenario.get('estimated_revenue'))} "
                f"({ _number(scenario.get('estimated_revenue_change_percent')) }% change)."
            )
            evidence.append("Scenario assumes current ROAS stays stable; it is not a guarantee.")
        return answer(
            "Use the forecast as an estimate and the budget scenario as a what-if comparison, not a promise.",
            "\n".join(f"- {item}" for item in evidence),
            "Compare the estimate with actual results as new data arrives; avoid changing the full budget based on one forecast alone.",
        )

    if asks_best or asks_worst:
        rows = channels if asks_channel else campaigns
        label = "Channel" if asks_channel else "Campaign"
        if rows:
            metric = "ROAS"
            candidates = [row for row in rows if row.get(metric) is not None]
            if candidates:
                chosen = sorted(candidates, key=lambda row: float(row.get(metric) or 0), reverse=not asks_worst)[0]
                name = chosen.get(label, "Unknown")
                evidence = (
                    f"{label}: {name}\n"
                    f"ROAS: {_number(chosen.get('ROAS'))}\n"
                    f"Revenue: {_currency(chosen.get('Revenue'))}\n"
                    f"Spend: {_currency(chosen.get('Spend'))}"
                )
                return answer(
                    f"{name} has the {'lowest' if asks_worst else 'highest'} ROAS among the available {label.lower()}s.",
                    evidence,
                    "Check conversions and CPA alongside ROAS before reducing or increasing its budget.",
                )

    if asks_action:
        selected_name = selected.get("name", "the selected campaign")
        metrics = selected.get("metrics") or {}
        ctr = metrics.get("CTR")
        cvr = metrics.get("Conversion_Rate")
        roas = metrics.get("ROAS")
        actions = []
        if ctr is not None and float(ctr) < float(kpis.get("ctr_percent") or 0):
            actions.append("Review the ad creative and audience because this campaign's CTR is below the overall CTR.")
        if cvr is not None and float(cvr) < 1:
            actions.append("Check the landing page and conversion path; the campaign conversion rate is below 1%.")
        if roas is not None and float(roas) < 1:
            actions.append("Review spend and targeting before scaling; this campaign's ROAS is below 1.")
        if not actions:
            actions.append("Compare this campaign's ROAS, CTR, conversion rate, and CPA with the channel and overall averages before changing budget.")
        evidence = (
            f"Selected campaign: {selected_name}\n"
            f"ROAS: {_number(roas)} | CTR: {_number(ctr)}% | Conversion rate: {_number(cvr)}%"
        )
        return answer("Start with the selected campaign's weakest available KPI.", evidence, "\n".join(f"{i}. {item}" for i, item in enumerate(actions, 1)))

    if asks_campaign and campaigns:
        row = next((item for item in campaigns if str(item.get("Campaign", "")).casefold() in q), None)
        row = row or next((item for item in campaigns if item.get("Campaign") == selected.get("name")), campaigns[0])
        name = row.get("Campaign", selected.get("name", "Selected campaign"))
        return answer(
            f"Here is the dashboard performance for {name}.",
            f"ROAS: {_number(row.get('ROAS'))}\nRevenue: {_currency(row.get('Revenue'))}\nSpend: {_currency(row.get('Spend'))}\nCTR: {_number(row.get('CTR'))}%\nConversion rate: {_number(row.get('Conversion_Rate'))}%\nCPA: {_currency(row.get('CPA'))}\nConversions: {_number(row.get('Conversions'), 0)}",
            "Compare ROAS, CTR, conversion rate, and CPA with other campaigns before changing its budget.",
        )

    if asks_channel and channels:
        row = next((item for item in channels if str(item.get("Channel", "")).casefold() in q), None)
        row = row or max(channels, key=lambda item: float(item.get("ROAS") or 0))
        name = row.get("Channel", "Selected channel")
        return answer(
            f"Here is the dashboard performance for {name}.",
            f"ROAS: {_number(row.get('ROAS'))}\nRevenue: {_currency(row.get('Revenue'))}\nSpend: {_currency(row.get('Spend'))}\nCTR: {_number(row.get('CTR'))}%\nConversion rate: {_number(row.get('Conversion_Rate'))}%\nConversions: {_number(row.get('Conversions'), 0)}",
            "Compare this channel with the others using the same date range and KPI definitions.",
        )

    metric_map = (
        (("roas", "return on ad spend", "return"), "roas", "ROAS is the revenue earned per unit of ad spend."),
        (("revenue", "sales", "varumanam"), "revenue", "Revenue is the total recorded revenue in the active date range."),
        (("spend", "cost", "budget", "selavu"), "spend", "Spend is the total recorded marketing cost in the active date range."),
        (("conversion rate", "conversion_rate", "cvr"), "conversion_rate", "Conversion rate is conversions divided by clicks."),
        (("conversion", "conversions", "orders"), "conversions", "Conversions are the total recorded conversions in the active date range."),
        (("ctr", "click through"), "ctr_percent", "CTR is clicks divided by impressions."),
        (("impression", "reach"), "impressions", "Impressions are the total recorded ad views in the active date range."),
        (("click", "traffic"), "clicks", "Clicks are the total recorded ad clicks in the active date range."),
        (("cpa", "cost per acquisition"), "cpa", "CPA is spend divided by conversions."),
    )
    for terms, metric, definition in metric_map:
        if any(term in q for term in terms):
            value = kpis.get(metric)
            if metric == "conversion_rate":
                clicks = float(kpis.get("clicks") or 0)
                value = (float(kpis.get("conversions") or 0) / clicks * 100) if clicks else None
            elif metric == "cpa":
                conversions = float(kpis.get("conversions") or 0)
                value = (float(kpis.get("spend") or 0) / conversions) if conversions else None
            formatted = _currency(value) if metric in ("revenue", "spend", "cpa") else _number(value)
            label = "CTR" if metric == "ctr_percent" else metric.upper().replace("_", " ")
            return answer(
                f"The current {label} is {formatted}{'%' if metric in ('ctr_percent', 'conversion_rate') and value is not None else ''}.",
                f"{definition}\nActive date range: {context.get('active_date_range', {}).get('from', 'N/A')} to {context.get('active_date_range', {}).get('to', 'N/A')}.",
                "Compare this value across campaigns or channels, then focus on the segment with the weakest result.",
            )

    return answer(
        "I can answer common questions about your dashboard, but I do not understand every open-ended question like a generative AI model.",
        "Supported topics: campaign/channel comparisons, ROAS, revenue, spend, conversions, CTR, CPA, forecasts, SEO, and PageSpeed.",
        "Try a specific question such as 'Which campaign has the highest ROAS?' or 'What is my total revenue?'.",
    )
