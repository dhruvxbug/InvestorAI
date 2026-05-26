from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from tools.exa_tools import ExaSearchClient


class ManagementAgent:
    name = "Management & Insider Intelligence Agent"

    def __init__(self) -> None:
        self.exa = ExaSearchClient()

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)

    def analyze(self, ticker: str, company_name: str) -> dict[str, Any]:
        current_year = datetime.utcnow().year
        searches = {
            "leadership": (f"{company_name} CEO MD leadership strategy {current_year}", 8),
            "insider": (f"{company_name} insider buying selling promoter stake {current_year}", 8),
            "earnings_call": (f"{company_name} earnings call transcript management commentary {current_year}", 5),
            "governance": (f"{company_name} board directors corporate governance SEBI compliance", 5),
            "expansion": (f"{company_name} expansion capex investment plans acquisition {current_year}", 8),
        }

        raw_items: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(self.exa.search, query=query, num_results=num_results): category
                for category, (query, num_results) in searches.items()
            }
            for future in as_completed(future_map):
                category = future_map[future]
                for item in future.result():
                    item["category"] = category
                    raw_items.append(item)

        leadership_changes: list[str] = []
        strategic_plans: list[str] = []
        red_flags: list[str] = []
        upcoming_catalysts: list[dict[str, str]] = []

        promoter_increase = 0
        promoter_decrease = 0
        insider_buy = 0
        insider_sell = 0
        guidance_raised = 0
        guidance_lowered = 0

        for item in raw_items:
            text = " ".join([item.get("title") or "", item.get("summary") or "", " ".join(item.get("highlights") or [])]).lower()
            title = item.get("title") or "Untitled"

            if self._contains_any(text, ["ceo", "cfo", "md", "managing director"]) and self._contains_any(
                text, ["resign", "appointed", "replaced", "stepped down", "transition"]
            ):
                leadership_changes.append(title)

            if self._contains_any(text, ["promoter stake increased", "promoter buying", "stake increase"]):
                promoter_increase += 1
            if self._contains_any(text, ["promoter stake reduced", "promoter selling", "stake sale"]):
                promoter_decrease += 1

            if self._contains_any(text, ["insider buying", "insider buy", "director bought"]):
                insider_buy += 1
            if self._contains_any(text, ["insider selling", "insider sell", "director sold"]):
                insider_sell += 1

            if self._contains_any(text, ["guidance raised", "raised guidance", "outlook improved"]):
                guidance_raised += 1
            if self._contains_any(text, ["guidance lowered", "cut guidance", "weak outlook"]):
                guidance_lowered += 1

            if self._contains_any(text, ["pledged shares", "auditor resignation", "regulatory probe", "legal case", "related party"]):
                red_flags.append(title)

            if self._contains_any(text, ["expansion", "capex", "new plant", "acquisition", "new product", "strategic partnership"]):
                strategic_plans.append(title)

            if self._contains_any(text, ["earnings date", "product launch", "approval", "regulatory decision", "board meeting"]):
                probability = "HIGH" if self._contains_any(text, ["confirmed", "scheduled", "announced"]) else "MEDIUM"
                upcoming_catalysts.append({"event": title, "positive_probability": probability})

        if promoter_increase > promoter_decrease:
            promoter_holding_trend = "increasing"
        elif promoter_decrease > promoter_increase:
            promoter_holding_trend = "decreasing"
        else:
            promoter_holding_trend = "stable"

        if insider_buy > insider_sell:
            insider_activity = "bullish"
        elif insider_sell > insider_buy:
            insider_activity = "bearish"
        else:
            insider_activity = "neutral"

        if guidance_raised > guidance_lowered:
            management_guidance = "raised"
        elif guidance_lowered > guidance_raised:
            management_guidance = "lowered"
        else:
            management_guidance = "maintained"

        consistency_score = 8 if management_guidance in {"raised", "maintained"} else 5
        capital_allocation_score = 8 if strategic_plans else 6
        transparency_score = 8 if not red_flags else 4
        governance_score = 8 if len(red_flags) <= 1 else 4
        promoter_conviction_score = 9 if promoter_holding_trend == "increasing" else 6 if promoter_holding_trend == "stable" else 3

        management_score = round(
            (consistency_score + capital_allocation_score + transparency_score + governance_score + promoter_conviction_score)
            / 5,
            1,
        )

        if management_score >= 8.5:
            management_label = "EXCELLENT"
        elif management_score >= 7:
            management_label = "GOOD"
        elif management_score >= 5:
            management_label = "AVERAGE"
        else:
            management_label = "POOR"

        if management_label in {"EXCELLENT", "GOOD"} and insider_activity == "bullish" and promoter_holding_trend != "decreasing":
            management_signal = "BUY"
        elif management_label == "POOR" or red_flags:
            management_signal = "SELL"
        else:
            management_signal = "HOLD"

        unique_catalysts = []
        seen = set()
        for catalyst in upcoming_catalysts:
            event = catalyst["event"]
            if event not in seen:
                seen.add(event)
                unique_catalysts.append(catalyst)

        return {
            "leadership_changes": leadership_changes[:5],
            "promoter_holding_trend": promoter_holding_trend,
            "insider_activity": insider_activity,
            "management_guidance": management_guidance,
            "red_flags": red_flags[:5],
            "strategic_plans": strategic_plans[:5],
            "management_score": management_score,
            "management_label": management_label,
            "upcoming_catalysts": unique_catalysts[:5],
            "management_signal": management_signal,
            "management_reasoning": (
                f"Leadership changes: {len(leadership_changes)}, red flags: {len(red_flags)}, "
                f"promoter trend: {promoter_holding_trend}, insider activity: {insider_activity}."
            ),
        }
