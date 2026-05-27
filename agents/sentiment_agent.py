from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from tools.exa_tools import ExaSearchClient


class SentimentAgent:
    name = "News & Sentiment Agent"

    POSITIVE_WORDS = {
        "upgrade",
        "beat",
        "growth",
        "strong",
        "outperform",
        "bullish",
        "profit rise",
        "record",
    }
    NEGATIVE_WORDS = {
        "downgrade",
        "miss",
        "decline",
        "weak",
        "probe",
        "fraud",
        "loss",
        "lawsuit",
        "warning",
    }

    HIGH_IMPACT_WORDS = {
        "earnings miss",
        "ceo change",
        "cfo change",
        "regulatory action",
        "fraud",
        "ban",
        "default",
    }
    MEDIUM_IMPACT_WORDS = {
        "guidance",
        "target price",
        "downgrade",
        "upgrade",
        "margin",
        "acquisition",
        "results",
    }

    SOURCE_CREDIBILITY = {
        "economictimes.com": 1.0,
        "moneycontrol.com": 0.9,
        "livemint.com": 1.0,
        "reuters.com": 1.0,
        "business-standard.com": 0.9,
        "financialexpress.com": 0.9,
        "cnbctv18.com": 0.9,
        "bloomberg.com": 1.0,
        "default": 0.4,
    }
    MAX_FACT_LENGTH = 180
    FACT_TRUNCATE_LENGTH = MAX_FACT_LENGTH - 3

    def __init__(self) -> None:
        self.exa = ExaSearchClient()

    @staticmethod
    def _source_from_url(url: str | None) -> str:
        if not url:
            return "Unknown"
        domain = urlparse(url).netloc.replace("www.", "")
        return domain or "Unknown"

    def _source_weight(self, source: str) -> float:
        source_lower = source.lower()
        return self.SOURCE_CREDIBILITY.get(source_lower, self.SOURCE_CREDIBILITY["default"])

    @staticmethod
    def _parse_date(raw_date: str | None) -> datetime | None:
        if not raw_date:
            return None
        candidate = raw_date.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    def _recency_weight(self, article_date: datetime | None) -> float:
        if article_date is None:
            return 0.5
        now = datetime.now(timezone.utc)
        days_old = max((now - article_date).days, 0)
        if days_old <= 3:
            return 1.0
        if days_old <= 7:
            return 0.85
        if days_old <= 14:
            return 0.65
        if days_old <= 30:
            return 0.4
        return 0.2

    def _sentiment(self, text: str) -> str:
        lowered = text.lower()
        positive_hits = sum(1 for word in self.POSITIVE_WORDS if word in lowered)
        negative_hits = sum(1 for word in self.NEGATIVE_WORDS if word in lowered)
        if positive_hits > negative_hits:
            return "POSITIVE"
        if negative_hits > positive_hits:
            return "NEGATIVE"
        return "NEUTRAL"

    def _impact(self, text: str) -> str:
        lowered = text.lower()
        if any(word in lowered for word in self.HIGH_IMPACT_WORDS):
            return "HIGH"
        if any(word in lowered for word in self.MEDIUM_IMPACT_WORDS):
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _impact_weight(level: str) -> int:
        return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(level, 1)

    @staticmethod
    def _extract_target_prices(text: str) -> list[float]:
        matches = re.findall(r"(?:₹|rs\.?\s*)(\d+(?:\.\d+)?)", text.lower())
        return [float(match) for match in matches]

    @staticmethod
    def _extract_rating_counts(texts: list[str]) -> dict[str, int]:
        joined = " ".join(texts).lower()
        return {
            "strong_buy": joined.count("strong buy"),
            "buy": joined.count(" buy ") + joined.count("buy,"),
            "hold": joined.count(" hold "),
            "sell": joined.count(" sell "),
            "strong_sell": joined.count("strong sell"),
        }

    @staticmethod
    def _one_line_fact(title: str, summary: str) -> str:
        sentence = (title or summary).strip()
        if not sentence:
            return "No material fact extracted."
        sentence = sentence.replace("\n", " ")
        if len(sentence) > SentimentAgent.MAX_FACT_LENGTH:
            sentence = sentence[: SentimentAgent.FACT_TRUNCATE_LENGTH].rstrip() + "..."
        return sentence

    def analyze(
        self, ticker: str, company_name: str, sector_name: str = "Indian equity"
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        current_month = now.strftime("%B")
        current_year = now.year

        searches = {
            "recent_news": (f"{company_name} stock news {current_month} {current_year}", 15),
            "earnings": (f"{company_name} quarterly results earnings {current_year}", 8),
            "analyst_ratings": (
                f"{company_name} analyst rating target price buy sell {current_year}",
                10,
            ),
            "regulatory": (f"{company_name} SEBI RBI regulatory government policy", 5),
            "competitors": (f"{sector_name} India sector outlook {current_year}", 8),
        }

        raw_results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_map = {
                executor.submit(
                    self.exa.search, query=query, num_results=num_results
                ): key
                for key, (query, num_results) in searches.items()
            }
            for future in as_completed(future_map):
                category = future_map[future]
                for item in future.result():
                    item["category"] = category
                    raw_results.append(item)

        articles = []
        sentiment_score_raw = 0.0
        total_weight = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        all_texts = []
        all_target_prices: list[float] = []
        key_facts: list[str] = []

        for item in raw_results:
            snippet = " ".join(
                [
                    item.get("title") or "",
                    item.get("summary") or "",
                    " ".join(item.get("highlights") or []),
                ]
            ).strip()
            sentiment = self._sentiment(snippet)
            impact = self._impact(snippet)
            source = self._source_from_url(item.get("url"))
            article_date = self._parse_date(item.get("published_date"))
            weight = (
                self._impact_weight(impact)
                * self._source_weight(source)
                * self._recency_weight(article_date)
            )
            all_texts.append(snippet)
            all_target_prices.extend(self._extract_target_prices(snippet))
            key_facts.append(
                self._one_line_fact(item.get("title") or "", item.get("summary") or "")
            )

            if sentiment == "POSITIVE":
                positive_count += 1
                sentiment_score_raw += weight
            elif sentiment == "NEGATIVE":
                negative_count += 1
                sentiment_score_raw -= weight
            else:
                neutral_count += 1
            total_weight += weight

            articles.append(
                {
                    "headline": item.get("title"),
                    "source": source,
                    "date": item.get("published_date"),
                    "snippet": (item.get("summary") or "")[:350],
                    "sentiment": sentiment,
                    "impact_level": impact,
                    "category": item.get("category"),
                    "url": item.get("url"),
                    "credibility_weight": round(self._source_weight(source), 2),
                    "recency_weight": round(self._recency_weight(article_date), 2),
                    "one_line_fact": key_facts[-1],
                }
            )

        normalized_score = (
            round((sentiment_score_raw / total_weight), 2) if total_weight else 0.0
        )
        if normalized_score >= 0.5:
            overall_sentiment = "VERY POSITIVE"
        elif normalized_score >= 0.2:
            overall_sentiment = "POSITIVE"
        elif normalized_score <= -0.5:
            overall_sentiment = "VERY NEGATIVE"
        elif normalized_score <= -0.2:
            overall_sentiment = "NEGATIVE"
        else:
            overall_sentiment = "NEUTRAL"

        rating_counts = self._extract_rating_counts(all_texts)
        avg_target_price = (
            round(sum(all_target_prices) / len(all_target_prices), 2)
            if all_target_prices
            else None
        )

        theme_buckets = {
            "Earnings momentum": ["earnings", "results", "profit", "margin", "beat"],
            "Analyst action": [
                "rating",
                "target",
                "upgrade",
                "downgrade",
                "brokerage",
            ],
            "Regulatory developments": [
                "sebi",
                "rbi",
                "policy",
                "regulatory",
                "compliance",
            ],
            "Growth and expansion": [
                "growth",
                "expansion",
                "capex",
                "demand",
                "outlook",
            ],
            "Risk events": ["fraud", "probe", "weak", "decline", "lawsuit", "default"],
        }
        theme_counter: Counter[str] = Counter()
        merged_text = " ".join(all_texts).lower()
        for theme, keywords in theme_buckets.items():
            theme_counter[theme] = sum(merged_text.count(keyword) for keyword in keywords)
        sentiment_themes = [
            theme for theme, count in theme_counter.most_common(5) if count > 0
        ]

        key_events = [
            article["headline"]
            for article in sorted(
                articles,
                key=lambda a: {
                    "HIGH": 3,
                    "MEDIUM": 2,
                    "LOW": 1,
                }[a["impact_level"]],
                reverse=True,
            )[:5]
            if article.get("headline")
        ]

        if overall_sentiment in {"VERY POSITIVE", "POSITIVE"}:
            sentiment_signal = "BUY"
        elif overall_sentiment in {"VERY NEGATIVE", "NEGATIVE"}:
            sentiment_signal = "SELL"
        else:
            sentiment_signal = "HOLD"

        return {
            "articles": articles,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": normalized_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "analyst_consensus": rating_counts,
            "avg_target_price": avg_target_price,
            "highest_target_price": max(all_target_prices) if all_target_prices else None,
            "lowest_target_price": min(all_target_prices) if all_target_prices else None,
            "sentiment_themes": sentiment_themes,
            "key_events": key_events,
            "key_facts": list(dict.fromkeys(key_facts))[:7],
            "sentiment_signal": sentiment_signal,
            "sentiment_reasoning": (
                f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral articles out of {len(articles)}. "
                f"Credibility+recency weighted sentiment score={normalized_score}."
            ),
        }
