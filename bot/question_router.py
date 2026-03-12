from dataclasses import dataclass
from datetime import date
import re
from typing import Literal


QuestionType = Literal[
    "current_events",
    "science_tech",
    "numerical",
    "long_horizon_binary",
    "other",
]


@dataclass
class ResearchRoute:
    question_type: QuestionType
    use_asknews: bool
    rationale: str


MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
)


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _contains_near_term_date(text: str) -> bool:
    """
    Detect explicit near-term dates like:
    - before April 3, 2026
    - by March 2026
    - in 2026
    - this year / this month / this quarter
    """
    if any(phrase in text for phrase in [
        "this year",
        "this month",
        "this quarter",
        "next month",
        "next quarter",
        "next year",
    ]):
        return True

    if re.search(r"\b(2026|2027|2028)\b", text):
        return True

    month_pattern = r"\b(" + "|".join(MONTH_NAMES) + r")\b"
    if re.search(month_pattern, text) and re.search(r"\b20\d{2}\b", text):
        return True

    return False


def _contains_event_language(text: str) -> bool:
    event_markers = [
        "meet",
        "meeting",
        "summit",
        "visit",
        "visited",
        "call",
        "talks",
        "negotiation",
        "agreement",
        "announce",
        "announcement",
        "sign",
        "signed",
        "approve",
        "approval",
        "launch",
        "tariff",
        "sanction",
        "ceasefire",
        "election",
        "vote",
        "appoint",
        "resign",
        "merger",
        "acquire",
        "acquisition",
        "lawsuit",
        "court ruling",
    ]
    return any(marker in text for marker in event_markers)


def _contains_public_figure_language(text: str) -> bool:
    public_figure_markers = [
        "president",
        "prime minister",
        "secretary",
        "minister",
        "senator",
        "governor",
        "king",
        "queen",
        "chairman",
        "ceo",
        "xi jinping",
        "donald trump",
        "trump",
        "xi",
        "putin",
        "zelensky",
        "modi",
        "macron",
        "starmer",
        "biden",
    ]
    return any(marker in text for marker in public_figure_markers)


def _looks_like_near_term_binary_event_question(text: str) -> bool:
    patterns = [
        r"\bwill\b.+\bbefore\b",
        r"\bwill\b.+\bby\b",
        r"\bwill\b.+\bin\b\s+20\d{2}\b",
        r"\bwill\b.+\bthis year\b",
        r"\bwill\b.+\bthis month\b",
        r"\bwill\b.+\bnext month\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def route_question(question_text: str) -> ResearchRoute:
    """
    Improved deterministic router.

    Priority order:
    1. Long-horizon structural questions
    2. Numerical questions
    3. Science/tech questions
    4. Near-term current-events / event-resolution questions
    5. Default
    """
    text = _normalize(question_text)

    long_horizon_markers = [
        "by 2050",
        "by 2060",
        "by 2070",
        "by 2080",
        "by 2090",
        "by 2100",
        "extinct",
        "existential risk",
        "civilization collapse",
        "civilisation collapse",
        "superintelligence",
        "human extinction",
        "world government",
        "mars colony",
    ]

    numerical_markers = [
        "how many",
        "what percentage",
        "what proportion",
        "what will be the price",
        "what will the price",
        "what will inflation",
        "gdp",
        "cpi",
        "population",
        "unemployment",
        "stock price",
        "market cap",
        "revenue",
        "sales",
        "temperature",
        "co2",
        "emissions",
        "number of",
        "total of",
        "median",
        "average",
    ]

    science_markers = [
        "artificial intelligence",
        "clinical trial",
        "benchmark",
        "fusion",
        "quantum",
        "battery",
        "drug",
        "vaccine",
        "semiconductor",
        "chip",
        "robot",
        "spacecraft",
        "satellite",
        "gene",
        "protein",
    ]

    # Keep AGI and similar long-run topics out of current-events mode.
    if any(marker in text for marker in long_horizon_markers):
        return ResearchRoute(
            question_type="long_horizon_binary",
            use_asknews=False,
            rationale="Long-horizon structural question: prioritize base rates and outside-view reasoning over news.",
        )

    if any(marker in text for marker in numerical_markers):
        return ResearchRoute(
            question_type="numerical",
            use_asknews=False,
            rationale="Numerical question: prioritize historical patterns, data, and trend framing over news search.",
        )

    if any(marker in text for marker in science_markers):
        return ResearchRoute(
            question_type="science_tech",
            use_asknews=False,
            rationale="Science/technology question: prioritize technical and background research over general news.",
        )

    near_term_signals = [
        _contains_near_term_date(text),
        _contains_event_language(text),
        _contains_public_figure_language(text),
        _looks_like_near_term_binary_event_question(text),
    ]

    if sum(bool(x) for x in near_term_signals) >= 2:
        return ResearchRoute(
            question_type="current_events",
            use_asknews=True,
            rationale="Near-term event-driven question involving current developments and key actors.",
        )

    return ResearchRoute(
        question_type="other",
        use_asknews=False,
        rationale="Default route: prefer non-news research unless the question is clearly driven by current events.",
    )