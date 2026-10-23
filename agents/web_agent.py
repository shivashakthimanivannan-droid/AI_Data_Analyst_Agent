
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ddgs import DDGS


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_MAX_RESULTS = 6
MAX_RESULTS = 10
MAX_SNIPPET_LENGTH = 1200

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "the", "to", "what", "which", "with",
    "about", "show", "me", "find", "search", "web",
    "latest", "recent", "today", "week", "month",
    "this", "last", "news", "updates", "trends",
    "artificial", "intelligence",
}

TERM_GROUPS = {
    "ai": {
        "ai", "artificial", "intelligence",
        "generative", "genai", "llm", "agentic",
        "agents", "machine", "learning",
    },
    "analytics": {
        "analytics", "analysis", "analytical",
        "analyst", "insights",
    },
    "data": {
        "data", "dataset", "datasets",
        "database", "databases", "pipeline",
        "pipelines",
    },
    "business": {
        "business", "enterprise", "enterprises",
        "organization", "organizations",
        "commercial",
    },
}


# =========================================================
# INPUT AND URL HELPERS
# =========================================================

def _clean_query(question):
    if not isinstance(question, str):
        raise ValueError(
            "The research question must be text."
        )

    question = " ".join(question.split())

    if not question:
        raise ValueError(
            "Please enter a research question."
        )

    if len(question) > 1000:
        raise ValueError(
            "Please keep the research question "
            "under 1000 characters."
        )

    return question


def _clean_text(value):
    return re.sub(
        r"\s+", " ", str(value or "")
    ).strip()


def _is_valid_http_url(url):
    if not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())

        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )

    except (ValueError, TypeError):
        return False


def _normalize_url(url):
    parsed = urlparse(url.strip())

    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/",
        "",
        parsed.query,
        "",
    ))


def _safe_markdown_label(value):
    text = _clean_text(value)

    return (
        text.replace("[", "")
        .replace("]", "")
        .replace("|", "-")
    )


# =========================================================
# SEARCH INTENT
# =========================================================

def _is_news_question(question):
    q = question.lower()

    keywords = (
        "news",
        "latest",
        "recent",
        "today",
        "this week",
        "this month",
        "breaking",
        "announcement",
        "updates",
        "current developments",
    )

    return any(keyword in q for keyword in keywords)


def _time_window(question):
    q = question.lower()

    if "today" in q or "last 24 hours" in q:
        return "d"

    if any(
        phrase in q
        for phrase in (
            "this week",
            "past week",
            "last 7 days",
        )
    ):
        return "w"

    if any(
        phrase in q
        for phrase in (
            "this month",
            "past month",
            "last 30 days",
        )
    ):
        return "m"

    if any(
        phrase in q
        for phrase in (
            "this year",
            "past year",
            "last 12 months",
        )
    ):
        return "y"

    if _is_news_question(question):
        return "m"

    return None


# =========================================================
# RELEVANCE SCORING
# =========================================================

def _tokenize(text):
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower(),
        )
    )


def _expand_terms(tokens):
    expanded = set(tokens)

    for group in TERM_GROUPS.values():
        if tokens.intersection(group):
            expanded.update(group)

    return expanded


def _relevance_score(question, result):
    """
    Lightweight lexical relevance scoring.

    This ranks search results; it does not verify
    the factual accuracy of an article.
    """
    query_tokens = _tokenize(question)

    meaningful = {
        token
        for token in query_tokens
        if token not in STOPWORDS
        and len(token) > 2
    }

    title_tokens = _tokenize(result["title"])
    snippet_tokens = _tokenize(result["snippet"])

    expanded_title = _expand_terms(title_tokens)
    expanded_snippet = _expand_terms(snippet_tokens)

    title_matches = meaningful.intersection(
        expanded_title
    )

    snippet_matches = meaningful.intersection(
        expanded_snippet
    )

    score = (
        len(title_matches) * 4
        + len(snippet_matches) * 2
    )

    # Reward coverage of distinct question concepts.
    if meaningful:
        coverage = len(
            meaningful.intersection(
                expanded_title | expanded_snippet
            )
        ) / len(meaningful)

        score += round(coverage * 5, 2)

    # Article-level news is preferred to topic hubs.
    if result["source_type"] == "news":
        score += 1

    if result["publication_date"]:
        score += 0.5

    if result["likely_landing_page"]:
        score -= 4

    return round(score, 2)


def _is_relevant(question, result):
    query_tokens = _tokenize(question)

    meaningful = {
        token
        for token in query_tokens
        if token not in STOPWORDS
        and len(token) > 2
    }

    # For vague questions, do not reject all results.
    if not meaningful:
        return True

    document_tokens = _expand_terms(
        _tokenize(
            result["title"] + " " + result["snippet"]
        )
    )

    matches = meaningful.intersection(
        document_tokens
    )

    required_matches = (
        2 if len(meaningful) >= 3 else 1
    )

    return len(matches) >= required_matches


# =========================================================
# RESULT NORMALIZATION
# =========================================================

def _parse_date(value):
    if not value:
        return None

    if isinstance(value, datetime):
        dt = value

    else:
        text = str(value).strip()

        try:
            dt = datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )
        except ValueError:
            return text or None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.date().isoformat()


def _is_landing_page(url, title):
    path = urlparse(url).path.lower().rstrip("/")
    title_lower = title.lower()

    generic_titles = (
        "latest news",
        "recent news",
        "news and insights",
        "news & videos",
    )

    hub_prefixes = (
        "/topic/",
        "/topics/",
        "/category/",
        "/categories/",
        "/tag/",
        "/tags/",
    )

    if any(
        phrase in title_lower
        for phrase in generic_titles
    ):
        return True

    if any(
        path.startswith(prefix)
        for prefix in hub_prefixes
    ):
        return True

    if path in ("/latest", "/news"):
        return True

    return False


def _format_results(raw_results, source_type):
    formatted = []
    seen_urls = set()

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        title = _clean_text(
            item.get("title") or "Untitled result"
        )

        url = _clean_text(
            item.get("href")
            or item.get("url")
            or ""
        )

        if not _is_valid_http_url(url):
            continue

        normalized = _normalize_url(url)

        if normalized in seen_urls:
            continue

        seen_urls.add(normalized)

        snippet = _clean_text(
            item.get("body")
            or item.get("snippet")
            or ""
        )

        publication_date = _parse_date(
            item.get("date")
            or item.get("published")
            or item.get("published_at")
        )

        publisher = _clean_text(
            item.get("source")
            or urlparse(url).hostname
            or "Unknown source"
        )

        formatted.append({
            "title": title,
            "url": url,
            "snippet": snippet[:MAX_SNIPPET_LENGTH],
            "publication_date": publication_date,
            "publisher": publisher,
            "source_type": source_type,
            "likely_landing_page": _is_landing_page(
                url, title
            ),
        })

    return formatted


def _rank_and_filter(question, results, max_results):
    unique = []
    seen = set()

    for result in results:
        normalized = _normalize_url(result["url"])

        if normalized in seen:
            continue

        seen.add(normalized)

        result["relevance_score"] = (
            _relevance_score(question, result)
        )

        if _is_relevant(question, result):
            unique.append(result)

    unique.sort(
        key=lambda item: (
            -item["relevance_score"],
            item["likely_landing_page"],
        )
    )

    return unique[:max_results]


# =========================================================
# SEARCH FUNCTIONS
# =========================================================

def _search_news(client, question, max_results, timelimit):
    kwargs = {
        "max_results": max_results,
    }

    if timelimit:
        kwargs["timelimit"] = timelimit

    return list(
        client.news(question, **kwargs) or []
    )


def _search_web(client, question, max_results, timelimit):
    kwargs = {
        "max_results": max_results,
    }

    if timelimit:
        kwargs["timelimit"] = timelimit

    return list(
        client.text(question, **kwargs) or []
    )


# =========================================================
# ANSWER GENERATION
# =========================================================

def _build_answer(question, results, search_mode):
    if not results:
        return (
            "No sufficiently relevant results were found. "
            "Try a more specific research question."
        )

    lines = [
        "## Web Research Findings",
        "",
        f"**Research question:** {_clean_text(question)}",
        "",
        f"**Search mode:** {search_mode}",
        "",
        (
            "Results are ranked using keyword relevance. "
            "Publication dates are provider-reported, "
            "and article contents have not been "
            "independently verified."
        ),
        "",
    ]

    for index, result in enumerate(results, start=1):
        title = _safe_markdown_label(
            result["title"]
        )

        url = result["url"]

        lines.extend([
            f"### {index}. [{title}](<{url}>)",
            "",
            (
                "**Publication date:** "
                + (
                    result["publication_date"]
                    or "Not provided"
                )
            ),
            f"**Publisher:** {result['publisher']}",
            "",
            (
                result["snippet"]
                or "No description available."
            ),
            "",
            f"**Source:** [{url}](<{url}>)",
            "",
        ])

    lines.append(
        "**Research note:** Relevance scoring is "
        "heuristic. Verify the source, date, and "
        "original context before relying on findings."
    )

    return "\n".join(lines)


# =========================================================
# MAIN WEB AGENT
# =========================================================

def run_web_agent(
    question,
    max_results=DEFAULT_MAX_RESULTS,
):
    searched_at = datetime.now(
        timezone.utc
    ).isoformat()

    try:
        question = _clean_query(question)

        max_results = max(
            1,
            min(int(max_results), MAX_RESULTS),
        )

        news_mode = _is_news_question(question)
        timelimit = _time_window(question)

        # Retrieve more candidates than we display,
        # allowing the relevance filter to reject noise.
        candidate_count = min(
            max_results * 3,
            30,
        )

        client = DDGS()

        raw_results = []
        warnings = []
        used_modes = []

        if news_mode:
            try:
                news = _search_news(
                    client,
                    question,
                    candidate_count,
                    timelimit,
                )

                raw_results.extend(
                    _format_results(news, "news")
                )

                used_modes.append("News")

            except Exception as exc:
                warnings.append(
                    f"News search unavailable: {exc}"
                )

        try:
            web = _search_web(
                client,
                question,
                candidate_count,
                timelimit if news_mode else None,
            )

            raw_results.extend(
                _format_results(web, "web")
            )

            used_modes.append("Web")

        except Exception as exc:
            warnings.append(
                f"Web search unavailable: {exc}"
            )

        results = _rank_and_filter(
            question,
            raw_results,
            max_results,
        )

        search_mode = (
            " + ".join(used_modes)
            if used_modes
            else "Unavailable"
        )

        answer = _build_answer(
            question,
            results,
            search_mode,
        )

        if warnings:
            answer += (
                "\n\n**Search warnings:** "
                + "; ".join(warnings)
            )

        error = None

        if not results:
            error = (
                "; ".join(warnings)
                if warnings
                else "No relevant search results found."
            )

        return {
            "answer": answer,
            "route": "web",
            "source": "DDGS Web Research Agent",
            "results": results,
            "research_question": question,
            "searched_at": searched_at,
            "search_mode": search_mode,
            "warnings": warnings,
            "error": error,
        }

    except Exception as exc:
        message = str(exc)

        return {
            "answer": (
                "The Web Research Agent could not "
                "complete the search.\n\n"
                f"Details: {message}"
            ),
            "route": "web",
            "source": "DDGS Web Research Agent",
            "results": [],
            "research_question": (
                question
                if isinstance(question, str)
                else ""
            ),
            "searched_at": searched_at,
            "search_mode": None,
            "warnings": [],
            "error": message,
        }


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":
    print("=" * 65)
    print("WEB AGENT - RELEVANCE TEST")
    print("=" * 65)

    questions = [
        "Latest news about AI in data analytics this week",
        "Recent trends in artificial intelligence for business analytics",
    ]

    for question in questions:
        print("\n" + "-" * 65)
        print("Question:", question)
        print("\nSearching...\n")

        result = run_web_agent(question)

        print(result["answer"])

        print(
            "\nResults found:",
            len(result["results"]),
        )

        print(
            "Search mode:",
            result["search_mode"],
        )

        for item in result["results"]:
            print(
                f"Relevance: {item['relevance_score']} "
                f"| {item['title']}"
            )

        if result.get("error"):
            print(
                "\nError:",
                result["error"],
            )
