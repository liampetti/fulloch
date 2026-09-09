"""Bounded, read-only market research through SerpApi Google Finance."""

import logging
import math
import os
import re

import requests

from server.credentials_store import get_credential
from utils.local_time import now

from ._config import config
from .thinking_playbooks import thinking_playbook
from .tool_registry import ThinkingResult, tool

API_URL = "https://serpapi.com/search.json"
TIMEOUT_S = 10
MAX_WATCHLIST = 8
MAX_NEWS_ITEMS = 3
MAX_GRAPH_POINTS = 48
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*(?::[A-Za-z0-9][A-Za-z0-9.-]*)?$")
_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_EXCHANGE_CODES = frozenset({"AMEX", "ASX", "HKEX", "LSE", "NASDAQ", "NYSE", "NZX", "SSE", "TSX"})
_CURRENCY_NAMES = {
    "AUD": "Australian dollars",
    "EUR": "euros",
    "GBP": "British pounds",
    "JPY": "Japanese yen",
    "USD": "US dollars",
}

logger = logging.getLogger(__name__)


def _api_key() -> str:
    return os.environ.get("SERPAPI_API_KEY") or get_credential("serpapi_api_key")


def _available() -> bool:
    return bool(_api_key())


def _watchlist() -> list[str]:
    values = (config.get("finance") or {}).get("watchlist") or []
    if not isinstance(values, list):
        return []
    symbols: list[str] = []
    for value in values:
        symbol = _provider_symbol(str(value).strip().upper())
        if symbol and _SYMBOL_RE.fullmatch(symbol) and symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) == MAX_WATCHLIST:
            break
    return symbols


def _provider_symbol(symbol: str) -> str:
    """Accept common EXCHANGE:SYMBOL config notation and SerpApi's SYMBOL:EXCHANGE."""
    left, separator, right = symbol.partition(":")
    return f"{right}:{left}" if separator and left in _EXCHANGE_CODES else symbol


def _configured_symbol(symbol: str) -> str:
    """Trust configured exchanges; otherwise use NASDAQ rather than an inferred suffix."""
    left = symbol.partition(":")[0]
    matches = [candidate for candidate in _watchlist() if candidate.partition(":")[0] == left]
    if len(matches) == 1:
        return matches[0]
    return f"{left}:NASDAQ"


def _text(value: object, maximum: int = 300) -> str | None:
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    return text[:maximum] if text else None


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(mapping: dict, *names: str) -> str | None:
    for name in names:
        value = _text(mapping.get(name))
        if value is not None:
            return value
    return None


def _source_url(payload: dict) -> str | None:
    metadata = payload.get("search_metadata")
    if isinstance(metadata, dict):
        return _first(metadata, "google_finance_url", "source_url")
    return None


def _quote_source_url(payload: dict, summary: dict) -> str | None:
    source_url = _source_url(payload) or _first(summary, "link", "url")
    if source_url:
        return source_url
    symbol = _first(summary, "stock", "symbol")
    exchange = _first(summary, "exchange")
    if symbol and exchange:
        identifier = symbol if ":" in symbol else f"{symbol}:{exchange}"
        return f"https://www.google.com/finance/quote/{identifier}"
    return None


def _news(payload: dict) -> list[dict]:
    items = payload.get("news_results") or payload.get("news") or []
    if not isinstance(items, list):
        return []
    news = []
    for item in items[:MAX_NEWS_ITEMS]:
        if not isinstance(item, dict):
            continue
        title = _first(item, "title")
        if title is None:
            continue
        news.append(
            {
                "title": title,
                "publisher": _first(item, "source", "publisher"),
                "published_at": _first(item, "date", "published_date"),
                "url": _first(item, "link", "url"),
            }
        )
    return news


def _financials(payload: dict) -> list[dict]:
    sections = payload.get("financials") or []
    if not isinstance(sections, list):
        return []
    output = []
    for section in sections[:3]:
        if not isinstance(section, dict):
            continue
        periods = []
        for result in (section.get("results") or [])[:3]:
            if not isinstance(result, dict):
                continue
            metrics = []
            for item in (result.get("table") or [])[:5]:
                if not isinstance(item, dict):
                    continue
                label = _first(item, "title", "label")
                value = _first(item, "value")
                if label and value:
                    metrics.append({"label": label, "value": value})
            if metrics:
                periods.append(
                    {
                        "date": _first(result, "date"),
                        "period_type": _first(result, "period_type"),
                        "metrics": metrics,
                    }
                )
        if periods:
            output.append(
                {"title": _first(section, "title", "name") or "Financials", "periods": periods}
            )
    return output


def _key_stats(payload: dict) -> list[dict]:
    knowledge_graph = payload.get("knowledge_graph")
    stats = knowledge_graph.get("key_stats") if isinstance(knowledge_graph, dict) else None
    values = stats.get("stats") if isinstance(stats, dict) else None
    if not isinstance(values, list):
        return []
    output = []
    for item in values[:10]:
        if not isinstance(item, dict):
            continue
        label = _first(item, "label", "title")
        value = _first(item, "value")
        if label and value:
            output.append({"label": label, "value": value})
    return output


def _movement(item: dict) -> tuple[str | None, str | None]:
    movement = item.get("price_movement") if isinstance(item.get("price_movement"), dict) else {}
    value = _first(movement, "value", "price", "change") or _first(item, "change")
    direction = _first(movement, "movement")
    change = " ".join(part for part in (direction, value) if part) or None
    percentage = _first(movement, "percentage", "change_percent") or _first(item, "change_percent")
    return change, percentage


def _cache_mode(fresh: bool) -> str:
    return "cache bypass requested" if fresh else "provider cache allowed (up to one hour)"


def _fresh(value: object) -> bool:
    """Accept a boolean tool argument and the worker fallback's task text."""
    return value is True or (
        isinstance(value, str)
        and bool(re.search(r"\b(now|current|latest|today)\b", value, re.IGNORECASE))
    )


def _window_and_fresh(window: object, fresh: object) -> tuple[object, bool]:
    """Recover common planner encodings for optional chart and freshness arguments."""
    if isinstance(window, bool):
        return "5D", _fresh(fresh) or window
    if window is None:
        return "5D", _fresh(fresh)
    if isinstance(window, str) and window.strip().lower() in {"fresh", "today", "current", "latest"}:
        return "5D", True
    return window, _fresh(fresh)


def _graph_points(graph: list) -> list[dict]:
    points = []
    for point in graph:
        if not isinstance(point, dict):
            continue
        value = _number(point.get("price"))
        timestamp = _first(point, "date", "timestamp")
        if value is not None and timestamp:
            points.append({"time": timestamp, "value": value})
    if len(points) <= MAX_GRAPH_POINTS:
        return points
    indexes = {
        round(index * (len(points) - 1) / (MAX_GRAPH_POINTS - 1))
        for index in range(MAX_GRAPH_POINTS)
    }
    return [point for index, point in enumerate(points) if index in indexes]


def _quote(
    payload: dict, requested_symbol: str, window: str, retrieved_at: str, fresh: bool
) -> dict:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    change, change_percent = _movement(summary)
    graph = payload.get("graph") if isinstance(payload.get("graph"), list) else []
    latest = graph[-1] if graph and isinstance(graph[-1], dict) else {}
    numeric_price = _number(summary.get("extracted_price"))
    if numeric_price is None:
        numeric_price = _number(latest.get("price"))
    return {
        "requested_symbol": requested_symbol,
        "name": _first(summary, "title", "name"),
        "symbol": _first(summary, "stock", "symbol") or requested_symbol,
        "exchange": _first(summary, "exchange"),
        "currency": _first(summary, "currency") or _first(latest, "currency"),
        "price": _first(summary, "price", "current_price") or _first(latest, "price", "value"),
        "numeric_price": numeric_price,
        "change": change,
        "change_percent": change_percent,
        "quoted_at": _first(summary, "date", "timestamp", "price_date")
        or _first(latest, "date", "timestamp"),
        "market_status": _first(summary, "market_status"),
        "window": window,
        "volume": _first(latest, "volume"),
        "points": _graph_points(graph),
        "source_url": _quote_source_url(payload, summary),
        "retrieved_at": retrieved_at,
        "cache_mode": _cache_mode(fresh),
        "key_stats": _key_stats(payload),
        "financials": _financials(payload),
        "news": _news(payload),
    }


def _matches_requested_symbol(requested: str, quote: dict) -> bool:
    if ":" not in requested:
        return True
    symbol, exchange = requested.rsplit(":", 1)
    returned_symbol = str(quote.get("symbol") or "").upper()
    returned_exchange = str(quote.get("exchange") or "").upper()
    return returned_symbol == symbol and returned_exchange == exchange


def _spoken_amount(value: object, currency: str | None) -> str:
    text = _text(value) or "price not supplied"
    number_text = re.sub(r"[$\u00a3\u20ac]|\b[A-Z]{3}\b", "", text).replace(",", "").strip()
    number = _number(number_text)
    if number is None:
        return text
    amount = f"{number:,.2f}"
    return f"{amount} {_CURRENCY_NAMES.get(currency or '', currency)}" if currency else amount


def _spoken_percent(value: object) -> str | None:
    number = _number(str(value).replace("%", "").replace(",", ""))
    return f"{number:.2f} percent" if number is not None else None


def _format_quote(quote: dict) -> str:
    identity = quote["name"] or quote["symbol"] or quote["requested_symbol"]
    price = _spoken_amount(quote["price"], quote["currency"])
    location = f" on {quote['exchange']}" if quote["exchange"] else ""
    sentences = [f"{identity} is trading at {price}{location}."]
    movement = quote["change"] or ""
    match = re.fullmatch(r"(up|down)\s+(.+)", movement, re.IGNORECASE)
    percent = _spoken_percent(quote["change_percent"])
    if match:
        change = _spoken_amount(match.group(2), quote["currency"])
        detail = f"It is {match.group(1).lower()} {change}"
        sentences.append(f"{detail}{f', or {percent}' if percent else ''}.")
    elif percent:
        sentences.append(f"It has moved {percent}.")
    return " ".join(sentences)


def _market_records(payload: dict) -> list[dict]:
    markets = payload.get("markets")
    groups = (
        markets.items()
        if isinstance(markets, dict)
        else enumerate(markets)
        if isinstance(markets, list)
        else ()
    )
    records = []
    for group, items in groups:
        if isinstance(items, dict):
            items = items.get("items")
        if not isinstance(items, list):
            continue
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            name = _first(item, "name", "title")
            if name is None:
                continue
            change, change_percent = _movement(item)
            records.append(
                {
                    "group": str(group),
                    "name": name,
                    "symbol": _first(item, "stock", "symbol"),
                    "price": _first(item, "price"),
                    "currency": _first(item, "currency"),
                    "change": change,
                    "change_percent": change_percent,
                    "source_url": _first(item, "link", "url"),
                }
            )
    return records


def _request(params: dict) -> tuple[dict | None, str | None]:
    try:
        response = requests.get(
            API_URL, params={**params, "api_key": _api_key()}, timeout=TIMEOUT_S
        )
        if response.status_code in (401, 403):
            return (
                None,
                "Finance search authentication failed; update serpapi_api_key in credentials.json.",
            )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None, "Finance search is temporarily unavailable."
    except ValueError:
        return None, "Finance search returned an invalid response."
    if not isinstance(payload, dict):
        return None, "Finance search returned an invalid response."
    if payload.get("error"):
        logger.warning("Finance provider rejected request: %s", _text(payload["error"]))
        return None, "Finance search provider rejected the request."
    return payload, None


def _get_quote(symbol: str, window: str, fresh: bool) -> ThinkingResult:
    if not _available():
        return ThinkingResult(
            "Finance research is not configured; add serpapi_api_key to credentials.json.",
            status="unavailable",
            scope="A SerpApi key is required for finance research.",
        )
    if not isinstance(symbol, str) or not _SYMBOL_RE.fullmatch(symbol.strip()):
        return ThinkingResult(
            "Ask for a stock symbol such as NASDAQ:TSLA.",
            status="needs_input",
            scope="A valid stock symbol is required; company names can be ambiguous.",
        )
    if window not in {"1D", "5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"}:
        return ThinkingResult(
            "Choose a chart window from 1D, 5D, 1M, 6M, YTD, 1Y, 5Y, or MAX.",
            status="needs_input",
            scope="The requested chart window is unsupported.",
        )
    symbol = _configured_symbol(_provider_symbol(symbol.strip().upper()))
    payload, error = _request(
        {
            "engine": "google_finance",
            "q": symbol,
            "window": window,
            "no_cache": str(bool(fresh)).lower(),
        }
    )
    if error:
        if error == "Finance search provider rejected the request.":
            return ThinkingResult(
                "I couldn't find information on that stock.",
                status="rejected",
                scope="The finance provider returned no matching listing.",
            )
        return ThinkingResult(
            error,
            status="unavailable",
            scope="The finance provider did not complete the quote request.",
        )
    quote = _quote(payload, symbol, window, now().strftime("%Y-%m-%d %H:%M %Z"), fresh)
    if quote["price"] is None and quote["name"] is None:
        return ThinkingResult(
            f"I couldn't verify an instrument for {symbol}.",
            status="rejected",
            scope="The finance provider returned no identifiable quote.",
        )
    if not _matches_requested_symbol(symbol, quote):
        return ThinkingResult(
            f"I couldn't verify that the returned listing matches {symbol}.",
            status="rejected",
            scope="The provider returned a different or incomplete instrument identity.",
        )
    return ThinkingResult(
        _format_quote(quote),
        evidence={
            "quote": quote,
            "disclosure": "Market data may be delayed; verify before trading.",
        },
        scope=f"One {window} market snapshot for {quote['symbol']} from Google Finance via SerpApi.",
        next_actions=("get_finance_quote", "get_market_brief"),
        artifact={"type": "finance_quote", "quote": quote},
    )


@tool(
    name="get_finance_quote",
    description="Retrieve a read-only Google Finance snapshot for one exact stock symbol, such as TSLA:NASDAQ. Exchange-first symbols such as NASDAQ:TSLA are also accepted. Returns facts and sources, not investment advice.",
    available=_available,
    thinking_outcome=True,
)
def get_finance_quote(symbol: str, window: str = "5D", fresh: bool = False) -> str:
    window, fresh = _window_and_fresh(window, fresh)
    return _get_quote(symbol, window, fresh)


@tool(
    name="get_exchange_rate",
    description="Retrieve a read-only current foreign-exchange market rate for two ISO currency codes, optionally converting a supplied amount. Returns market data, not bank/transfer pricing or FX trading advice.",
    available=_available,
    thinking_outcome=True,
)
def get_exchange_rate(
    base_currency: str, quote_currency: str, amount: float | None = None, fresh: bool = False
) -> str:
    if isinstance(amount, bool):
        fresh = fresh or amount
        amount = None
    base = base_currency.strip().upper() if isinstance(base_currency, str) else ""
    quote_currency = quote_currency.strip().upper() if isinstance(quote_currency, str) else ""
    if not _CURRENCY_RE.fullmatch(base) or not _CURRENCY_RE.fullmatch(quote_currency):
        return ThinkingResult(
            "Use three-letter ISO currency codes, such as AUD and USD.",
            status="needs_input",
            scope="A base and quote ISO currency code are required.",
        )
    if base == quote_currency:
        return ThinkingResult(
            "Choose two different currencies.",
            status="needs_input",
            scope="A currency conversion requires different base and quote currencies.",
        )
    if amount is not None and (
        not isinstance(amount, (int, float))
        or isinstance(amount, bool)
        or not math.isfinite(amount)
        or amount < 0
    ):
        return ThinkingResult(
            "Use a non-negative numeric amount.",
            status="needs_input",
            scope="The requested conversion amount is invalid.",
        )
    result = _get_quote(f"{base}-{quote_currency}", "1D", _fresh(fresh))
    if result.thinking_status != "evidence":
        return result
    quote = result.evidence["quote"]
    rate = quote["numeric_price"]
    if rate is None:
        return ThinkingResult(
            f"I retrieved {base}/{quote_currency} but the provider did not supply a numeric rate.",
            status="rejected",
            scope="The provider returned a currency instrument without a numeric rate for conversion.",
        )
    conversion = None if amount is None else round(amount * rate, 6)
    text = f"{base}/{quote_currency}: 1 {base} = {rate:g} {quote_currency}."
    if conversion is not None:
        text += f" {amount:g} {base} is approximately {conversion:g} {quote_currency}."
    text += f" Quote timestamp: {quote['quoted_at'] or 'provider timestamp not supplied'}; retrieved {quote['retrieved_at']}; {quote['cache_mode']}."
    exchange_rate = {
        "pair": f"{base}-{quote_currency}",
        "base_currency": base,
        "quote_currency": quote_currency,
        "rate": rate,
        "amount": amount,
        "converted_amount": conversion,
        "quoted_at": quote["quoted_at"],
        "retrieved_at": quote["retrieved_at"],
        "cache_mode": quote["cache_mode"],
        "source_url": quote["source_url"],
    }
    return ThinkingResult(
        text,
        evidence={
            "exchange_rate": exchange_rate,
            "disclosure": "FX market data may be delayed and excludes bank spreads, fees, and transfer pricing.",
        },
        scope=f"One {base}/{quote_currency} market-rate snapshot from Google Finance via SerpApi.",
        next_actions=("get_exchange_rate",),
        artifact={"type": "finance_exchange_rate", "exchange_rate": exchange_rate},
    )


@tool(
    name="get_watchlist_brief",
    description="Retrieve bounded market snapshots for symbols configured in finance.watchlist. Use for a user's stocks or watchlist update.",
    available=_available,
    deep_think_only=True,
    thinking_outcome=True,
)
def get_watchlist_brief(window: str = "5D", fresh: bool = False) -> str:
    window, fresh = _window_and_fresh(window, fresh)
    symbols = _watchlist()
    if not symbols:
        return ThinkingResult(
            "No finance watchlist is configured.",
            status="needs_input",
            scope="Add symbols under finance.watchlist or ask about one exact stock symbol.",
        )
    results = [_get_quote(symbol, window, fresh) for symbol in symbols]
    quotes = [
        result.evidence["quote"] for result in results if result.thinking_status == "evidence"
    ]
    if not quotes:
        return ThinkingResult(
            "I couldn't retrieve the configured watchlist.",
            status="unavailable",
            scope="No watchlist quote request completed successfully.",
            next_actions=("get_watchlist_brief",),
        )
    return ThinkingResult(
        "\n".join(_format_quote(quote) for quote in quotes),
        evidence={
            "watchlist": quotes,
            "disclosure": "Market data may be delayed; verify before trading.",
        },
        scope=f"{len(quotes)} configured watchlist snapshots, capped at {MAX_WATCHLIST} symbols.",
        next_actions=("get_finance_quote", "get_market_brief"),
        artifact={"type": "finance_watchlist", "window": window, "quotes": quotes},
    )


@tool(
    name="get_market_brief",
    description="Retrieve a read-only broad market snapshot from Google Finance. Use for today's finance or stock-market situation.",
    available=_available,
    deep_think_only=True,
    thinking_outcome=True,
)
def get_market_brief(fresh: bool = False) -> str:
    fresh = _fresh(fresh)
    if not _available():
        return ThinkingResult(
            "Finance research is not configured; add serpapi_api_key to credentials.json.",
            status="unavailable",
            scope="A SerpApi key is required for finance research.",
        )
    payload, error = _request(
        {
            "engine": "google_finance_markets",
            "trend": "indexes",
            "no_cache": str(bool(fresh)).lower(),
        }
    )
    if error:
        return ThinkingResult(
            error,
            status="unavailable",
            scope="The finance provider did not complete the market request.",
        )
    records = _market_records(payload)
    if not records:
        return ThinkingResult(
            "I couldn't retrieve a broad market snapshot.",
            status="rejected",
            scope="The provider returned no usable market-index records.",
        )
    retrieved_at = now().strftime("%Y-%m-%d %H:%M %Z")
    return ThinkingResult(
        "\n".join(
            f"{item['name']}: {item['price'] or 'price not supplied'} {' '.join(part for part in (item['change'], item['change_percent']) if part)}".strip()
            for item in records
        ),
        evidence={
            "markets": records,
            "source_url": _source_url(payload),
            "retrieved_at": retrieved_at,
            "cache_mode": _cache_mode(fresh),
            "news": _news(payload),
            "disclosure": "Market data may be delayed; verify before trading.",
        },
        scope=f"{len(records)} broad market records from Google Finance via SerpApi.",
        next_actions=("get_watchlist_brief", "get_finance_quote"),
        artifact={
            "type": "finance_market",
            "markets": records,
            "retrieved_at": retrieved_at,
            "cache_mode": _cache_mode(fresh),
            "news": _news(payload),
        },
    )


def _is_finance_request(request: str) -> bool:
    return bool(
        re.search(
            r"\b(stock|stocks|shares|market|markets|finance|financial|ticker|portfolio|watchlist|invest)\b",
            request,
            re.IGNORECASE,
        )
    )


thinking_playbook(
    name="finance research",
    triggers=(),
    capabilities=(
        "get_finance_quote",
        "get_exchange_rate",
        "get_watchlist_brief",
        "get_market_brief",
    ),
    solve_path=(
        "For a configured-stocks or portfolio update, start with get_watchlist_brief.",
        "For one named stock, retrieve an exact symbol with get_finance_quote; ask a clarification rather than guessing an ambiguous listing.",
        "For a broad finance-news or market summary, start with get_watchlist_brief using a 5D window when a watchlist is configured, then call get_market_brief for market context.",
        "For a live currency conversion, call get_exchange_rate with ISO base and quote currency codes; explain that it excludes bank spreads and fees.",
        "Pass fresh=true when the user explicitly asks for today, now, current, or latest data; otherwise use the default cache-allowed mode.",
        "Separate reported facts from possible catalysts and data gaps; state quote timing, exchange, currency, and any available source links.",
    ),
    completion_rule="The response is limited to retrieved market evidence, clearly identifies its freshness limits, and does not make a personalised investment recommendation.",
    prohibited_shortcuts=(
        "Do not claim a news item caused a price move without two independent sources.",
        "Do not give buy, sell, hold, allocation, tax, legal, margin, options, or trade-execution advice.",
    ),
    matcher=_is_finance_request,
    fallback_capability="get_market_brief",
)
