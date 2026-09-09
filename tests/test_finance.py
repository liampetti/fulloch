"""Mocked SerpApi finance research and worker-policy tests."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.capabilities import native_access_class, native_requires_deep_think
from tools.tool_registry import tool_registry


def _module(monkeypatch):
    sys.modules.pop("tools.finance", None)
    module = importlib.import_module("tools.finance")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    return module


class _Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _quote_payload():
    return {
        "search_metadata": {
            "google_finance_url": "https://www.google.com/finance/quote/TSLA:NASDAQ"
        },
        "summary": {
            "title": "Tesla Inc",
            "stock": "TSLA",
            "exchange": "NASDAQ",
            "currency": "USD",
            "price": "250.00",
            "price_movement": {"value": "2.00", "percentage": "0.8%", "movement": "Up"},
            "date": "Sep 8, 4:00 PM EDT",
        },
        "graph": [
            {"price": "250.00", "currency": "USD", "volume": 12345, "date": "Sep 8, 4:00 PM EDT"}
        ],
        "knowledge_graph": {"key_stats": {"stats": [{"label": "Open", "value": "$248.00"}]}},
        "financials": [
            {
                "title": "Income statement",
                "results": [
                    {
                        "date": "Mar 2026",
                        "period_type": "Quarterly",
                        "table": [{"title": "Revenue", "value": "1000"}],
                    }
                ],
            }
        ],
        "news_results": [
            {
                "title": "Example news",
                "source": "Publisher",
                "date": "Today",
                "link": "https://example.test/news",
            }
        ],
    }


def test_quote_normalises_provider_data_and_requests_fresh_result(monkeypatch):
    finance = _module(monkeypatch)
    calls = []

    def get(_url, params, timeout):
        calls.append((params, timeout))
        return _Response(_quote_payload())

    monkeypatch.setattr(finance.requests, "get", get)
    result = finance.get_finance_quote("NASDAQ:TSLA", fresh=True)

    assert calls == [
        (
            {
                "engine": "google_finance",
                "q": "TSLA:NASDAQ",
                "window": "5D",
                "no_cache": "true",
                "api_key": "test-key",
            },
            finance.TIMEOUT_S,
        )
    ]
    assert "test-key" not in result
    assert result.thinking_status == "evidence"
    quote = result.evidence["quote"]
    assert quote["change"] == "Up 2.00"
    assert quote["change_percent"] == "0.8%"
    assert quote["volume"] == "12345"
    assert quote["points"] == [{"time": "Sep 8, 4:00 PM EDT", "value": 250.0}]
    assert quote["cache_mode"] == "cache bypass requested"
    assert quote["key_stats"] == [{"label": "Open", "value": "$248.00"}]
    assert quote["financials"] == [
        {
            "title": "Income statement",
            "periods": [
                {
                    "date": "Mar 2026",
                    "period_type": "Quarterly",
                    "metrics": [{"label": "Revenue", "value": "1000"}],
                }
            ],
        }
    ]
    assert quote["news"][0]["url"] == "https://example.test/news"


def test_quote_text_is_short_and_tts_friendly(monkeypatch):
    finance = _module(monkeypatch)
    quote = finance._quote(_quote_payload(), "TSLA:NASDAQ", "5D", "2026-09-08 15:15 AEST", True)

    assert finance._format_quote(quote) == (
        "Tesla Inc is trading at 250.00 US dollars on NASDAQ. "
        "It is up 2.00 US dollars, or 0.80 percent."
    )


def test_quote_rejects_invalid_symbol_without_network(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setattr(
        finance.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )

    result = finance.get_finance_quote("Tesla Motors")

    assert result.thinking_status == "needs_input"
    assert "stock symbol" in result.scope


def test_boolean_quote_window_is_treated_as_fresh(monkeypatch):
    finance = _module(monkeypatch)
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    result = finance.get_finance_quote("NASDAQ:TSLA", True)

    assert result.thinking_status == "evidence"
    assert calls[0]["window"] == "5D"
    assert calls[0]["no_cache"] == "true"


def test_fresh_quote_window_is_treated_as_fresh(monkeypatch):
    finance = _module(monkeypatch)
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    result = finance.get_finance_quote("NASDAQ:TSLA", "fresh")

    assert result.thinking_status == "evidence"
    assert calls[0]["window"] == "5D"
    assert calls[0]["no_cache"] == "true"


def test_quote_uses_unambiguous_watchlist_exchange_mapping(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setitem(finance.config, "finance", {"watchlist": ["NASDAQ:TSLA"]})
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    result = finance.get_finance_quote("TSLA")

    assert result.thinking_status == "evidence"
    assert calls[0]["q"] == "TSLA:NASDAQ"


def test_quote_prefers_watchlist_exchange_over_planner_suffix(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setitem(finance.config, "finance", {"watchlist": ["NYSE:TSLA"]})
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: (
            calls.append(params)
            or _Response({**_quote_payload(), "summary": {**_quote_payload()["summary"], "exchange": "NYSE"}})
        ),
    )

    result = finance.get_finance_quote("TSLA:LSE")

    assert result.thinking_status == "evidence"
    assert calls[0]["q"] == "TSLA:NYSE"


def test_quote_defaults_unconfigured_symbol_to_nasdaq_even_with_a_suffix(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setitem(finance.config, "finance", {"watchlist": []})
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    result = finance.get_finance_quote("TSLA:LSE")

    assert result.thinking_status == "evidence"
    assert calls[0]["q"] == "TSLA:NASDAQ"


def test_quote_listing_identity_check_rejects_a_mismatched_exchange(monkeypatch):
    finance = _module(monkeypatch)

    assert finance._matches_requested_symbol("TSLA:NYSE", _quote_payload()["summary"]) is False


def test_quote_graph_is_bounded_and_keeps_both_ends(monkeypatch):
    finance = _module(monkeypatch)
    graph = [{"date": f"t{index}", "price": index} for index in range(100)]

    points = finance._graph_points(graph)

    assert len(points) == finance.MAX_GRAPH_POINTS
    assert points[0] == {"time": "t0", "value": 0.0}
    assert points[-1] == {"time": "t99", "value": 99.0}


def test_watchlist_is_deduplicated_bounded_and_collects_successes(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setitem(
        finance.config,
        "finance",
        {
            "watchlist": ["nasdaq:tsla", "NASDAQ:TSLA", "NYSE:BRK.B"]
            + [f"NASDAQ:X{i}" for i in range(10)]
        },
    )
    calls = []

    def get(_url, params, timeout):
        calls.append(params["q"])
        stock, exchange = params["q"].split(":", 1)
        payload = _quote_payload()
        payload["summary"].update({"stock": stock, "exchange": exchange})
        return _Response(payload)

    monkeypatch.setattr(finance.requests, "get", get)
    result = finance.get_watchlist_brief(None)

    assert calls == [
        "TSLA:NASDAQ",
        "BRK.B:NYSE",
        "X0:NASDAQ",
        "X1:NASDAQ",
        "X2:NASDAQ",
        "X3:NASDAQ",
        "X4:NASDAQ",
        "X5:NASDAQ",
    ]
    assert len(result.evidence["watchlist"]) == finance.MAX_WATCHLIST
    assert result.artifact["window"] == "5D"


def test_boolean_watchlist_argument_is_treated_as_fresh_not_a_window(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setitem(finance.config, "finance", {"watchlist": ["NASDAQ:TSLA"]})
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    result = finance.get_watchlist_brief(True)

    assert result.thinking_status == "evidence"
    assert calls[0]["window"] == "5D"
    assert calls[0]["no_cache"] == "true"


def test_fresh_watchlist_window_is_treated_as_fresh(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setitem(finance.config, "finance", {"watchlist": ["NASDAQ:TSLA"]})
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    result = finance.get_watchlist_brief("fresh")

    assert result.thinking_status == "evidence"
    assert calls[0]["window"] == "5D"
    assert calls[0]["no_cache"] == "true"


def test_market_brief_normalises_index_records(monkeypatch):
    finance = _module(monkeypatch)
    calls = []

    def get(_url, params, timeout):
        calls.append(params)
        return _Response(
            {
                "search_metadata": {"google_finance_url": "https://example.test/market"},
                "markets": {
                    "us": [
                        {
                            "stock": ".INX:INDEXSP",
                            "link": "https://example.test/spx",
                            "name": "S&P 500",
                            "price": "5,000",
                            "currency": "USD",
                            "price_movement": {
                                "value": "+1",
                                "percentage": "+0.02%",
                                "movement": "Up",
                            },
                        }
                    ]
                },
                "news_results": [
                    {
                        "title": "Market news",
                        "source": "Publisher",
                        "date": "Today",
                        "link": "https://example.test/market-news",
                    }
                ],
            }
        )

    monkeypatch.setattr(finance.requests, "get", get)
    result = finance.get_market_brief()

    assert calls[0]["engine"] == "google_finance_markets"
    assert calls[0]["trend"] == "indexes"
    assert result.evidence["markets"] == [
        {
            "group": "us",
            "name": "S&P 500",
            "symbol": ".INX:INDEXSP",
            "price": "5,000",
            "currency": "USD",
            "change": "Up +1",
            "change_percent": "+0.02%",
            "source_url": "https://example.test/spx",
        }
    ]
    assert result.evidence["news"][0]["url"] == "https://example.test/market-news"
    assert result.artifact["type"] == "finance_market"


def test_finance_provider_failure_is_safe_and_capabilities_are_read_only(monkeypatch):
    finance = _module(monkeypatch)

    def get(*_args, **_kwargs):
        raise finance.requests.Timeout("test-key")

    monkeypatch.setattr(finance.requests, "get", get)
    result = finance.get_finance_quote("NASDAQ:TSLA")

    assert result.thinking_status == "unavailable"
    assert "test-key" not in result
    assert native_access_class("get_finance_quote") == "read"
    assert native_requires_deep_think("get_finance_quote") is False
    assert native_requires_deep_think("get_watchlist_brief") is True
    assert native_requires_deep_think("get_market_brief") is True
    assert tool_registry._schemas["get_finance_quote"].thinking_outcome is True
    assert tool_registry._schemas["get_exchange_rate"].thinking_outcome is True


def test_quote_provider_no_result_has_a_clear_stock_not_found_message(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda *_args, **_kwargs: _Response({"error": "No matching listing"}),
    )

    result = finance.get_finance_quote("TSLA")

    assert result.thinking_status == "rejected"
    assert str(result) == "I couldn't find information on that stock."


def test_environment_key_takes_precedence_over_credentials(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setattr(finance, "get_credential", lambda _key: "stored-key")

    assert finance._api_key() == "test-key"
    monkeypatch.delenv("SERPAPI_API_KEY")
    assert finance._api_key() == "stored-key"


def test_exchange_rate_converts_an_amount_without_deep_think(monkeypatch):
    finance = _module(monkeypatch)
    payload = _quote_payload()
    payload["summary"].update(
        {
            "stock": "USD-AUD",
            "title": "US Dollar / Australian Dollar",
            "price": "1.5",
            "extracted_price": 1.5,
        }
    )
    monkeypatch.setattr(finance.requests, "get", lambda *_args, **_kwargs: _Response(payload))

    result = finance.get_exchange_rate("USD", "AUD", 100)

    assert "100 USD is approximately 150 AUD" in result
    assert result.evidence["exchange_rate"] == {
        "pair": "USD-AUD",
        "base_currency": "USD",
        "quote_currency": "AUD",
        "rate": 1.5,
        "amount": 100,
        "converted_amount": 150.0,
        "quoted_at": "Sep 8, 4:00 PM EDT",
        "retrieved_at": result.evidence["exchange_rate"]["retrieved_at"],
        "cache_mode": "provider cache allowed (up to one hour)",
        "source_url": "https://www.google.com/finance/quote/TSLA:NASDAQ",
    }
    assert native_access_class("get_exchange_rate") == "read"
    assert native_requires_deep_think("get_exchange_rate") is False


def test_exchange_rate_rejects_invalid_codes_and_amount_without_network(monkeypatch):
    finance = _module(monkeypatch)
    monkeypatch.setattr(
        finance.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )

    assert finance.get_exchange_rate("Australian dollars", "USD").thinking_status == "needs_input"
    assert finance.get_exchange_rate("AUD", "AUD").thinking_status == "needs_input"
    assert finance.get_exchange_rate("AUD", "USD", -1).thinking_status == "needs_input"


def test_boolean_exchange_rate_amount_is_treated_as_fresh(monkeypatch):
    finance = _module(monkeypatch)
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    finance.get_exchange_rate("USD", "AUD", True)

    assert calls[0]["no_cache"] == "true"


def test_provider_symbol_accepts_exchange_first_and_serpapi_order(monkeypatch):
    finance = _module(monkeypatch)

    assert finance._provider_symbol("NASDAQ:TSLA") == "TSLA:NASDAQ"
    assert finance._provider_symbol("TSLA:NASDAQ") == "TSLA:NASDAQ"


def test_finance_playbook_matches_stocks_but_not_unrelated_requests(monkeypatch):
    finance = _module(monkeypatch)
    from tools.thinking_playbooks import matching_playbooks

    capabilities = {"get_finance_quote", "get_watchlist_brief", "get_market_brief"}
    matches = matching_playbooks("Update me on my stocks", capabilities)

    assert [playbook.name for playbook in matches] == ["finance research"]
    assert matching_playbooks("What should I have for dinner?", capabilities) == []
    assert finance._available() is True


def test_finance_playbook_prioritises_watchlist_before_market_context(monkeypatch):
    _module(monkeypatch)
    from tools.thinking_playbooks import matching_playbooks

    playbook = matching_playbooks(
        "Give me today's finance summary",
        {"get_finance_quote", "get_watchlist_brief", "get_market_brief"},
    )[0]

    assert "start with get_watchlist_brief" in playbook.solve_path[2]
    assert "then call get_market_brief" in playbook.solve_path[2]


def test_non_fresh_task_text_does_not_bypass_cache(monkeypatch):
    finance = _module(monkeypatch)
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: calls.append(params) or _Response(_quote_payload()),
    )

    finance.get_market_brief("Summarise the markets")

    assert calls[0]["no_cache"] == "false"


def test_current_task_text_bypasses_cache_when_used_as_worker_fallback(monkeypatch):
    finance = _module(monkeypatch)
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: (
            calls.append(params)
            or _Response({"markets": {"us": [{"name": "S&P 500", "price": "5,000"}]}})
        ),
    )

    finance.get_market_brief("Give me the current market update")

    assert calls[0]["no_cache"] == "true"


def test_today_task_text_bypasses_cache(monkeypatch):
    finance = _module(monkeypatch)
    calls = []
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda _url, params, timeout: (
            calls.append(params)
            or _Response({"markets": {"us": [{"name": "S&P 500", "price": "5,000"}]}})
        ),
    )

    finance.get_market_brief("Give me today's finance summary")

    assert calls[0]["no_cache"] == "true"


def test_finance_report_prompt_requires_neutral_non_advice_summary():
    from utils.prompts import get_thinking_report_prompt

    prompt = get_thinking_report_prompt("Should I buy Tesla shares?", "Retrieved quote")

    assert "Do not recommend buying" in prompt
    assert "Verify quotes before trading" in prompt
