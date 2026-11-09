from pathlib import Path

import pytest

from core.satellite_context import current_response_intents_dir, current_response_language
from tools.calculator import calculate
from utils.response_language import detect_response_language, response

ROOT = Path(__file__).resolve().parent.parent / "data" / "intents"


@pytest.mark.parametrize("query,expected", [
    ("combien font deux plus deux", "fr"),
    ("wie viel ist zwei plus zwei", "de"),
    ("cuánto es dos más dos", "es"),
    ("2+2", "en"),
])
def test_local_language_detection(query, expected):
    assert detect_response_language(query) == expected


@pytest.mark.parametrize("language,expected", [
    ("en", "That comes to 4."),
    ("fr", "Cela fait 4."),
    ("de", "Das ergibt 4."),
    ("es", "That comes to 4."),
])
def test_calculator_uses_turn_language_with_english_fallback(language, expected):
    language_token = current_response_language.set(language)
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        assert calculate("2+2") == expected
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_turn_context_is_scoped_to_agent_run(monkeypatch):
    from types import SimpleNamespace

    from core.agent_loop import AgentLoop

    host = SimpleNamespace(_hassil_intents_dir=str(ROOT))
    loop = AgentLoop(host)
    monkeypatch.setattr(loop, "_run", lambda *args: calculate("2+2"))
    assert loop.run("combien font deux plus deux") == "Cela fait 4."
    assert current_response_language.get() == "en"


def test_missing_local_key_uses_bundled_translation(tmp_path):
    directory = tmp_path / "fr"
    directory.mkdir()
    (directory / "fulloch-responses.yaml").write_text(
        'language: fr\nschema_version: 1\nresponses:\n  time:\n    success: ["{time}"]\n'
    )
    lang_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(tmp_path))
    try:
        assert response("calculation_result", value="4") == "Cela fait 4."
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(lang_token)


def test_non_english_tts_does_not_expand_numbers_in_english():
    from core.text_utils import spoken_for_tts

    assert spoken_for_tts("Cela fait 25 pour cent.", "fr") == "Cela fait vingt-cinq pour cent."
    assert spoken_for_tts("That is 25 percent.", "en") == "That is twenty-five percent."


def test_note_error_uses_resource_language():
    from tools.notes import read_note

    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        # A missing note does not change the underlying storage operation.
        from unittest.mock import patch

        with patch("tools.notes._find_note", return_value=None), patch(
            "tools.notes._find_note_semantic", return_value=None
        ):
            assert read_note("cuisine") == "Je n'ai pas trouvé de note sur cuisine."
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_media_controls_use_localized_templates(monkeypatch):
    from tools import ha_media

    monkeypatch.setattr(ha_media, "_media_target", lambda *_args, **_kwargs: "media_player.test")
    monkeypatch.setattr(ha_media.client, "_friendly_for", lambda _entity: "Salon")
    monkeypatch.setattr(ha_media.client, "_call_service", lambda *args, **kw: kw["success_message"])
    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        assert ha_media.volume_set("Salon", 25) == "Volume de Salon réglé à 25 pour cent."
        assert ha_media.pause() == "Salon est en pause."
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_ha_error_localization_preserves_raw_observation():
    from types import SimpleNamespace

    from core.agent_loop import _localized_ha_error
    from tools.tool_registry import OutcomeText
    from utils.intents import classify_step

    original = "I couldn't reach Home Assistant."
    step = classify_step(OutcomeText(original, success=False, code="ha_unreachable"))
    language_token = current_response_language.set("de")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        localized = _localized_ha_error(SimpleNamespace(), "", None, {}, step)
        assert localized.text == "Ich konnte Home Assistant nicht erreichen."
        assert localized.raw_text == original
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_todo_list_uses_localized_join_and_preserves_artifact(monkeypatch):
    from tools import ha_calendar

    monkeypatch.setattr(ha_calendar.client, "TODO_ENTITY", "todo.test")
    monkeypatch.setattr(ha_calendar, "_fetch_pending_todo_items", lambda: [
        {"summary": "pain"}, {"summary": "lait"},
    ])
    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        result = ha_calendar.get_todo_items()
        assert result == "Vous avez 2 éléments : pain et lait."
        assert result.artifact["items"] == ["pain", "lait"]
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_verified_device_confirmation_uses_localized_target(monkeypatch):
    from tools import ha_devices

    monkeypatch.setattr(ha_devices.client, "_resolve_entity", lambda *_args, **_kw: "lock.front")
    monkeypatch.setattr(ha_devices.client, "_friendly_for", lambda _id: "Porte")
    monkeypatch.setattr(ha_devices.client, "_call_service", lambda *args, **kw: kw["success_message"])
    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        assert ha_devices.lock("Porte") == "Porte a été verrouillé."
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_weather_day_and_temperature_use_localized_templates():
    from tools import ha_state

    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        assert ha_state._format_day("Demain", {
            "condition": "partlycloudy", "templow": 10, "temperature": 20,
            "precipitation_probability": 40,
        }, ha_state._spoken_temperature_unit({})) == (
            "Demain : partiellement nuageux de 10 à 20 degrés Celsius "
            "40 pour cent de risque de pluie"
        )
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_model_timer_outcome_is_localized_without_changing_raw():
    from types import SimpleNamespace

    from core.agent_loop import _localized_timer_step
    from tools.tool_registry import OutcomeText
    from utils.intents import classify_step

    raw = "Timer timer_1 extended by 60 seconds"
    step = classify_step(OutcomeText(raw, success=True, code="timer_extended", values={
        "timer_id": "timer_1", "duration_seconds": 60,
    }))
    lang_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        result = _localized_timer_step(SimpleNamespace(), "", None, {"intent": "extend_timer"}, step)
        assert result.text == "Minuteur timer_1 prolongé de 1 minute."
        assert result.raw_text == raw
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(lang_token)


def test_finance_quote_narration_is_localized():
    from tools.finance import _format_quote

    lang_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        text = _format_quote({
            "name": "Example", "symbol": "EX", "requested_symbol": "EX",
            "price": "$100.00", "currency": "USD", "exchange": "NASDAQ",
            "change": "up 2", "change_percent": "2%",
        })
        assert "se négocie" in text
        assert "augmenté" in text
        assert "pour cent" in text
        assert "NASDAQ" in text
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(lang_token)


def test_background_job_preserves_response_language(tmp_path):
    import threading

    from core.background_jobs import BackgroundJobManager

    seen = []
    done = threading.Event()

    def worker(_job, _cancelled):
        seen.append((current_response_language.get(), current_response_intents_dir.get()))
        done.set()
        return "done", ""

    manager = BackgroundJobManager(worker, checkpoint_dir=tmp_path)
    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        manager.submit("test multilingual background task")
        assert done.wait(5)
        assert seen == [("fr", str(ROOT))]
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_voice_ack_uses_localized_text_not_english_cache():
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    from core.assistant import Assistant

    cache = [([b"audio"], 24000)]
    host = SimpleNamespace(
        ack_cache=cache, music_search_stall_cache=[], web_search_stall_cache=[],
        _turn_local=threading.local(), tts_session=None, voice_clone_prompt=None,
        play_chunks=Mock(), speak_stream=Mock(),
    )
    Assistant._play_random_ack(host, language="fr", intents_dir=str(ROOT))
    host.play_chunks.assert_not_called()
    host.speak_stream.assert_called_once_with("D'accord.", None, session=None)
    assert not hasattr(host._turn_local, "response_language")


def test_travel_summary_and_validation_are_localized():
    from tools.travel import _flight_summary, search_hotels

    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        summary = _flight_summary({"flights": [{"airline": "Example",
                                    "departure_airport": {"id": "CDG", "time": "2027-05-01 10:00"},
                                    "arrival_airport": {"id": "LHR", "time": "2027-05-01 11:00"}}],
                                   "total_duration": 60, "price": 100})
        assert "départ CDG heure locale" in summary
        assert "prix indiqué 100" in summary
        assert "Indiquez une ville" in str(search_hotels("", "2027-05-01", "2027-05-02"))
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_fastpath_accepts_additional_installed_language(tmp_path):
    from datetime import datetime, timezone

    from tools.tool_registry import OperationOutcome
    from utils.fastpath_responses import select_fastpath_response
    from utils.hassil_fastpath import MatchMetadata

    language_dir = tmp_path / "es"
    language_dir.mkdir()
    (language_dir / "fulloch-responses.yaml").write_text(
        'language: es\nschema_version: 1\nresponses:\n  time:\n    success: ["Son las {time}."]\n',
        encoding="utf-8",
    )
    match = MatchMetadata("es", "HassGetCurrentTime", "HassGetCurrentTime", {}, None)
    outcome = OperationOutcome(True, "current_time", {"datetime": datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc)})
    assert select_fastpath_response(match, outcome, "original", tmp_path, language_confirmed=True) == "Son las 13:30."


def test_bundled_response_packs_match_english_placeholders():
    from string import Formatter

    from utils.fastpath_responses import load_responses

    def fields(template):
        return {field for _, field, _, _ in Formatter().parse(template) if field is not None}

    packs = {language: load_responses(ROOT, language) for language in ("en", "fr", "de")}
    assert packs["en"]
    for language in ("fr", "de"):
        assert packs[language].keys() == packs["en"].keys()
        for key, outcomes in packs["en"].items():
            assert packs[language][key].keys() == outcomes.keys()
            for outcome, templates in outcomes.items():
                assert fields(packs[language][key][outcome][0]) == fields(templates[0])


def test_note_search_translates_the_result_wrapper(monkeypatch):
    from tools import notes

    monkeypatch.setattr(notes, "_keyword_search", lambda _query: [("Cuisine", "pain", "Cuisine.md"), ("Liste", "lait", "Liste.md")])
    monkeypatch.setattr(notes, "_semantic_search", lambda _query: [])
    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        result = notes.search_notes("repas")
        assert "2 résultats possibles" in result
        assert "dans « Cuisine » : pain" in result
        assert result.artifact["query"] == "repas"
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_web_search_localizes_wrapper_without_translating_source(monkeypatch):
    from tools import search_web

    monkeypatch.setattr(search_web, "searxng_search", lambda *args, **kwargs: [
        {"url": "https://example.test/article", "content": "Original source text."},
    ])
    monkeypatch.setattr(search_web, "fetch_website_summary", lambda *args, **kwargs: "Original source text.")
    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        result = search_web.external_information("recherche météo")
        assert result.startswith("User question: ")
        assert "Une recherche web a trouvé" in result
        assert "Original source text." in result
        assert result.thinking_status == "evidence"
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)


def test_saved_report_localizes_headings_without_changing_code(tmp_path, monkeypatch):
    from core import thinking_reports
    from core.background_jobs import BackgroundJob, JobSnapshot

    monkeypatch.setattr(thinking_reports.notes_root, "get_notes_root", lambda: tmp_path)
    monkeypatch.setattr(thinking_reports.notes, "_after_write", lambda _path: None)
    job = BackgroundJob(
        "test-report", JobSnapshot("Exemple", response_language="fr", response_intents_dir=str(ROOT)),
        summary="## Summary\n\nRésultat vérifié.\n\n```text\n## Summary\n```\n",
    )
    note_id = thinking_reports.save_report(job)
    report = (tmp_path / f"{note_id}.md").read_text(encoding="utf-8")
    assert "## Résumé\n\nRésultat vérifié." in report
    assert "```text\n## Summary\n```" in report
    assert thinking_reports.report_summary(report).startswith("Résultat vérifié.")


def test_research_stage_localizes_known_labels_and_preserves_titles():
    from utils.response_language import research_stage

    language_token = current_response_language.set("fr")
    dir_token = current_response_intents_dir.set(str(ROOT))
    try:
        assert research_stage("Planning research") == "Planification de la recherche"
        assert research_stage("Researching Quantum mechanics") == "Recherche sur Quantum mechanics"
        assert research_stage("Unknown stage") == "Unknown stage"
    finally:
        current_response_intents_dir.reset(dir_token)
        current_response_language.reset(language_token)
