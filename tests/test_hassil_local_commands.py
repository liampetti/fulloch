"""Local multilingual intents use existing tools without requiring HA discovery."""

import shutil
from pathlib import Path

import pytest

from tools import ha_client
from utils.hassil_fastpath import HassilFastPath
from utils.intent_catch import catch_with_hassil

PACKS = Path(__file__).parent / "fixtures" / "hassil"


@pytest.fixture
def matchers(monkeypatch, tmp_path):
    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    shutil.copytree(PACKS, tmp_path, dirs_exist_ok=True)
    (tmp_path / "es" / "fulloch-fastpath.yaml").write_text(
        "hassil:\n  local:\n"
        "    HassMediaPause: ['(pausa|para|detén) [la|mi] música']\n"
        "    HassMediaUnpause: ['(reanuda|continúa) [la|mi] música']\n"
        "    HassMediaSearchAndPlay: ['(reproduce|pon) {search_query}']\n"
        "    FullochVolumeUp: ['(sube|aumenta) [el] volumen', 'más fuerte']\n"
        "    FullochVolumeDown: ['(baja|reduce) [el] volumen', 'más bajo']\n",
        encoding="utf-8",
    )
    return {language: HassilFastPath([language], str(tmp_path)) for language in ("de", "fr", "es")}


@pytest.mark.parametrize("language,text,intent,args", [
    ("de", "um wie viel uhr ist es", "get_current_time", []),
    ("de", "um wie viel uhr ist das", "get_current_time", []),
    ("de", "wie viel uhr ist das", "get_current_time", []),
    ("de", "wie viel uhr ist es", "get_current_time", []),
    ("de", "wie spät ist das", "get_current_time", []),
    ("fr", "quelle heure est il", "get_current_time", []),
    ("es", "qué hora es", "get_current_time", []),
    ("fr", "quelle est la date", "get_current_time", []),
    ("de", "wie ist das wetter", "get_weather_forecast", []),
    ("de", "stelle einen timer auf fünf minuten", "start_countdown", ["300 seconds"]),
    ("de", "timer für 1 stunde und 5 minuten", "start_countdown", ["3900 seconds"]),
    ("fr", "minuteur de cinq minutes", "start_countdown", ["300 seconds"]),
    ("es", "temporizador de cinco minutos", "start_countdown", ["300 seconds"]),
    ("de", "stoppe den timer", "cancel_timer", []),
    ("fr", "annule le minuteur", "cancel_timer", []),
    ("es", "cancela el temporizador", "cancel_timer", []),
    ("de", "wie lange läuft der timer noch", "get_timer_status", []),
    ("de", "pausiere musik", "pause", []),
    ("de", "spiele musik weiter", "resume", []),
    ("de", "stopp", "pause", []),
    ("de", "lauter", "ha_volume_up", []),
    ("de", "leiser", "ha_volume_down", []),
    ("de", "nächster titel", "skip", []),
    ("fr", "mets la musique en pause", "pause", []),
    ("fr", "reprends la musique", "resume", []),
    ("fr", "plus fort", "ha_volume_up", []),
    ("fr", "moins fort", "ha_volume_down", []),
    ("es", "para la música", "pause", []),
    ("es", "sube el volumen", "ha_volume_up", []),
    ("de", "spiele jazz", "play_song", ["jazz"]),
    ("de", "spiele musik", "play_song", ["music"]),
    ("fr", "joue du jazz", "play_song", ["du jazz"]),
    ("fr", "joue la musique", "play_song", ["music"]),
    ("es", "reproduce jazz", "play_song", ["jazz"]),
    ("es", "pon música", "play_song", ["music"]),
    ("de", "stelle die lautstärke auf 40 prozent", "ha_volume_set", [None, 40]),
])
def test_local_intents_without_ha_cache(matchers, language, text, intent, args):
    assert matchers[language].match(text) == {"actions": [{"intent": intent, "args": args}]}


def test_user_supplied_language_fastpath_precedes_hassil(tmp_path, monkeypatch):
    shutil.copytree(PACKS / "es", tmp_path / "es")
    (tmp_path / "es" / "fulloch-fastpath.yaml").write_text(
        "commands:\n  - {operation: time, pattern: 'mi forma de pedir la hora'}\n", encoding="utf-8",
    )
    monkeypatch.setattr(ha_client, "_loaded", False)
    monkeypatch.setattr(ha_client, "HA_TOKEN", "")
    matcher = HassilFastPath(["es"], str(tmp_path))
    assert catch_with_hassil("mi forma de pedir la hora", matcher, voice=True) == {
        "actions": [{"intent": "get_current_time", "args": []}],
    }


@pytest.mark.parametrize("language,text,expected", [
    ("de", "stelle einen timer auf fünf minuten", {"intent": "start_countdown", "args": ["300 seconds"]}),
    ("fr", "joue du jazz", {"intent": "play_song", "args": ["du jazz"]}),
])
def test_yaml_music_and_timer_use_hassil_argument_validation(matchers, language, text, expected):
    assert catch_with_hassil(text, matchers[language], voice=True) == {"actions": [expected]}


def test_yaml_music_does_not_swallow_an_explicit_unknown_room(matchers):
    text = "spiele jazz in der küche"
    assert catch_with_hassil(text, matchers["de"], voice=True) == text


@pytest.mark.parametrize("text", [
    "stelle einen timer namens pizza auf fünf minuten",
    "stoppe den pizza timer",
    "stoppe alle timer",
    "wie spät ist es um acht uhr",
    "wie viel uhr ist es in london",
    "um wie viel uhr ist das in berlin",
    "spiele jazz in der küche",
    "spiele jazz um acht uhr",
])
def test_unsupported_target_or_qualifier_never_becomes_default_command(matchers, text):
    assert matchers["de"].match(text) is None


def test_targeted_media_and_denylist(matchers, monkeypatch):
    monkeypatch.setattr(ha_client, "_loaded", True)
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES", {"küchenradio": "media_player.kitchen"})
    monkeypatch.setattr(ha_client, "_ENTITY_ALIASES_MULTI", {"küchenradio": ["media_player.kitchen"]})
    monkeypatch.setattr(ha_client, "_AREA_MAP", {"kitchen": "küche"})
    monkeypatch.setattr(ha_client, "_FLOOR_MAP", {})
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset())
    matcher = matchers["de"]
    assert matcher.match("spiele jazz in der küche") == {
        "actions": [{"intent": "play_song", "args": ["jazz", None, "kitchen"]}],
    }
    assert matcher.match("spiele jazz am küchenradio") == {
        "actions": [{"intent": "play_song", "args": ["jazz", None, "media_player.kitchen"]}],
    }
    monkeypatch.setattr(ha_client, "_DENIED_ENTITIES", frozenset({"media_player.kitchen"}))
    assert matcher.match("spiele jazz am küchenradio") is None


def test_english_local_rule_still_precedes_multilingual_matcher(matchers):
    assert catch_with_hassil("stop", matchers["de"]) == {"actions": [{"intent": "pause", "args": []}]}


def test_german_time_mistranscription_uses_only_enabled_language(matchers):
    text = "um wie viel uhr ist das"
    assert catch_with_hassil(text, matchers["de"]) == {
        "actions": [{"intent": "get_current_time", "args": []}],
    }
    assert matchers["fr"].match(text) is None
    assert matchers["es"].match(text) is None


def test_translated_timer_duration_is_understood_by_the_local_timer_tool(matchers):
    from tools.time_tools import _parse_duration

    emission = matchers["de"].match("timer für 1 stunde und 5 minuten")
    assert _parse_duration(emission["actions"][0]["args"][0]) == 3900
