"""Restricted, separately loaded Fulloch response resources.

Missing or invalid files produce no templates; callers retain the original
truthful observation.
"""

import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from string import Formatter

from babel.core import UnknownLocaleError
from babel.dates import format_date, format_time
from babel.lists import format_list
from babel.numbers import format_decimal
from babel.units import format_unit
from ruamel.yaml import YAML

from tools.tool_registry import OperationOutcome
from utils.hassil_fastpath import MatchMetadata

logger = logging.getLogger(__name__)

# A key's allowed fields are fixed by code, not by editable resources.
RESPONSE_FIELDS = {
    "time": frozenset({"time"}),
    "date": frozenset({"date"}),
    "timer_started": frozenset({"duration"}),
    "timer_cancelled": frozenset(),
    "timer_completed": frozenset(),
    "timer_status": frozenset({"duration"}),
    "timer_status_multiple": frozenset({"durations"}),
    "timer_none": frozenset(),
    "timer_ambiguous": frozenset(),
    "timer_limit": frozenset(),
    "timer_too_long": frozenset(),
    "timer_invalid_duration": frozenset(),
    "light_brightness": frozenset({"value"}),
    "climate_temperature": frozenset({"value"}),
    "cover_position": frozenset({"value"}),
    "media_volume": frozenset({"value"}),
    "ha_unreachable": frozenset(),
    "ha_timeout": frozenset(),
    "ha_not_configured": frozenset(),
    "voice_denied": frozenset(),
    "light_on": frozenset({"target"}),
    "light_off": frozenset({"target"}),
    "operation_failed": frozenset(),
    "calculation_result": frozenset({"value"}),
    "calculation_divide_by_zero": frozenset(),
    "notes_empty": frozenset(),
    "note_not_found": frozenset({"title"}),
    "note_read_failed": frozenset({"title"}),
    "note_already_exists": frozenset({"title"}),
    "note_update_failed": frozenset({"title"}),
    "note_updated": frozenset({"title"}),
    "note_save_failed": frozenset({"title"}),
    "note_saved": frozenset({"title"}),
    "note_append_empty": frozenset(),
    "note_append_failed": frozenset({"title"}),
    "note_appended": frozenset({"title"}),
    "daily_log_empty": frozenset(),
    "daily_log_failed": frozenset(),
    "daily_log_saved": frozenset({"time"}),
    "fact_empty": frozenset(),
    "fact_save_failed": frozenset(),
    "fact_saved": frozenset(),
    "media_no_speakers": frozenset(),
    "media_invalid_volume": frozenset(),
    "media_no_device": frozenset(),
    "media_volume_set": frozenset({"target", "value"}),
    "media_louder": frozenset({"target"}),
    "media_quieter": frozenset({"target"}),
    "media_source": frozenset({"target", "source"}),
    "media_paused": frozenset({"target"}),
    "media_resumed": frozenset({"target"}),
    "media_skipped": frozenset({"target"}),
    "media_previous": frozenset({"target"}),
    "media_muted": frozenset({"target"}),
    "media_unmuted": frozenset({"target"}),
    "spotify_paused": frozenset(),
    "spotify_resumed": frozenset(),
    "spotify_skipped": frozenset(),
    "spotify_previous": frozenset(),
    "spotify_no_device": frozenset(),
    "media_no_pause": frozenset(),
    "media_no_resume": frozenset(),
    "media_no_skip": frozenset(),
    "media_no_previous": frozenset(),
    "media_no_louder": frozenset(),
    "media_no_quieter": frozenset(),
    "media_no_mute": frozenset(),
    "calendar_no_calendar": frozenset(),
    "calendar_unreachable": frozenset(),
    "calendar_bad_date": frozenset(),
    "calendar_bad_time": frozenset(),
    "calendar_bad_end_time": frozenset(),
    "todo_added": frozenset({"item"}),
    "todo_unreachable": frozenset(),
    "todo_empty": frozenset(),
    "todo_one": frozenset({"item"}),
    "todo_multiple": frozenset({"count", "items"}),
    "todo_no_pending_match": frozenset({"item"}),
    "todo_not_found": frozenset({"item"}),
    "todo_completed": frozenset({"item"}),
    "device_locked": frozenset({"target"}),
    "device_unlocked": frozenset({"target"}),
    "device_opened": frozenset({"target"}),
    "device_closed": frozenset({"target"}),
    "device_stopped": frozenset({"target"}),
    "device_position": frozenset({"target", "value"}),
    "device_fan_speed": frozenset({"target", "value"}),
    "device_scene": frozenset({"target"}),
    "device_script": frozenset({"target"}),
    "fallback_no_ai": frozenset(),
    "fallback_ai_unreachable": frozenset(),
    "fallback_tool_unavailable": frozenset(),
    "research_queued": frozenset(),
    "research_started": frozenset(),
    "satellite_announcement_queued": frozenset({"target"}),
    "timer_extended": frozenset({"timer_id", "duration"}),
    "timer_not_found": frozenset({"timer_id"}),
    "weather_forecast": frozenset({"target"}),
    "weather_current": frozenset({"condition"}),
    "weather_current_temperature": frozenset({"condition", "temperature", "unit"}),
    "weather_day_condition": frozenset({"day", "condition"}),
    "weather_range": frozenset({"low", "high", "unit"}),
    "weather_high": frozenset({"high", "unit"}),
    "weather_rain_chance": frozenset({"value"}),
    "weather_today": frozenset(),
    "weather_tomorrow": frozenset(),
    "weather_celsius": frozenset(),
    "weather_fahrenheit": frozenset(),
    "weather_unknown": frozenset(),
    "weather_clear": frozenset(),
    "weather_cloudy": frozenset(),
    "weather_severe": frozenset(),
    "weather_fog": frozenset(),
    "weather_hail": frozenset(),
    "weather_thunder": frozenset(),
    "weather_partly_cloudy": frozenset(),
    "weather_heavy_rain": frozenset(),
    "weather_rain": frozenset(),
    "weather_snow": frozenset(),
    "weather_sleet": frozenset(),
    "weather_sunny": frozenset(),
    "weather_windy": frozenset(),
    "temperature_missing": frozenset({"target"}),
    "temperature_unreadable": frozenset({"target"}),
    "temperature_current": frozenset({"target", "value", "unit"}),
    "temperature_with_target": frozenset({"target", "value", "setpoint", "unit"}),
    "finance_percent": frozenset({"value"}),
    "finance_no_price": frozenset(),
    "finance_quote": frozenset({"name", "price"}),
    "finance_quote_exchange": frozenset({"name", "price", "exchange"}),
    "finance_up": frozenset({"amount"}),
    "finance_down": frozenset({"amount"}),
    "finance_movement_percent": frozenset({"detail", "percent"}),
    "finance_movement": frozenset({"detail"}),
    "finance_moved": frozenset({"percent"}),
    "finance_conversion": frozenset({"amount", "base", "conversion", "quote"}),
    "finance_timestamps": frozenset({"quoted", "retrieved", "cache"}),
    "finance_no_timestamp": frozenset(),
    "finance_cache_bypass": frozenset(),
    "finance_cache_allowed": frozenset(),
    "weather_history_missing": frozenset({"target", "date"}),
    "weather_history_unusable": frozenset({"date"}),
    "weather_history_title": frozenset({"target"}),
    "weather_yesterday": frozenset(),
    "finance_not_configured": frozenset(),
    "finance_bad_symbol": frozenset(),
    "finance_bad_window": frozenset(),
    "finance_not_found": frozenset(),
    "finance_unverified_symbol": frozenset({"symbol"}),
    "finance_wrong_symbol": frozenset({"symbol"}),
    "finance_bad_currency": frozenset(),
    "finance_same_currency": frozenset(),
    "finance_bad_amount": frozenset(),
    "finance_missing_rate": frozenset({"base", "quote"}),
    "finance_empty_watchlist": frozenset(),
    "finance_watchlist_failed": frozenset(),
    "finance_market_failed": frozenset(),
    "obsidian_insert_empty": frozenset(),
    "obsidian_disabled": frozenset(),
    "obsidian_insert_disconnected": frozenset(),
    "obsidian_insert_sent": frozenset(),
    "obsidian_rename_empty": frozenset(),
    "obsidian_rename_disconnected": frozenset(),
    "obsidian_rename_sent": frozenset({"title"}),
    "obsidian_delete_disconnected": frozenset(),
    "obsidian_delete_sent": frozenset(),
    "obsidian_replace_empty": frozenset(),
    "obsidian_replace_disconnected": frozenset(),
    "obsidian_replace_sent": frozenset(),
    "notes_list": frozenset({"count", "titles"}),
    "note_read": frozenset({"title", "text", "continuation"}),
    "note_continues": frozenset(),
    "notes_search_empty": frozenset(),
    "notes_search_not_found": frozenset({"query"}),
    "spotify_playing_music": frozenset(),
    "spotify_playing_playlist": frozenset({"name"}),
    "spotify_playing_artist": frozenset({"name"}),
    "spotify_playing_track": frozenset({"name", "artist"}),
    "spotify_missing_query": frozenset(),
    "spotify_play_failed": frozenset(),
    "calc_scientific": frozenset({"mantissa", "exponent"}),
    "calc_undefined": frozenset(),
    "calc_infinity": frozenset(),
    "calc_negative_infinity": frozenset(),
    "calc_million": frozenset({"value"}),
    "calc_billion": frozenset({"value"}),
    "calc_trillion": frozenset({"value"}),
    "calc_date": frozenset({"date"}),
    "calc_date_range": frozenset({"start", "end"}),
    "calc_same_day": frozenset(),
    "calc_days_between": frozenset({"count", "unit"}),
    "calc_date_of": frozenset({"date"}),
    "calc_conversion": frozenset({"source", "result"}),
    "daily_when_today": frozenset(),
    "daily_when_date": frozenset({"date"}),
    "daily_not_found": frozenset({"when"}),
    "daily_read_failed": frozenset({"when"}),
    "daily_read": frozenset({"when", "text", "continuation"}),
    "ack_generic": frozenset(),
    "ack_music": frozenset(),
    "ack_web": frozenset(),
    "device_on": frozenset({"target"}),
    "device_on_brightness": frozenset({"target", "value"}),
    "device_off": frozenset({"target"}),
    "device_toggle": frozenset({"target"}),
    "device_room_on": frozenset({"area"}),
    "device_room_on_brightness": frozenset({"area", "value"}),
    "device_room_off": frozenset({"area"}),
    "device_room_toggle": frozenset({"area"}),
    "device_room_brightness": frozenset({"area", "value"}),
    "device_brightness": frozenset({"target", "value"}),
    "device_color": frozenset({"target", "color"}),
    "device_climate": frozenset({"target", "value"}),
    "device_invalid_brightness": frozenset(),
    "color_red": frozenset(), "color_green": frozenset(), "color_blue": frozenset(),
    "color_yellow": frozenset(), "color_orange": frozenset(), "color_purple": frozenset(),
    "color_pink": frozenset(), "color_white": frozenset(), "color_warm_white": frozenset(),
    "color_cool_white": frozenset(), "color_cyan": frozenset(), "color_magenta": frozenset(),
    "calendar_no_events": frozenset(),
    "calendar_untitled": frozenset(),
    "calendar_event_all_day": frozenset({"day", "summary"}),
    "calendar_event_timed": frozenset({"time", "day", "summary"}),
    "calendar_created": frozenset({"summary", "time", "date", "recurrence"}),
    "calendar_created_time": frozenset({"time"}),
    "calendar_recurrence_daily": frozenset(),
    "calendar_recurrence_weekly": frozenset(),
    "calendar_recurrence_monthly": frozenset(),
    "travel_carrier_unavailable": frozenset(),
    "travel_duration": frozenset({"duration"}),
    "travel_duration_unavailable": frozenset(),
    "travel_price_unavailable": frozenset(),
    "travel_time_unavailable": frozenset(),
    "travel_schedule": frozenset({"airport", "time", "offset"}),
    "travel_flight_summary": frozenset({"carrier", "departure", "arrival", "stops", "duration", "price"}),
    "travel_round_trip_summary": frozenset({"carrier", "out_departure", "out_arrival", "back_departure", "back_arrival", "price"}),
    "travel_round_trip_title": frozenset(),
    "travel_flights_title": frozenset(),
    "travel_resolved_route": frozenset({"date", "origin", "origin_code", "destination", "destination_code", "results"}),
    "travel_hotels_title": frozenset(),
    "travel_hotel_summary": frozenset({"index", "name", "total", "rating", "address", "cancellation"}),
    "travel_no_rating": frozenset(),
    "travel_no_address": frozenset(),
    "travel_free_cancellation": frozenset(),
    "travel_cancellation_unknown": frozenset(),
    "travel_unnamed_property": frozenset(),
    "travel_origin_destination_dates": frozenset(),
    "travel_origin_destination": frozenset(),
    "travel_departure_date": frozenset(),
    "travel_endpoints_required": frozenset(),
    "travel_invalid_endpoints": frozenset(),
    "travel_departure_invalid": frozenset(),
    "travel_return_invalid": frozenset(),
    "travel_adult_required": frozenset(),
    "travel_cabin_stops_invalid": frozenset(),
    "travel_flights_empty": frozenset(),
    "travel_hotel_location_required": frozenset(),
    "travel_hotel_dates_invalid": frozenset(),
    "travel_hotels_empty": frozenset(),
    "travel_past_dates": frozenset({"dates"}),
    "current_date_and_time": frozenset({"date", "time"}),
    "travel_not_configured": frozenset(),
    "travel_auth_failed": frozenset(),
    "travel_provider_rejected": frozenset({"detail"}),
    "travel_unavailable": frozenset(),
    "travel_invalid_response": frozenset(),
    "travel_failed": frozenset(),
    "entity_state": frozenset({"target", "state"}),
    "entity_brightness": frozenset({"value"}),
    "entity_temperature": frozenset({"value"}),
    "entity_current_temperature": frozenset({"value"}),
    "entity_hvac_action": frozenset({"value"}),
    "entity_humidity": frozenset({"value"}),
    "entity_position": frozenset({"value"}),
    "entity_battery": frozenset({"value"}),
    "state_on": frozenset(), "state_off": frozenset(), "state_open": frozenset(),
    "state_closed": frozenset(), "state_unlocked": frozenset(), "state_locked": frozenset(),
    "state_unknown": frozenset(), "state_unavailable": frozenset(),
    "finance_auth_failed": frozenset(),
    "finance_unavailable": frozenset(),
    "finance_invalid_response": frozenset(),
    "finance_provider_rejected": frozenset(),
    "report_failed_partial": frozenset(),
    "report_failed_search_model": frozenset(),
    "report_failed_search": frozenset(),
    "report_failed_model": frozenset(),
    "report_failed": frozenset(),
    "timer_status_id": frozenset({"timer_id", "duration"}),
    "timer_status_list": frozenset({"timers"}),
    "timer_cancelled_id": frozenset({"timer_id"}),
    "area_not_found": frozenset(),
    "area_lookup_failed": frozenset(),
    "area_lookup_invalid": frozenset(),
    "area_no_lights": frozenset(),
    "unsupported_domain": frozenset(),
    "location_not_found": frozenset(),
    "location_no_devices": frozenset(),
    "invalid_temperature": frozenset(),
    "unsupported_operation": frozenset(),
    "overview_unreadable": frozenset(),
    "overview_settled": frozenset(),
    "overview_summary": frozenset({"groups"}),
    "overview_lights": frozenset({"count"}),
    "overview_openings": frozenset({"count"}),
    "overview_locks": frozenset({"count"}),
    "overview_active": frozenset({"count"}),
    "area_no_entities": frozenset({"scope", "area"}),
    "area_has_entities": frozenset({"area", "names", "more"}),
    "area_among_others": frozenset(),
    "area_bad_filter": frozenset(),
    "area_none_in_state": frozenset({"scope", "state", "area"}),
    "area_state_unreadable": frozenset({"area"}),
    "domain_light": frozenset(), "domain_switch": frozenset(), "domain_sensor": frozenset(),
    "domain_climate": frozenset(), "domain_fan": frozenset(), "domain_cover": frozenset(),
    "domain_lock": frozenset(), "domain_media_player": frozenset(), "domain_binary_sensor": frozenset(),
    "itinerary_windows_count": frozenset(),
    "itinerary_window_object": frozenset({"index"}),
    "itinerary_window_order": frozenset({"index"}),
    "itinerary_leg_object": frozenset({"index"}),
    "itinerary_leg_arrival_before_departure": frozenset({"index"}),
    "itinerary_leg_departure_before_arrival": frozenset({"index", "previous"}),
    "itinerary_early_departure": frozenset({"index", "place", "actual", "bound"}),
    "itinerary_late_arrival": frozenset({"index", "place", "actual", "bound"}),
    "itinerary_not_feasible": frozenset({"failures"}),
    "itinerary_feasible": frozenset(),
    "itinerary_chronology_feasible": frozenset(),
    "itinerary_no_combination": frozenset(),
    "itinerary_invalid_json": frozenset(),
    "itinerary_not_arrays": frozenset(),
    "itinerary_timestamp_invalid": frozenset({"field"}),
    "itinerary_offset_missing": frozenset({"field"}),
    "itinerary_field_window_start": frozenset({"index"}),
    "itinerary_field_window_end": frozenset({"index"}),
    "itinerary_field_departure": frozenset({"index"}),
    "itinerary_field_arrival": frozenset({"index"}),
    "itinerary_preliminary": frozenset({"assessment"}),
    "energy_no_sensors": frozenset(),
    "energy_reading": frozenset({"label", "value", "unit"}),
    "energy_overview": frozenset({"readings"}),
    "security_settled": frozenset(),
    "security_cameras_online": frozenset({"count"}),
    "security_attention": frozenset({"findings"}),
    "security_open_entries": frozenset({"count"}),
    "security_unlocked_locks": frozenset({"count"}),
    "security_cameras_unavailable": frozenset({"count"}),
    "satellite_unavailable": frozenset(),
    "satellite_target_not_found": frozenset({"target"}),
    "satellite_target_ambiguous": frozenset({"target", "names"}),
    "satellite_message_empty": frozenset(),
    "satellite_announcement_busy": frozenset(),
    "agent_research_correction": frozenset(),
    "agent_model_failed": frozenset(),
    "agent_plan_failed": frozenset(),
    "agent_steps_incomplete": frozenset(),
    "agent_compose_failed": frozenset(),
    "agent_translation_unavailable": frozenset(),
    "agent_live_lookup_unavailable": frozenset(),
    "agent_no_progress": frozenset(),
    "agent_failed": frozenset(),
    "agent_research_unavailable": frozenset(),
    "agent_not_completed": frozenset({"tasks"}),
    "calendar_today": frozenset(), "calendar_tomorrow": frozenset(),
    "calendar_yesterday": frozenset(), "calendar_this_week": frozenset(),
    "calendar_this_weekday": frozenset({"day"}), "calendar_next_weekday": frozenset({"day"}),
    "calendar_last_weekday": frozenset({"day"}), "calendar_search_range": frozenset({"days"}),
    "calendar_no_match": frozenset({"event", "range"}),
    "calendar_match_all_day": frozenset({"summary", "date"}),
    "calendar_match_timed": frozenset({"summary", "time"}),
    "calendar_matches": frozenset({"events"}),
    "device_service_sent": frozenset({"service", "target"}),
    "vacuum_start": frozenset({"target"}), "vacuum_resume": frozenset({"target"}),
    "vacuum_pause": frozenset({"target"}), "vacuum_stop": frozenset({"target"}),
    "vacuum_dock": frozenset({"target"}), "vacuum_locate": frozenset({"target"}),
    "vacuum_invalid": frozenset({"action"}),
    "research_paper_unknown_year": frozenset(),
    "research_paper_citations": frozenset({"count"}),
    "research_paper_entry": frozenset({"title", "year", "citations", "digest", "source", "id"}),
    "research_missing_query": frozenset(),
    "research_bad_count": frozenset({"max"}),
    "research_unknown_source": frozenset({"source"}),
    "research_unavailable": frozenset(),
    "research_not_found": frozenset({"query"}),
    "research_paper_unavailable": frozenset(),
    "research_paper_invalid": frozenset(),
    "research_paper_detail": frozenset({"title", "digest", "doi", "source", "id", "url"}),
    "research_not_supplied": frozenset(),
    "research_wolfram_disabled": frozenset(),
    "research_wolfram_unavailable": frozenset({"error"}),
    "travel_plan_places": frozenset(), "travel_home_unknown": frozenset(),
    "travel_schedule_missing": frozenset(), "travel_schedule_incomplete": frozenset(),
    "travel_offers_missing": frozenset(), "travel_plan_leg": frozenset({"index", "result"}),
    "travel_plan_result": frozenset({"date", "legs"}), "travel_artifact_unavailable": frozenset(),
    "travel_duplicate_search": frozenset(), "travel_outbound_mismatch": frozenset(),
    "travel_outbound_not_selectable": frozenset(), "travel_return_mismatch": frozenset(),
    "travel_schedule_mismatch": frozenset(),
    "history_window_on": frozenset({"window"}), "history_window_last_days": frozenset({"days"}),
    "history_window_range": frozenset({"start", "end"}),
    "history_no_states": frozenset({"target", "window"}), "history_no_conversation": frozenset({"window"}),
    "history_title": frozenset({"target"}), "history_truncated": frozenset({"count"}),
    "history_when": frozenset({"day", "time"}), "history_user": frozenset(),
    "history_parse_failed": frozenset({"start"}), "history_fetch_failed": frozenset({"target"}),
    "history_conversation_failed": frozenset(), "history_weather_failed": frozenset({"target"}),
    "report_memory_stop": frozenset(), "report_partial_saved": frozenset(),
    "report_notes_unverified": frozenset(), "report_partial_open": frozenset(),
    "report_saved": frozenset(), "report_summary_question": frozenset(),
    "travel_param_date": frozenset({"field"}), "travel_param_integer": frozenset({"field"}),
    "travel_param_range": frozenset({"field", "maximum"}),
    "travel_field_return_date": frozenset(), "travel_field_check_in": frozenset(),
    "travel_field_check_out": frozenset(), "travel_field_adults": frozenset(), "travel_field_children": frozenset(),
    "notes_search_result_item": frozenset({"title", "snippet"}),
    "notes_search_results": frozenset({"count", "plural", "query", "items"}),
    "agent_repeat_request": frozenset(), "agent_summary_failed": frozenset(),
    "agent_research_needs_model": frozenset(), "agent_web_needs_model": frozenset(),
    "web_bad_query": frozenset(), "web_bad_profile": frozenset(), "web_bad_category": frozenset(),
    "web_bad_time_range": frozenset(), "web_bad_page": frozenset(),
    "web_today": frozenset({"date"}), "web_retrieved": frozenset(),
    "web_source": frozenset({"url", "scope", "snippet"}), "web_no_results": frozenset(),
    "web_scope_global": frozenset(), "web_scope_regional": frozenset(), "web_scope_local": frozenset(),
    "busy_phrase": frozenset(),
    "ha_entity_not_found": frozenset({"entity"}), "ha_area_not_found": frozenset({"area"}),
    "ha_weather_entity_not_found": frozenset({"entity"}), "ha_entity_unresolved": frozenset({"entity"}),
    "notes_daily_write_denied": frozenset({"title"}), "notes_append_target_missing": frozenset({"title", "existing"}),
    "ha_invalid_service_data": frozenset({"service"}),
    "spotify_unconfigured": frozenset(), "spotify_no_match": frozenset({"query"}),
    "notes_none": frozenset(),
    "history_transcript_intro": frozenset(),
    "thinking_not_ready": frozenset(), "thinking_memory_cooldown": frozenset(),
    "thinking_tasks_ambiguous": frozenset(), "thinking_tasks_multiple": frozenset(),
    "thinking_topic_required": frozenset(), "thinking_queue_failed": frozenset(),
    "research_match_ambiguous": frozenset(), "research_match_missing": frozenset(),
    "research_multiple_tasks": frozenset(), "research_no_task": frozenset(),
    "research_failure_detail": frozenset({"stage", "reason"}), "research_partial_link": frozenset({"link"}),
    "research_waiting_input": frozenset(), "research_failed_stage": frozenset({"stage"}),
    "research_stage_status": frozenset({"status", "stage"}), "research_last_stage_status": frozenset({"status", "stage"}),
    "research_task_reason": frozenset({"task", "reason"}),
    "research_task_status": frozenset({"task", "status", "stage"}),
    "research_task_failed": frozenset({"task", "reason"}),
    "research_task_ready": frozenset({"task"}), "research_task_last_status": frozenset({"task", "status"}),
    "research_status_queued": frozenset(), "research_status_running": frozenset(),
    "research_status_paused": frozenset(), "research_status_needs_input": frozenset(),
    "research_status_failed": frozenset(), "research_status_ready": frozenset(), "research_status_cancelled": frozenset(),
    "research_attempt_stopped": frozenset({"stage"}),
    "report_completed_fallback": frozenset(), "report_unavailable": frozenset(),
    "report_full_prefix": frozenset(), "report_short_prefix": frozenset(),
    "report_full_offer": frozenset(), "report_no_answer": frozenset(),
    "report_heading_summary": frozenset(), "report_heading_conclusions": frozenset(),
    "report_heading_references": frozenset(), "report_heading_chapters": frozenset(),
    "report_heading_research_notes": frozenset(), "report_heading_under_answered": frozenset(),
    "report_metadata": frozenset({"type", "timestamp", "objective"}),
    "report_type_report": frozenset(), "report_type_notes": frozenset(),
    "report_flight_offers_heading": frozenset(), "report_flight_offers_intro": frozenset(),
    "report_flight_route": frozenset({"origin", "destination", "date"}),
    "report_flight_return": frozenset({"date"}), "report_flight_nonstop": frozenset(),
    "report_flight_stops": frozenset({"count", "plural"}), "report_flight_stops_unknown": frozenset(),
    "report_flight_departure_unknown": frozenset(), "report_flight_arrival_unknown": frozenset(),
    "report_flight_oneway": frozenset({"index", "airlines", "departure", "arrival", "stops", "duration", "price"}),
    "report_flight_roundtrip": frozenset({"index", "airlines", "out_departure", "out_arrival", "back_departure", "back_arrival", "price"}),
    "report_flight_retrieved": frozenset({"timestamp"}), "report_flight_source": frozenset({"route", "query", "retrieved"}),
    "report_flight_live_heading": frozenset(),
    "report_additional_facts": frozenset(), "report_no_summary": frozenset(), "report_no_findings": frozenset(),
    "report_not_produced": frozenset(),
    "reminder_prefix": frozenset(),
    "conversation_listening": frozenset(),
    "calendar_reminder_unconfigured": frozenset(), "todo_unconfigured_add": frozenset(),
    "todo_unconfigured_read": frozenset(), "todo_unconfigured": frozenset(),
    "calc_no_expression": frozenset(), "calc_bad_expression": frozenset({"expression"}),
    "calc_bad_dates": frozenset(), "calc_bad_weekday": frozenset(),
    "calc_bad_amount": frozenset(), "calc_missing_unit": frozenset(),
    "calc_incompatible_units": frozenset({"source", "target"}),
    "calc_unknown_units": frozenset({"source", "target"}),
    "agent_confirmation_failed": frozenset(),
    "agent_result_unconfirmed": frozenset({"action"}),
    "agent_web_unconfirmed": frozenset(),
    "agent_reactive_fallback": frozenset(),
    "ha_target_override": frozenset(),
    "ha_target_unverified": frozenset({"target"}),
    "ha_target_rejected": frozenset({"target"}),
    "ha_service_failed": frozenset({"target"}),
    "calc_million_one": frozenset({"value"}), "calc_billion_one": frozenset({"value"}),
    "calc_trillion_one": frozenset({"value"}),
    "assistant_starting": frozenset(),
    "stage_queued": frozenset(), "stage_thinking": frozenset(), "stage_input": frozenset(),
    "stage_paused": frozenset(), "stage_report_ready": frozenset(), "stage_notes_ready": frozenset(),
    "stage_resuming": frozenset(), "stage_planning": frozenset(), "stage_synthesising": frozenset(),
    "stage_wait_backend": frozenset(), "stage_recover_section": frozenset(), "stage_wait_quiet": frozenset(),
    "stage_researching": frozenset({"title"}), "stage_analysing": frozenset({"title"}),
    "stage_writing_section": frozenset({"index"}), "stage_checking_section": frozenset({"index"}),
    "stage_saving_section": frozenset({"title"}), "stage_reviewing": frozenset({"current", "total"}),
    "color_rgb_invalid": frozenset(),
    "color_unreadable": frozenset({"color"}),
    "color_unknown": frozenset({"color"}),
}
_FORMATTER = Formatter()
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")


def _valid_template(text: object, fields: frozenset[str]) -> bool:
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        return False
    # Reject Jinja and escaped braces rather than treating them as executable
    # syntax or permitting confusing user edits.
    if "{{" in text or "}}" in text or "{%" in text or "{#" in text:
        return False
    try:
        for _, field, spec, conversion in _FORMATTER.parse(text):
            if field is not None and (field not in fields or spec or conversion):
                return False
    except ValueError:
        return False
    return True


def load_responses(intents_dir: str | Path, language: str) -> dict:
    """Load and validate one file; never execute templates or read other packs."""
    if not isinstance(language, str) or not _LANGUAGE.fullmatch(language):
        return {}
    path = Path(intents_dir) / language / "fulloch-responses.yaml"
    try:
        if path.stat().st_size > 65536:
            raise ValueError("response file exceeds 64 KiB")
        yaml = YAML(typ="safe")
        with path.open(encoding="utf-8") as stream:
            data = yaml.load(stream)
        if not isinstance(data, dict) or data.get("language") != language or data.get("schema_version") != 1:
            raise ValueError("invalid response file header")
        responses = data.get("responses")
        if not isinstance(responses, dict):
            raise ValueError("responses must be a mapping")
        validated = {}
        for key, outcomes in responses.items():
            if key not in RESPONSE_FIELDS or not isinstance(outcomes, dict) or not outcomes:
                raise ValueError("invalid response key")
            validated[key] = {}
            for outcome, choices in outcomes.items():
                if outcome not in {"success", "error"} or not isinstance(choices, list) or not choices or len(choices) > 20:
                    raise ValueError("invalid outcome choices")
                if not all(_valid_template(choice, RESPONSE_FIELDS[key]) for choice in choices):
                    raise ValueError("invalid response template")
                validated[key][outcome] = tuple(choices)
        return validated
    except FileNotFoundError:
        return {}
    except Exception as exc:  # A user's malformed file must not affect actions.
        logger.warning("Ignoring invalid response file %s: %s", path, exc)
        return {}


def render_response(responses: dict, key: str, outcome: str, values: dict) -> str | None:
    """Render the first validated choice, or return None for a safe fallback."""
    choices = responses.get(key, {}).get(outcome, ())
    if not choices or key not in RESPONSE_FIELDS or outcome not in {"success", "error"}:
        return None
    template = choices[0]
    if not _valid_template(template, RESPONSE_FIELDS[key]):
        return None
    try:
        return template.format_map(values)
    except (KeyError, ValueError, TypeError):
        return None


def format_local_datetime(value: datetime, language: str) -> dict[str, str]:
    """Format an already-zoned instant without using process-local timezone."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("a timezone-aware datetime is required")
    return {
        "date": format_date(value.date(), format="full", locale=language.replace("-", "_")),
        "time": format_time(value, format="short", tzinfo=value.tzinfo, locale=language.replace("-", "_")),
    }


def format_local_duration(seconds: int, language: str) -> str:
    """Format every nonzero duration component using CLDR plurals and lists."""
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
        raise ValueError("duration must be a non-negative integer number of seconds")
    parts = []
    for unit, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        value, seconds = divmod(seconds, size)
        if value:
            parts.append(format_unit(value, unit, length="long", locale=language.replace("-", "_")))
    locale = language.replace("-", "_")
    return format_list(parts, locale=locale) if parts else format_unit(0, "second", length="long", locale=locale)


def _render_with_bundled_fallback(responses, language, key, outcome, values):
    """Use new bundled keys without overwriting existing customized resources."""
    result = render_response(responses, key, outcome, values)
    if result or not responses or key in responses:
        return result
    seed = Path(os.environ.get("FULLOCH_SEED_DIR", "/app/seed")) / "intents"
    if not seed.is_dir():
        seed = Path(__file__).resolve().parent.parent / "data" / "intents"
    return render_response(load_responses(seed, language), key, outcome, values)


def _verified_numeric_service(metadata, outcome):
    """Return response key/value only for an exact, successful HA service."""
    specs = {
        "HassLightSet": ("brightness", "light", "turn_on", "brightness", "light_brightness"),
        "HassClimateSetTemperature": ("temperature", "climate", "set_temperature", "temperature", "climate_temperature"),
        "HassSetPosition": ("position", None, "set_cover_position", "position", "cover_position"),
        "HassSetVolume": ("volume_level", "media_player", "volume_set", "volume_level", "media_volume"),
    }
    spec = specs.get(metadata.intent)
    if not spec or outcome.code != "service_called":
        return None
    slot, domain, service, field, key = spec
    value = metadata.slots.get(slot)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    actual = outcome.values
    if actual.get("domain") != domain and not (domain is None and actual.get("domain") in {"cover", "valve"}):
        return None
    if actual.get("service") != ("set_valve_position" if metadata.intent == "HassSetPosition" and actual.get("domain") == "valve" else service):
        return None
    targets = actual.get("entity_id")
    targets = [targets] if isinstance(targets, str) else targets
    if not isinstance(targets, list) or not targets or any(not isinstance(t, str) or not t.startswith(actual["domain"] + ".") for t in targets):
        return None
    named = metadata.slots.get("name")
    if named and targets != [named]:
        return None
    data = actual.get("data")
    if not isinstance(data, dict) or field not in data:
        return None
    sent = data[field]
    expected = int(value * 255 / 100) if slot == "brightness" else value / 100 if slot == "volume_level" else value
    if isinstance(sent, bool) or not isinstance(sent, (int, float)) or not math.isclose(sent, expected, abs_tol=1e-9):
        return None
    return key, value


def select_fastpath_response(
    metadata: MatchMetadata | None,
    outcome: OperationOutcome | None,
    observation: str,
    intents_dir: str | Path,
    *,
    language_confirmed: bool = False,
) -> str:
    """Select a verified local success or retain the original observation.

    Callers must first establish an unambiguous response language. This
    supports verified time/date, timers, light, climate, position and volume
    outcomes; unknown actions and failures retain their original observations.
    No choice is inferred from an action name, a nonempty string, or a normal
    agent step. Resources are re-read on each call, so edits are visible on the
    next eligible turn without replacing a customized file.
    """
    if not language_confirmed or not isinstance(metadata, MatchMetadata) or not isinstance(outcome, OperationOutcome):
        return observation
    if not isinstance(metadata.language, str) or not _LANGUAGE.fullmatch(metadata.language):
        return observation
    if not isinstance(outcome.values, dict):
        return observation
    key = None
    values = {}
    try:
        if outcome.success is False:
            errors = {"HassStartTimer": {"timer_limit": "timer_limit", "duration_too_long": "timer_too_long", "invalid_duration": "timer_invalid_duration"},
                      "HassCancelTimer": {"no_active_timers": "timer_none", "ambiguous_timer": "timer_ambiguous"}}
            key = errors.get(metadata.intent, {}).get(outcome.code)
            if metadata.intent in {"HassTurnOn", "HassTurnOff", "HassLightSet", "HassClimateSetTemperature", "HassSetPosition", "HassSetVolume"} and outcome.code in {"ha_unreachable", "ha_timeout", "ha_not_configured", "voice_denied"}:
                key = outcome.code
            if not key:
                return observation
            responses = load_responses(intents_dir, metadata.language)
            return _render_with_bundled_fallback(responses, metadata.language, key, "error", {}) or observation
        if outcome.success is not True:
            return observation
        if metadata.intent in {"HassGetCurrentTime", "HassGetCurrentDate"} and outcome.code == "current_time":
            key = "time" if metadata.intent == "HassGetCurrentTime" else "date"
            values = format_local_datetime(outcome.values.get("datetime"), metadata.language)
        elif metadata.intent == "HassStartTimer" and outcome.code == "timer_started":
            seconds = outcome.values.get("duration_seconds")
            if not isinstance(outcome.values.get("timer_id"), str) or not outcome.values["timer_id"]:
                return observation
            if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
                return observation
            key = "timer_started"
            values = {"duration": format_local_duration(seconds, metadata.language)}
        elif metadata.intent == "HassCancelTimer" and outcome.code == "timer_cancelled":
            if not isinstance(outcome.values.get("timer_id"), str) or not outcome.values["timer_id"]:
                return observation
            key = "timer_cancelled"
        elif metadata.intent == "HassTimerStatus" and outcome.code == "no_active_timers" and outcome.values.get("timers") == []:
            key = "timer_none"
        elif metadata.intent == "HassTimerStatus" and outcome.code == "timer_status":
            timers = outcome.values.get("timers")
            if not isinstance(timers, list) or not timers or len(timers) > 32:
                return observation
            durations = []
            for timer in timers:
                if not isinstance(timer, dict) or not isinstance(timer.get("timer_id"), str):
                    return observation
                seconds = timer.get("remaining_seconds")
                if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    return observation
                durations.append(format_local_duration(int(seconds), metadata.language))
            key = "timer_status" if len(durations) == 1 else "timer_status_multiple"
            values = {"duration": durations[0], "durations": format_list(durations, locale=metadata.language.replace("-", "_"))}
        elif metadata.intent in {"HassTurnOn", "HassTurnOff"} and outcome.code == "service_called":
            service = "turn_on" if metadata.intent == "HassTurnOn" else "turn_off"
            targets = outcome.values.get("entity_id")
            targets = [targets] if isinstance(targets, str) else targets
            if (outcome.values.get("domain") != "light" or outcome.values.get("service") != service
                    or not isinstance(targets, list) or not targets
                    or any(not isinstance(target, str) or not target.startswith("light.") for target in targets)):
                return observation
            named_target = metadata.slots.get("name")
            if named_target and (not isinstance(named_target, str) or targets != [named_target]):
                return observation
            if not named_target and not (metadata.slots.get("area") or metadata.slots.get("floor")):
                return observation
            key = "light_on" if service == "turn_on" else "light_off"
        else:
            numeric = _verified_numeric_service(metadata, outcome)
            if not numeric:
                return observation
            key, number = numeric
            values = {"value": format_decimal(number, locale=metadata.language.replace("-", "_"))}
        responses = load_responses(intents_dir, metadata.language)
        return _render_with_bundled_fallback(responses, metadata.language, key, "success", values) or observation
    except (ValueError, TypeError, UnknownLocaleError, OverflowError):
        return observation
