"""Public Home Assistant tools and dashboard/reminder interfaces.

Implementation modules depend only on ha_client, the sole owner of HA
configuration, HTTP access, caches, lazy loading and deny-list state. Spotify
also uses that client; only media's optional Connect fallback imports Spotify
(locally). Importing this entry point registers configured tools without I/O.
"""

# The reminder poller also consumes these two configuration interfaces.
from .ha_calendar import (
    _reminder_calendar_entity as _reminder_calendar_entity,
)
from .ha_calendar import (
    add_todo_item as add_todo_item,
)
from .ha_calendar import (
    complete_todo_item as complete_todo_item,
)
from .ha_calendar import (
    create_calendar_event as create_calendar_event,
)
from .ha_calendar import (
    find_calendar_event as find_calendar_event,
)
from .ha_calendar import (
    get_todo_items as get_todo_items,
)
from .ha_calendar import (
    get_upcoming_events as get_upcoming_events,
)
from .ha_calendar import (
    whats_on as whats_on,
)
from .ha_client import (
    HA_CONFIG as HA_CONFIG,
)
from .ha_client import (
    get_denylist as get_denylist,
)
from .ha_client import (
    list_areas as list_areas,
)
from .ha_client import (
    list_entities as list_entities,
)
from .ha_client import (
    set_entity_denied as set_entity_denied,
)
from .ha_devices import (
    activate_scene as activate_scene,
)
from .ha_devices import (
    call_ha_service as call_ha_service,
)
from .ha_devices import (
    close_cover as close_cover,
)
from .ha_devices import (
    ha_vacuum as ha_vacuum,
)
from .ha_devices import (
    lock as lock,
)
from .ha_devices import (
    open_cover as open_cover,
)
from .ha_devices import (
    run_script as run_script,
)
from .ha_devices import (
    set_climate as set_climate,
)
from .ha_devices import (
    set_color as set_color,
)
from .ha_devices import (
    set_cover_position as set_cover_position,
)
from .ha_devices import (
    set_fan_speed as set_fan_speed,
)
from .ha_devices import (
    set_ha_brightness as set_ha_brightness,
)
from .ha_devices import (
    stop_cover as stop_cover,
)
from .ha_devices import (
    toggle as toggle,
)
from .ha_devices import (
    turn_off as turn_off,
)
from .ha_devices import (
    turn_on as turn_on,
)
from .ha_devices import (
    unlock as unlock,
)
from .ha_media import (
    ha_mute as ha_mute,
)
from .ha_media import (
    pause as pause,
)
from .ha_media import (
    previous as previous,
)
from .ha_media import (
    resume as resume,
)
from .ha_media import (
    select_source as select_source,
)
from .ha_media import (
    skip as skip,
)
from .ha_media import (
    volume_down as volume_down,
)
from .ha_media import (
    volume_set as volume_set,
)
from .ha_media import (
    volume_up as volume_up,
)
from .ha_state import (
    get_conversation_history as get_conversation_history,
)
from .ha_state import (
    get_energy_overview as get_energy_overview,
)
from .ha_state import (
    get_entities_in_area_state as get_entities_in_area_state,
)
from .ha_state import (
    get_entity_history as get_entity_history,
)
from .ha_state import (
    get_entity_state as get_entity_state,
)
from .ha_state import (
    get_home_overview as get_home_overview,
)
from .ha_state import (
    get_security_overview as get_security_overview,
)
from .ha_state import (
    get_temperature as get_temperature,
)
from .ha_state import (
    get_weather_forecast as get_weather_forecast,
)
from .ha_state import (
    list_entities_in_area as list_entities_in_area,
)
from .thinking_playbooks import thinking_playbook

thinking_playbook(
    name="home context",
    triggers=(
        r"\b(home|house|light|lights|thermostat|weather|calendar|todo|energy|security|device|sensor)\b",
    ),
    capabilities=(
        "get_entity_state",
        "get_home_overview",
        "get_energy_overview",
        "get_security_overview",
        "get_weather_forecast",
        "find_calendar_event",
        "whats_on",
        "get_todo_items",
        "get_entity_history",
        "list_entities_in_area",
    ),
    solve_path=(
        "Retrieve the relevant Home Assistant state or records before describing the home.",
        "Use a narrower follow-up lookup only when the first result leaves a material ambiguity.",
        "Report observed state separately from recommendations or inference.",
    ),
    completion_rule="Home-status claims are supported by a retrieved Home Assistant observation.",
)

thinking_playbook(
    name="conversation history",
    triggers=(r"\b(previous|earlier|conversation|chat history|what did (?:i|we) say)\b",),
    capabilities=("get_conversation_history",),
    solve_path=(
        "Retrieve the relevant conversation history before recalling prior wording or decisions.",
        "Quote or summarize only the retrieved entries relevant to the request.",
    ),
    completion_rule="Claims about prior conversation are supported by retrieved history.",
)
