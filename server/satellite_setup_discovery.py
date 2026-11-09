"""Best-effort discovery and browser timezone conversion for USB setup."""

import calendar
import re
import socket
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import TZPATH, ZoneInfo, ZoneInfoNotFoundError

UNSUPPORTED_TIMEZONE = (
    "The browser timezone cannot be represented reliably on the satellite. "
    "Use the USB CLI fallback with a suitable POSIX timezone; a fixed offset "
    "will not follow daylight-saving changes."
)

# Deliberately accept only the POSIX subset whose interpretation is shared by
# ESP-IDF newlib and this validator. In particular, newlib cannot reliably
# parse negative transition times. Reject other forms rather than guessing.
_NAME = r"(?:[A-Za-z]{3,10}|<[-+0-9A-Za-z]{3,10}>)"
_OFFSET = r"[+-]?\d{1,2}(?::\d{1,2}(?::\d{1,2})?)?"
_RULE = r"M\d{1,2}\.\d\.\d(?:/\d{1,2}(?::\d{1,2}(?::\d{1,2})?)?)?"
_POSIX = re.compile(
    rf"^(?P<std>{_NAME})(?P<offset>{_OFFSET})"
    rf"(?:(?P<dst>{_NAME})(?P<dstoffset>{_OFFSET})?,(?P<start>{_RULE}),(?P<end>{_RULE}))?$"
)


def _seconds(value, maximum):
    sign = -1 if value.startswith("-") else 1
    parts = [int(part) for part in value.lstrip("+-").split(":")]
    if parts[0] > maximum or any(part > 59 for part in parts[1:]):
        raise ValueError(UNSUPPORTED_TIMEZONE)
    return sign * (parts[0] * 3600 + (parts[1] if len(parts) > 1 else 0) * 60 + (parts[2] if len(parts) > 2 else 0))


def _transition(year, rule, previous_offset):
    date, _, clock = rule.partition("/")
    month, week, weekday = (int(part) for part in date[1:].split("."))
    if not (1 <= month <= 12 and 1 <= week <= 5 and 0 <= weekday <= 6):
        raise ValueError(UNSUPPORTED_TIMEZONE)
    first_weekday, days = calendar.monthrange(year, month)
    day = 1 + (weekday - (first_weekday + 1) % 7) % 7 + (week - 1) * 7
    if day > days:
        day -= 7
    # Restrict transition times to ordinary same-day times, even though some
    # newlib versions accept larger numbers, to avoid parser discrepancies.
    seconds = _seconds(clock or "2", 23)
    local = datetime(year, month, day, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return local - timedelta(seconds=previous_offset)


def _validate_posix_timezone(zone, value, now=None):
    match = _POSIX.fullmatch(value)
    if not match:
        raise ValueError(UNSUPPORTED_TIMEZONE)
    standard = -_seconds(match["offset"], 24)
    daylight = -_seconds(match["dstoffset"], 24) if match["dstoffset"] else standard + 3600
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=366 * 2)
    end = now + timedelta(days=366 * 5)
    transitions = []
    if match["dst"]:
        for year in range(start.year - 1, end.year + 2):
            transitions.extend((
                (_transition(year, match["start"], standard), daylight),
                (_transition(year, match["end"], daylight), standard),
            ))
        transitions.sort()

    def expected(instant):
        offset = standard
        for when, new_offset in transitions:
            if when > instant:
                break
            offset = new_offset
        return offset

    def check(instant):
        actual = instant.astimezone(zone).utcoffset()
        if actual is None or actual.total_seconds() != expected(instant):
            raise ValueError(UNSUPPORTED_TIMEZONE)

    # Check recent history as well as the future: some TZif files end their
    # explicit transitions early and use a constant footer thereafter.
    # This cannot predict subsequent changes to timezone legislation.
    instant = start
    while instant <= end:
        check(instant)
        instant += timedelta(hours=6)
    for when, _ in transitions:
        if start <= when <= end:
            check(when - timedelta(seconds=1))
            check(when)


def posix_timezone(name):
    if not isinstance(name, str) or len(name) > 128:
        raise ValueError("Unable to determine browser timezone")
    try:
        zone = ZoneInfo(name)
        for directory in TZPATH:
            path = (Path(directory) / name).resolve()
            if not path.is_relative_to(Path(directory).resolve()) or not path.is_file():
                continue
            tail = path.read_bytes().rsplit(b"\n", 2)
            if len(tail) == 3 and tail[-2]:
                result = tail[-2].decode("ascii")
                if len(result.encode()) <= 64:
                    _validate_posix_timezone(zone, result)
                    return result
    except ValueError as error:
        if str(error) == UNSUPPORTED_TIMEZONE:
            raise
    except (ZoneInfoNotFoundError, OSError, UnicodeError):
        pass
    raise ValueError("Unable to determine browser timezone")


def wifi_networks():
    try:
        result = subprocess.run(
            ["nmcli", "-t", "--escape", "yes", "-f", "SSID", "device", "wifi", "list", "--rescan", "yes"],
            capture_output=True, text=True, timeout=8, check=True,
        )
        # nmcli escapes backslashes and colons in terse mode.
        return sorted({re.sub(r"\\([\\:])", r"\1", line) for line in result.stdout.splitlines() if line})
    except (OSError, subprocess.SubprocessError):
        return []


def sonos_hosts():
    request = ("M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\n"
               'MAN: "ssdp:discover"\r\nMX: 1\r\nST: urn:schemas-upnp-org:device:ZonePlayer:1\r\n\r\n')
    hosts = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(request.encode(), ("239.255.255.250", 1900))
            deadline = time.monotonic() + 2
            while (remaining := deadline - time.monotonic()) > 0:
                sock.settimeout(remaining)
                try:
                    data, address = sock.recvfrom(8192)
                except socket.timeout:
                    break
                if b"urn:schemas-upnp-org:device:zoneplayer:1" in data.lower():
                    hosts.add(address[0])
    except OSError:
        pass
    return sorted(hosts)
