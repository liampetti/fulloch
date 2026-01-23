'''
Regex Intent Catch 

Optional functions for catching simple intent based queries quickly before sending on to LLM
'''

import re

import logging

logger = logging.getLogger(__name__)

# Special regex to just directly send intent without ai check for music control and time checks
def extract_after_play(command):
    pattern = r'play\s+(.+)$'
    match = re.search(pattern, command, re.IGNORECASE)
    if match:
        logger.debug(f"Caught Play Match: {match}")
        return match.group(1).strip()
    return None

def extract_stop(command):
    # pattern to match "stop", "pause", or "halt" commands
    # This will match any of these words at the start of the command
    # and ignore case sensitivity.
    # It will also ignore any leading or trailing whitespace and any punctuation.
    # Example matches: "stop", " pause ", "halt now", "stop."
    pattern = r'^\s*(stop|pause|halt)\b'
    match = re.match(pattern, command, re.IGNORECASE)
    if match:
        logger.debug(f"Caught Stop Match: {match}")
        return True
    return None

def extract_skip(command):
    pattern = r'^\s*skip\b'
    match = re.match(pattern, command, re.IGNORECASE)
    if match:
        logger.debug(f"Caught Skip Match: {match}")
        return True
    return None

def extract_resume(command):
    pattern = r'^\s*resume\b'
    match = re.match(pattern, command, re.IGNORECASE)
    if match:
        logger.debug(f"Caught Resume Match: {match}")
        return True
    return None

def has_time_query(text):
    pattern = r"(what time is it|what'?s the time|what time it is)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        logger.debug(f"Caught Time Match: {match}")
        return True
    return None

def extract_timer(command):
    """Extract timer duration from commands like 'start timer ten minutes'."""
    pattern = r'(?:start|set)\s+(?:a\s+)?(?:timer|time)\s+(?:for\s+)?(.+?)(?:\s+please)?$'

    match = re.search(pattern, command, re.IGNORECASE)
    if match:
        logger.debug(f"Caught Timer Match: {match}")
        return match.group(1).strip()
    return None

def list_timers(command):
    """Get list of current timers"""
    pattern = r'get\s+(?:a\s+)?(?:timers|timer|time)(?:\s+(.*?))?(?:\s+status)?$'

    match = re.search(pattern, command, re.IGNORECASE)
    if match:
        logger.debug(f"Caught List Timers Match: {match}")
        return True
    return None

def catchAll(user_message):
    """Catch all intents from user message."""
    to_play = extract_after_play(user_message)
    if to_play is not None:
        return {"intent": "play_song", "args": [to_play]}
        
    to_stop = extract_stop(user_message)
    if to_stop is not None:
        return {"intent": "pause", "args": []}
        
    time_query = has_time_query(user_message)
    if time_query is not None:
        return {"intent": "get_time", "args": []}
        
    to_skip = extract_skip(user_message)
    if to_skip is not None:
        return {"intent": "skip", "args": []}
        
    to_resume = extract_resume(user_message)
    if to_resume is not None:
        return {"intent": "resume", "args": []}
        
    timer_duration = extract_timer(user_message)
    if timer_duration is not None:
        return {"intent": "start_countdown", "args": [timer_duration]}
        
    timer_list = list_timers(user_message)
    if timer_list is not None:
        return {"intent": "list_timers", "args": []}
    
    return user_message

if __name__ == "__main__":
    # Test cases for different intents
    test_messages = [
        "play some rock music",
        "stop",
        "what time is it",
        "skip",
        "resume",
        "start timer ten minutes",
        "set timer for 2 hours",
        "start a timer thirty seconds please",
        "random message that shouldn't match",
        "get time or status",
        "get timers "
    ]
    
    print("Testing intent detection:")
    print("-" * 40)
    
    for message in test_messages:
        result = catchAll(message)
        print(f"Input: {message!r}")
        print(f"Output: {result}")
        print("-" * 40)