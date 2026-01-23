"""
Tool for connecting to Pioneer eISCP protocol 
Communication standard used to control Pioneer and Onkyo audio/video receivers over a network
"""
import yaml

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

import asyncio
import logging
import re
import sys
from typing import Optional, Dict

from .tool_registry import tool, tool_registry
import logging

logger = logging.getLogger(__name__)

HOST = config['pioneer']['avr_host']
PORT = config['pioneer']['avr_port']


# Command mappings for Pioneer eISCP protocol
COMMANDS = {
    'power_on': 'PO',
    'power_off': 'PF',
    'volume_set': '{:03d}VL',
    'mute_on': 'MO',
    'mute_off': 'MF',
    'input_select': '{}FN',
}

QUERIES = {
    'power': '?P',
    'volume': '?V',
    'mute': '?M',
    'input': '?F',
}

RESPONSE_PATTERNS = {
    'power': r'PWR(\d)',
    'volume': r'VOL(\d{3})',
    'mute': r'MUT(\d)',
    'input': r'FN(\d{2})',
}

DEFAULT_INPUTS = {
    '00': 'PHONO', '01': 'CD', '02': 'TUNER', '03': 'CD-R/TAPE',
    '04': 'MUSIC', '05': 'TV', '10': 'VIDEO 1', '14': 'VIDEO 2',
    '19': 'HDMI 1', '20': 'HDMI 2', '21': 'HDMI 3', '22': 'HDMI 4',
    '23': 'HDMI 5', '24': 'HDMI 6', '25': 'BD',
}

class AVR:
    """Simplified Pioneer AVR IP control using eISCP protocol.
    
    This class provides async methods to control and query Pioneer AV receivers
    over the network using the eISCP protocol
    """
    
    def __init__(self, host: str, port: int = 60128, 
                 input_list: Optional[Dict[str, str]] = None):
        """Initialize AVR controller.
        
        Args:
            host: IP address of the AVR
            port: Port number (default 60128 for eISCP)[1]
            input_list: Custom input mapping (optional)
        """
        self.host = host
        self.port = port
        self._input_list = input_list or DEFAULT_INPUTS
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._state = {'power': False, 'volume': 0, 'mute': False, 'input': '00'}
        
    async def connect(self):
        """Establish TCP connection to AVR."""
        logger.info(f"Connecting to AVR at {self.host}:{self.port}")
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=5.0)
            logger.info("Connection established")
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout to {self.host}:{self.port}")
            raise
        
    async def disconnect(self):
        """Close TCP connection to AVR."""
        if self._writer:
            logger.info("Closing connection")
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None
            
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit."""
        await self.disconnect()
        
    async def _send_raw(self, data: str):
        """Send raw command string to AVR."""
        if not self._writer:
            raise RuntimeError("Not connected to AVR")
            
        logger.debug(f"Sending: {data}")
        self._writer.write((data + "\r").encode('ascii'))
        await self._writer.drain()
        
    async def _read_response(self, timeout: float = 2.0) -> str:
        """Read response from AVR."""
        if not self._reader:
            raise RuntimeError("Not connected to AVR")
            
        try:
            data = await asyncio.wait_for(
                self._reader.readuntil(b'\r\n'), timeout)
            response = data.decode('ascii').strip()
            logger.debug(f"Received: {response}")
            return response
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for response")
            return ""
            
    async def query(self, prop: str) -> Optional[str]:
        """Query a property from AVR."""
        if prop not in QUERIES:
            logger.warning(f"Invalid query property: {prop}")
            return None
            
        await self._send_raw(QUERIES[prop])
        response = await self._read_response()
        
        pattern = RESPONSE_PATTERNS.get(prop)
        if pattern:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        return None
        
    async def update_state(self):
        """Update all state properties from AVR."""
        logger.info("Updating AVR state")
        for prop in QUERIES:
            value = await self.query(prop)
            if value is not None:
                self._parse_state(prop, value)
            else:
                break # Break loop if None returned on any query
                
    def _parse_state(self, prop: str, value: str):
        """Parse and store state value."""
        if prop == 'power':
            self._state['power'] = value == '0'  # PWR0 = on[1]
        elif prop == 'volume':
            self._state['volume'] = int(value)
        elif prop == 'mute':
            self._state['mute'] = value == '1'
        elif prop == 'input':
            self._state['input'] = value
            
    @property
    def power(self) -> bool:
        """Power state (True = on, False = off)."""
        return self._state['power']
        
    @property
    def volume(self) -> int:
        """Volume level in raw value"""
        return self._state['volume']
        
    @property
    def mute(self) -> bool:
        """Mute state (True = muted)."""
        return self._state['mute']
        
    @property
    def input_number(self) -> str:
        """Current input number (two-digit string)."""
        return self._state['input']
        
    @property
    def input_name(self) -> str:
        """Current input name."""
        return self._input_list.get(self._state['input'], 'Unknown')
        
    async def set_power(self, value: bool):
        """Turn power on/off."""
        cmd = 'power_on' if value else 'power_off'
        await self._send_raw(COMMANDS[cmd])
        self._state['power'] = value
        logger.info(f"Power {'ON' if value else 'OFF'}")
        
    async def set_volume(self, db_value: int):
        # Normalize input to negative dB
        clean_db = -abs(db_value)
        # Calculate raw value based on linear formula
        raw_result = 2 * clean_db + 160
        # Apply limits: Max 140, Min 0
        value = int(max(0, min(raw_result, 140)))
        await self._send_raw(COMMANDS['volume_set'].format(value))
        self._state['volume'] = value
        logger.info(f"Volume set to {value} (raw)")

    async def set_volume_raw(self, raw_value: int):
        # Apply limits: Max 140, Min 0
        value = int(max(0, min(raw_value, 140)))
        await self._send_raw(COMMANDS['volume_set'].format(value))
        self._state['volume'] = value
        logger.info(f"Volume set to {value} (raw)")
            
    async def set_mute(self, value: bool):
        """Set mute on/off."""
        cmd = 'mute_on' if value else 'mute_off'
        await self._send_raw(COMMANDS[cmd])
        self._state['mute'] = value
        logger.info(f"Mute {'ON' if value else 'OFF'}")
        
    async def set_input_number(self, number: str):
        """Set input by number."""
        if number in self._input_list:
            await self._send_raw(COMMANDS['input_select'].format(number))
            self._state['input'] = number
            logger.info(f"Input set to {number}: {self._input_list[number]}")
        else:
            logger.warning(f"Input number {number} not in input list")
            
    async def set_input_name(self, name: str):
        """Set input by name."""
        for num, input_name in self._input_list.items():
            if input_name.lower() == name.lower():
                await self.set_input_number(num)
                return
        logger.warning(f"Input name '{name}' not found")

async def setup_avr(input_type: str = "Music"):
    """
    Setup function for AVR control.
    Connects to AVR, checks power state, turns on, sets input and volume.
    """    
    if input_type.upper() == "MUSIC":
        vol = 50
    else:
        vol = 35

    try:
        async with AVR(HOST, PORT) as avr:
            await avr.update_state()
            # Only run setup if receiver is turned off, otherwise ignore
            if not avr.power:
                await avr.set_power(True)
                await asyncio.sleep(10)  # Wait for AVR to boot (slow!!!)
                input_no = next((k for k, v in DEFAULT_INPUTS.items() if v == input_type.upper()), '04')
                await avr.set_input_number(input_no)
                await asyncio.sleep(2) # Need to be gentle to the AVR
                await avr.set_volume(vol)            
    except asyncio.TimeoutError:
        logger.error(f"Connection timeout to {HOST}:{PORT}")
        sys.exit(1)
    except ConnectionRefusedError:
        logger.error(f"Connection refused by {HOST}:{PORT}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(setup_avr("Music"))

# Decibal Ref:
"""
160 = 0dB
140 = -10dB
120 = -20dB
100 = -30dB
90 = -35dB
80 = -40dB
70 = -45dB
60 = -50dB
50 = -55dB
40 = -60dB
"""

#########################
# Tool Calls
#########################

# Turn On
async def _turn_on_sound_system():
    """Turn on the sound system."""
    try:
        async with AVR(HOST, PORT) as avr:
            if not avr.power:
                await avr.set_power(True)
                await asyncio.sleep(5)  # Wait for AVR to boot (slow)
    except Exception as e:
        logger.error(f"Pioneer power on failed with error: {e}")
        sys.exit(1)


@tool(
    name="turn_on_sound_system",
    description="Turn on the Pioneer AVR",
    aliases=["sound_on", "sound_system_on", "turn_on_sound_system"]
)
def turn_on_sound_system():
    """Turn on the sound system"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_turn_on_sound_system())
    else:
        return loop.create_task(_turn_on_sound_system())
    

# Turn Off
async def _turn_off_sound_system():
    """Turn off the sound system."""
    try:
        async with AVR(HOST, PORT) as avr:
            await avr.set_power(False)  
    except Exception as e:
        logger.error(f"Pioneer power off failed with error: {e}")
        sys.exit(1)


@tool(
    name="turn_off_sound_system",
    description="Turn off the Pioneer AVR",
    aliases=["sound_off", "sound_system_off", "turn_off_sound_system"]
)
def turn_off_sound_system():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_turn_off_sound_system())
    else:
        return loop.create_task(_turn_off_sound_system())
    

# Set Input
async def _set_input_sound_system(input_no: str = '04'):
    """Set the sound system input."""
    try:
        async with AVR(HOST, PORT) as avr:
            await avr.set_input_number(input_no)
    except Exception as e:
        logger.error(f"Pioneer input change failed with error: {e}")
        sys.exit(1)


@tool(
    name="set_input_sound_system",
    description="Set the input for the Pioneer AVR",
    aliases=["sound_input", "sound_system_input", "set_input_sound_system"]
)
def set_input_sound_system(input_type: str = "Music"):
    # Convert input type to number, defaults to music
    input_no = next((k for k, v in DEFAULT_INPUTS.items() if v == input_type.upper()), '04')
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_set_input_sound_system(input_no))
    else:
        return loop.create_task(_set_input_sound_system(input_no))
    
    
# Set Volume
async def _set_volume_sound_system(volume_no: int = 35):
    """Set the sound system volume."""
    try:
        async with AVR(HOST, PORT) as avr:
            await avr.set_volume(volume_no)
    except Exception as e:
        logger.error(f"Pioneer volume change failed with error: {e}")
        sys.exit(1)


@tool(
    name="set_volume_sound_system",
    description="Set the volume for the Pioneer AVR",
    aliases=["volume", "sound_volume", "sound_system_volume", "set_volume_sound_system"]
)
def set_volume_sound_system(volume: Optional[str] = None):
    volume_no = int(volume or 35) # Default to -35dB (90 raw)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_set_volume_sound_system(volume_no))
    else:
        return loop.create_task(_set_volume_sound_system(volume_no))
    

# Increase Volume
async def _increase_volume_sound_system():
    """increase the sound system volume."""
    try:
        async with AVR(HOST, PORT) as avr:
            await avr.update_state()
            await asyncio.sleep(1)
            curr_volume = int(avr.volume or 80) # Default to -35dB (90 raw)
            await avr.set_volume(curr_volume+10)
    except Exception as e:
        logger.error(f"Pioneer volume change failed with error: {e}")
        sys.exit(1)


@tool(
    name="increase_volume_sound_system",
    description="Increase the volume for the Pioneer AVR",
    aliases=["louder", "increase_volume", "increase_sound_volume", "increase_volume_sound_system"]
)
def increase_volume_sound_system():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_increase_volume_sound_system())
    else:
        return loop.create_task(_increase_volume_sound_system())


# Decrease Volume
async def _decrease_volume_sound_system():
    """Decrease the sound system volume."""
    try:
        async with AVR(HOST, PORT) as avr:
            await avr.update_state()
            await asyncio.sleep(1)
            curr_volume = int(avr.volume or 100) # Default to -35dB (90 raw)
            await avr.set_volume(curr_volume-10)
    except Exception as e:
        logger.error(f"Pioneer volume change failed with error: {e}")
        sys.exit(1)


@tool(
    name="decrease_volume_sound_system",
    description="decrease the volume for the Pioneer AVR",
    aliases=["quieter", "decrease_volume", "decrease_sound_volume", "decrease_volume_sound_system"]
)
def decrease_volume_sound_system():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_decrease_volume_sound_system())
    else:
        return loop.create_task(_decrease_volume_sound_system())