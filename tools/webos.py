"""
Connect to WebOS - LG TV Control
"""
import yaml

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

import asyncio
import socket
from bscpylgtv import WebOsClient
import os
import json

from .lighting import turn_off_lights
from .pioneer_avr import setup_avr
from .tool_registry import tool, tool_registry

import logging

logger = logging.getLogger(__name__)

TV_IP = config['webos']["ip_address"]
TV_MAC = config['webos']["mac_address"]

class LGTVController:
    def __init__(self, tv_ip, mac_address=None):
        self.tv_ip = tv_ip
        self.mac_address = mac_address
        self.client = None
    
    async def connect(self):
        """Connect to the TV"""
        try:
            self.client = await WebOsClient.create(
                self.tv_ip, 
                ping_interval=None, 
                states=[]
            )
            await self.client.connect()
            logger.debug(f"✅ Connected to TV at {self.tv_ip}")
        except Exception as e:
            logger.debug(f"❌ Failed to connect to TV: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from the TV"""
        if self.client:
            await self.client.disconnect()
            logger.debug("🔌 Disconnected from TV")

    def wake_on_lan(self):
        """Turn on TV using Wake-on-LAN"""
        if not self.mac_address:
            logger.debug("❌ MAC address required for Wake-on-LAN")
            return False
            
        try:
            # Remove any separators from MAC address
            mac = self.mac_address.replace(':', '').replace('-', '').upper()
            
            # Create magic packet
            magic_packet = 'FF' * 6 + mac * 16
            magic_packet = bytes.fromhex(magic_packet)
            
            # Send magic packet
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Send to broadcast address
            # broadcast_ip = self.tv_ip.rsplit('.', 1)[0] + '.255'
            # sock.sendto(magic_packet, (broadcast_ip, 9))
            # Direct to TV IP
            sock.sendto(magic_packet, (self.tv_ip, 9))  
            sock.close()         

            logger.debug(f"📺 Wake-on-LAN packet sent to {self.mac_address}")
            return True
            
        except Exception as e:
            logger.debug(f"❌ Failed to send Wake-on-LAN: {e}")
            return False
    
    async def power_on(self):
        """Turn on TV using Wake-on-LAN, then establish connection"""
        logger.debug("🔌 Attempting to turn on TV...")
        
        # Send Wake-on-LAN packet
        if not self.wake_on_lan():
            return "Unable to turn on TV"
        
        # Wait for TV to boot up
        logger.debug("⏳ Waiting for TV to start...")
        await asyncio.sleep(10)  # Give TV time to boot
        
        # Try to establish connection
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                await self.connect()
                return "TV is now on"
            except Exception:
                logger.debug(f"🔄 Connection attempt {attempt + 1}/{max_attempts} failed, retrying...")
                await asyncio.sleep(3)

        return "TV may be on but connection failed"
    
    async def power_off(self):
        """Turn the TV off (standby)"""
        try:
            await self.connect()
            await self.client.power_off()
            return "TV is now off"
        except Exception as e:
            logger.debug(f"❌ Failed to turn off TV: {e}")
            return "Failed to turn off TV"
    
    async def volume_up(self):
        """Increase volume"""
        try:
            await self.connect()
            result = await self.client.volume_up()
            return f"Volume increased to {result}"
        except Exception as e:
            logger.debug(f"❌ Failed to increase volume: {e}")
            return "Failed to increase volume"
    
    async def volume_down(self):
        """Decrease volume"""
        try:
            await self.connect()
            result = await self.client.volume_down()
            return f"Volume decreased to {result}"
        except Exception as e:
            logger.debug(f"❌ Failed to decrease volume: {e}")
            return "Failed to decrease volume"
    
    async def set_volume(self, level):
        """Set volume to specific level (0-100)"""
        try:
            await self.connect()
            result = await self.client.set_volume(level)
            return f"Volume set to {level}"
        except Exception as e:
            logger.debug(f"❌ Failed to set volume: {e}")
            return "Failed to set volume"

@tool(
    name="turn_on_tv",
    description="Turn on the TV",
    aliases=["tv_on", "watch_tv", "tv"]
)
def turn_on_tv():
    tv = LGTVController(TV_IP, TV_MAC)
    """Turn on the TV"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.power_on())
    else:
        return loop.create_task(tv.power_on())
    
@tool(
    name="turn_off_tv",
    description="Turn off the TV",
    aliases=["tv_off", "no_tv"]
)
def turn_off_tv():
    tv = LGTVController(TV_IP, TV_MAC)
    """Turn off the TV"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.power_off())
    else:
        return loop.create_task(tv.power_off())

@tool(
    name="set_tv_volume",
    description="Set TV Volume",
    aliases=["tv_volume", "volume_tv"]
)
def set_tv_volume(new_volume: str):
    tv = LGTVController(TV_IP, TV_MAC)
    """Set TV Volume"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(tv.set_volume(int(new_volume)))
    else:
        return loop.create_task(tv.set_volume(int(new_volume)))

async def _movie_night():
    tv = LGTVController(TV_IP, TV_MAC)

    try:
        # Turn on sound system (do first, takes longest)
        await setup_avr("TV")
        # Turn off lights
        turn_off_lights("Living Room")
        # Turn on TV from standby
        await tv.power_on()
    except Exception as e:
        logger.debug(f"❌ Error: {e}")
    finally:
        await tv.disconnect()

@tool(
    name="movie_night",
    description="Turn on TV, sets up sound system and dims lights",
    aliases=["movie_night", "movie", "watch_movie"]
)
def movie_night():    
    """Setup for Movie Night"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_movie_night())
    else:
        return loop.create_task(_movie_night())

if __name__ == "__main__":    
    logger.debug("🎯 LG webOS TV Controller")
    logger.debug("=" * 30)
    
    asyncio.run(_movie_night())
