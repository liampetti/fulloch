"""
ThinQ Connect Tool - Currently only gets dishwasher status
"""
import os
import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env vars

with open("./data/config.yml", "r") as f:
    config = yaml.safe_load(f)

import asyncio
from aiohttp import ClientSession
from thinqconnect.thinq_api import ThinQApi


from .tool_registry import tool, tool_registry

import logging

logger = logging.getLogger(__name__)

CREDS = config['thinq']

async def _get_dishwasher_info():
    
    async with ClientSession() as session:
        logger.debug("Created HTTP session")
        
        # Initialize ThinQ API
        try:
            thinq_api = ThinQApi(
                session=session, 
                access_token=CREDS['access_token'],
                country_code=CREDS['country_code'], 
                client_id=CREDS['client_id']
            )
            logger.debug("ThinQ API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ThinQ API: {e}")
            raise
        
        try:
            device_list = await thinq_api.async_get_device_list()
            
            if device_list is None:
                logger.warning("API returned None for device list")
                return None, None, None
            
            logger.debug(f"Device list: {device_list}")
            
            # Filter for dishwasher devices
            dishwashers = [
                device for device in device_list if device.get('deviceInfo').get('deviceType') == 'DEVICE_DISH_WASHER'
            ]
            
            if not dishwashers:
                logger.debug("No dishwasher devices found in device list")
                return None, None, None
            
            logger.debug(f"Dishwasher devices: {[d.get('deviceId') for d in dishwashers]}")
            
            # Get status for each dishwasher
            for i, dishwasher in enumerate(dishwashers, 1):
                device_id = dishwasher.get('deviceId')
                device_name = dishwasher.get('alias', f"Dishwasher {device_id}")
                
                logger.debug(f"Processing dishwasher {i}/{len(dishwashers)}: {device_name} (ID: {device_id})")
                
                try:
                    # Get device status to retrieve timer information
                    logger.debug(f"Fetching status for device {device_id}")
                    status_response = await thinq_api.async_get_device_status(device_id)
                    
                    if status_response:
                        logger.debug(f"Status response for {device_id}: {status_response}")

                        timer_info = status_response.get('timer')
                        state_info = status_response.get('runState')

                        if timer_info is not None:
                            # Extract timer information
                            remain_hours = timer_info.get('remainHour')
                            remain_minutes = timer_info.get('remainMinute')

                        if state_info is not None:
                            run_state = state_info.get('currentState')
                        
                except Exception as e:
                    logger.error(f"Error getting status for dishwasher {device_id} ({device_name}): {e}", exc_info=True)
                    return None, None, None
        except Exception as e:
            logger.error(f"Error retrieving dishwasher information: {e}", exc_info=True)
            return None, None, None
    
    return run_state, remain_hours, remain_minutes

async def _get_dishwasher_text():
    run_state, remain_hours, remain_minutes = await _get_dishwasher_info()

    if run_state is not None:
        status = f"Dishwasher has {remain_hours} hours, {remain_minutes} minutes left. Currently {run_state}."
    else:
        status = "Dishwasher is not running"
        
    return status

@tool(
    name="dishwasher_status",
    description="Get dishwasher status",
    aliases=["time_left_dishwasher", "dishwasher", "is_dishwasher_finished"]
)
def dishwasher_status():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_get_dishwasher_text())
    else:
        return loop.create_task(_get_dishwasher_text())

if __name__ == "__main__": 
    asyncio.run(_get_dishwasher_text())
     


