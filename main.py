import weave
import json
import time
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agents import (
    Agent,
    ScreenshotDescriptionAgent,
    InventoryAgent,
    StorageAgent,
)
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction
from rpg_agent.utils import get_bbox_center

from rpg_agent.agents import ScreenshotDescriptionAgent


weave.init(project_name="ml-colabs/rpg-agent")


# Two agents needed by other agents
GAME_WINDOW_SIZE = (1920, 1080)
executor = InputExecutor(game_window_size=GAME_WINDOW_SIZE)
screenshot_description_agent = ScreenshotDescriptionAgent()

# Microagent 1: Inventory Agent
inventory_agent = InventoryAgent()
# print(inventory_agent.predict(executor, screenshot_description_agent))

# Microagent 2: Storage Agent
storage_agent = StorageAgent()
print(storage_agent.predict(executor))

# executor.execute_mouse_action(MouseAction.mouse_left, 1200, 560)

# executor.execute_keystroke(KeyStroke.i)
