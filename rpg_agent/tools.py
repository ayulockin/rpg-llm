
from .control_interface import KEYSTROKE_STRING_MAPPING

inventory_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "screenshot_description_agent",
            "description": "Call this whenever you need to know the current frame of the game. This function takes in no arguments.",
            "parameters": {},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_keystroke",
            "description": "Call this whenever you need to press a key. This function takes in a `KeyStroke` enum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keystroke": {
                        "type": "string",
                        "enum": list(KEYSTROKE_STRING_MAPPING.keys()),
                    }
                },
                "required": ["keystroke"],
            },
        },
    },
]
