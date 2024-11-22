
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

storage_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "screenshot_description_agent",
            "description": "Call this whenever you need to know the current frame of the game. This function takes in no arguments.",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_mouse_action",
            "description": "Call this whenever you need to move the mouse. This function takes in a `MouseAction` enum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mouse_action": {
                        "type": "string",
                        "enum": ["mouse_left"],
                    }
                },
                "required": ["mouse_action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": "Call this whenever you need to get the coordinates for the storage unit. This function takes in no arguments.",
            "parameters": {},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_template_match",
            "description": "Call this whenever you need to get the coordinates of the `take all` button or the `close` button for the storage unit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template_img_path": {
                        "type": "string",
                        "enum": [
                            "/Users/ayushthakur/integrations/RPG/template_matching/data/close_temp.png",
                            "/Users/ayushthakur/integrations/RPG/template_matching/data/take_all_temp.png"
                        ],
                        "description": "Path to the template image file to search for",
                    }
                },
                "required": ["template_img_path"],
            },
        }
    }
]
