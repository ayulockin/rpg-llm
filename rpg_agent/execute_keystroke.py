from enum import Enum
from typing import Union

import pywinctl
from pywinctl._pywinctl_macos import MacOSWindow
from Quartz.CoreGraphics import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap


class KeyStroke(Enum):
    enter = 0x24
    command = 0x37
    escape = 0x35
    shift = 0x38
    space = 0x31
    control = 0x3B
    option = 0x3A
    tab = 0x30


def press_key(key_code):
    # Create and post key down event
    event_down = CGEventCreateKeyboardEvent(None, key_code, True)
    CGEventPost(kCGHIDEventTap, event_down)
    # Create and post key up event
    event_up = CGEventCreateKeyboardEvent(None, key_code, False)
    CGEventPost(kCGHIDEventTap, event_up)


class InputeExecutor:

    def __init__(self, game_window_title: str = "DOS II"):
        self.game_window_title = game_window_title
        self.game_window = self.focus_game_window()

    def focus_game_window(self) -> Union[MacOSWindow, None]:
        try:
            all_titles = pywinctl.getAllTitles()
            correct_title = ""
            for title in all_titles:
                if self.game_window_title in title:
                    correct_title = title
                    break
            windows = pywinctl.getWindowsWithTitle(correct_title)
            if windows:
                game_window = windows[0]
                game_window.activate()
                print(f"Activated window: {correct_title}")
                return game_window
            else:
                print(f"No window found with title: {correct_title}")
                return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def execute_keystroke(self, keystroke: KeyStroke) -> None:
        if self.game_window:
            press_key(keystroke.value)
