from enum import Enum
from typing import Optional, Union

import pyautogui
import pywinctl
from pynput.mouse import Button, Controller
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


class MouseAction(Enum):
    mouse_left = "mouse_left"
    mouse_right = "mouse_right"
    move = "move"


def press_key(key_code):
    # Create and post key down event
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, key_code, True))
    # Create and post key up event
    CGEventPost(kCGHIDEventTap, CGEventCreateKeyboardEvent(None, key_code, False))


class InputeExecutor:

    def __init__(
        self,
        game_window_title: str = "DOS II",
        game_window_size: tuple[int, int] = (2560, 1440),
    ):
        self.game_window_title = game_window_title
        self.game_window_size = game_window_size
        self.mouse_controller = Controller()
        self.game_window = self.focus_game_window()

    def focus_game_window(self) -> Union[MacOSWindow, None]:
        try:
            all_titles = pywinctl.getAllTitles()
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

    def execute_mouse_action(
        self,
        mouse_action: MouseAction,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> None:
        if self.game_window:
            if mouse_action == MouseAction.mouse_left:
                if x and y:
                    y -= self.game_window_size[1]
                    pyautogui.moveTo(x, y)
                print("HERE")
                self.mouse_controller.click(Button.left)
            elif mouse_action == MouseAction.mouse_right:
                if x and y:
                    y -= self.game_window_size[1]
                    pyautogui.moveTo(x, y)
                self.mouse_controller.click(Button.right)
            elif mouse_action == MouseAction.move:
                y -= self.game_window_size[1]
                pyautogui.moveTo(x, y)
