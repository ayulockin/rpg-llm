from enum import Enum
from typing import Union

import pywinctl
from pywinctl._pywinctl_macos import MacOSWindow
from Quartz.CoreGraphics import (
    CGEventCreateKeyboardEvent,
    CGEventCreateMouseEvent,
    CGEventPost,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
)


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


def move_mouse(x: int, y: int):
    CGEventPost(
        kCGHIDEventTap,
        CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft),
    )


def mouse_left_click(x: int, y: int):
    move_mouse(x, y)
    # Simulate left mouse button down
    CGEventPost(
        kCGHIDEventTap,
        CGEventCreateMouseEvent(
            None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft
        ),
    )
    # Simulate left mouse button up
    CGEventPost(
        kCGHIDEventTap,
        CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft),
    )


def mouse_right_click(x: int, y: int):
    move_mouse(x, y)
    # Simulate left mouse button down
    CGEventPost(
        kCGHIDEventTap,
        CGEventCreateMouseEvent(
            None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonRight
        ),
    )
    # Simulate left mouse button up
    CGEventPost(
        kCGHIDEventTap,
        CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonRight),
    )


class InputeExecutor:

    def __init__(self, game_window_title: str = "DOS II"):
        self.game_window_title = game_window_title
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

    def execute_mouse_action(self, mouse_action: MouseAction, x: int, y: int) -> None:
        if self.game_window:
            if mouse_action == MouseAction.mouse_left:
                mouse_left_click(x, y)
            elif mouse_action == MouseAction.mouse_right:
                mouse_right_click(x, y)
            elif mouse_action == MouseAction.move:
                move_mouse(x, y)
