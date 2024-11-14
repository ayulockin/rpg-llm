from time import sleep
from rpg_agent.control_interface import InputExecutor, MouseAction, KeyStroke


def search_for_equipped_items(grid_size: int = 60, sleep_time_between_actions: int = 3):
    input_executor = InputExecutor()
    input_executor.execute_mouse_action(mouse_action=MouseAction.move, x=100, y=100)
    sleep(sleep_time_between_actions)

    # Open the inventory
    input_executor.execute_keystroke(KeyStroke.i)

    sleep(sleep_time_between_actions)

    screen_width, screen_height = input_executor.game_window_size

    # Calculate the number of grid cells along each dimension
    num_cells_x = screen_width // grid_size
    num_cells_y = screen_height // grid_size

    # Search for equipped items on the left side of the character model
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            # Calculate the center of the current grid cell
            center_x = (i * grid_size) + (grid_size // 2)
            center_y = (j * grid_size) + (grid_size // 2)

            # Move the mouse to the center of the grid cell
            input_executor.execute_mouse_action(
                MouseAction.move, x=center_x + 80, y=center_y
            )

        break

    sleep(sleep_time_between_actions)

    # Close the inventory
    input_executor.execute_keystroke(KeyStroke.i)
