import weave
from PIL import ImageGrab


class DivinityAgent(weave.Model):
    game_window_size: tuple[int, int] = (2560, 1440)
    
    @weave.op()
    def get_game_window(self):
        return ImageGrab.grab(
            bbox=(0, -self.game_window_size[1], self.game_window_size[0], 0)
        )

    @weave.op()
    def predict(self):
        return self.get_game_window()
