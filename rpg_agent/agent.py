import weave
import json
import cv2
import numpy as np
import random
from PIL import ImageGrab, Image, ImageDraw
import pyautogui
import time
import os

from .llm_predictor import LLMPredictor


def get_game_window():
    """Get the current game window.
    """
    return ImageGrab.grab() # defaults to whole window capture


class ScreenshotDescriptionAgent(weave.Model):
    frame_predictor: LLMPredictor
    prompt: str = """
    We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game. The player or character is likely in the center of the screen. 
    
    We are trying to play this game autonomously thus we need to know what the current state of the game is. Your task is to describe the frame in great detail. Use bullet points and be very specific.
    
    - Tell the count of the objects of interest if there are multiple objects of the same type where necessary.
    - Describe the position of the objects of interest with respect to the player or character.
    - Also describe how far the object of interest is from the player or character.
    """.strip()

    @weave.op()
    def predict(self):
        response = self.frame_predictor.predict(
            system_prompt=self.prompt,
            user_prompts=[get_game_window()],
        )

        return {
            "game_frame": get_game_window(),
            "prediction": response.choices[0].message.content,
        }


class WhereIsTheCharacterAgent(weave.Model):
    object_detector: LLMPredictor  # it can also be a standard object detection model
    prompt_task: str = """We need to detect the main character in the frame of the game. We are playing Divinity: Original Sin 2. Please carefully examine the frame and detect the main character with precise bounding boxes that closely enclose the character. 

    The frame is 1920x1080 (width, height) with the main character likely to be centered in the screen. The bounding boxes should be non-normalized (pixel values). The detection should be precise and accurate because we will use the detected position to control the character.

    Other than the image itself, we are also providing the description of the frame to help you detect the objects better.

    Frame description: {frame_description}
    """.strip()

    prompt_instructions: str = """
    Return ONLY a JSON object containing the bounding boxes. 
        
    Example output:
    {
        "element": "main_character", 
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": 0.95
    }
    """.strip()

    @weave.op()
    def parse_json(self, text: str) -> dict:
        """
        Parse JSON from text that starts with ```json and ends with ```
        """
        import re
        
        # Use regex to extract JSON content between ```json and ``` markers
        pattern = r"```json(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return json.loads(text)
    
        # Extract and parse the JSON content
        json_str = match.group(1).strip()
        return json.loads(json_str)
    
    
    def draw_bbox(self, image, bbox_data, image_size=(1920, 1080)):
        # Convert the image from PIL to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Draw bounding box
        element = bbox_data["element"]
        bbox = bbox_data["bbox"]
        confidence = bbox_data["confidence"]

        x_min, y_min = int(bbox[0]), int(bbox[1])
        x_max, y_max = int(bbox[2]), int(bbox[3])

        # Draw rectangle and label
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(open_cv_image, (x_min, y_min), (x_max, y_max), color, 2)
        label = f"{element} ({confidence:.2f})"
        cv2.putText(open_cv_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert the image back from OpenCV format (BGR) to PIL format (RGB)
        result_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
        return result_image

    @weave.op()
    def predict(self, frame_description: str):
        image = get_game_window()

        system_prompt = self.prompt_task.format(frame_description=frame_description)
        system_prompt += "\n" + self.prompt_instructions

        print(system_prompt)

        response = self.object_detector.predict(
            system_prompt=system_prompt,
            user_prompts=[image],
        )

        try:
            bboxes = self.parse_json(response.choices[0].message.content)
            print(bboxes)
            output = {
                "game_frame": image,
                "prediction_image": self.draw_bbox(image, bboxes),
                "prediction": bboxes,
            }        
        except json.JSONDecodeError:
            output = {
                "game_frame": image,
                "error": str(response.choices[0].message.content),
            }

        return output


@weave.op()
def save_combined_image(before_img, after_img, x, y, count, save_dir):
    """Save concatenated before and after images with coordinates annotation."""
    # Convert OpenCV images to PIL format for combining
    before_pil = Image.fromarray(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
    after_pil = Image.fromarray(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))

    # Concatenate images side by side
    combined_image = Image.new('RGB', (before_pil.width + after_pil.width, before_pil.height))
    combined_image.paste(before_pil, (0, 0))
    combined_image.paste(after_pil, (before_pil.width, 0))

    # Annotate the image with coordinates
    draw = ImageDraw.Draw(combined_image)
    annotation_text = f"Change detected at ({x}, {y})"
    draw.text((10, 10), annotation_text, fill="red")

    # Save the combined image
    filename = f"{save_dir}/detection_{count}_{x}_{y}.png"
    os.makedirs(save_dir, exist_ok=True)
    combined_image.save(filename)
    print(f"Saved detection image: {filename}")

    return combined_image


class HoverDetectionAgent(weave.Model):
    save_dir: str = "detected_changes"
    crop_size: int = 80
    grid_step: int = 50
    contour_threshold: int = 100
    exclusion_size: int = 200
    search_width: int = 500
    search_height: int = 500

    @weave.op()
    def hover_and_detect_around_character(self, character_x, character_y):
        """Hover mouse in a square area around the character, excluding a central region, and detect changes."""
        detected_points = []
        detection_count = 0  # Counter for naming saved images

        # Define the bounds of the search area
        x_min = max(0, character_x - self.search_width // 2)
        x_max = min(pyautogui.size().width, character_x + self.search_width // 2)
        y_min = max(0, character_y - self.search_height // 2)
        y_max = min(pyautogui.size().height, character_y + self.search_height // 2)

        # Loop through points within the defined square search area
        for x in range(x_min, x_max, self.grid_step):
            for y in range(y_min, y_max, self.grid_step):
                # Skip points within the exclusion zone around the character
                if abs(x - character_x) < self.exclusion_size // 2 and abs(y - character_y) < self.exclusion_size // 2:
                    continue

                # Move mouse to the (x, y) position
                pyautogui.moveTo(x, y)
                time.sleep(0.1)  # Small delay to allow the screen to update

                # Capture a larger region around the pointer (e.g., 80x80 pixels)
                region = (x - self.crop_size // 2, y - self.crop_size // 2, x + self.crop_size // 2, y + self.crop_size // 2)
                current_frame = ImageGrab.grab(bbox=region)
                current_frame = cv2.cvtColor(np.array(current_frame), cv2.COLOR_RGB2BGR)

                # Apply edge detection to emphasize highlighted regions
                current_edges = cv2.Canny(current_frame, 50, 150)

                # If we have a previous frame, compare it with the current frame
                if 'previous_edges' in locals():
                    # Calculate absolute difference between edge frames
                    diff = cv2.absdiff(previous_edges, current_edges)
                    
                    # Find contours of the changes
                    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by area to reduce noise
                    significant_change = False
                    for contour in contours:
                        if cv2.contourArea(contour) > self.contour_threshold:
                            significant_change = True
                            break

                    # If a significant contour is found, record this point
                    if significant_change:
                        detected_points.append((x, y))
                        print(f"Significant change detected at ({x}, {y})")

                        # Annotate and save the before and after images side by side
                        combined_image = save_combined_image(previous_frame, current_frame, x, y, detection_count, self.save_dir)
                        detection_count += 1  # Increment the detection count

                # Update the previous frame edges and current frame
                previous_edges = current_edges
                previous_frame = current_frame

        return detected_points
