import io
import os
from glob import glob
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import weave
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)
from ultralytics import YOLO


class Owlv2DetectionModel(weave.Model):
    _processor: Owlv2Processor
    _model: Owlv2ForObjectDetection

    def model_post_init(self, __context):
        self._processor = Owlv2Processor.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        )
        self._model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to("cpu")

    @weave.op()
    def predict(self, prompts: list[list[str]], image: Image, threshold: float = 0.5):
        inputs = self._processor(text=prompts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )
        for idx in range(len(results)):
            results[idx]["boxes"] = results[idx]["boxes"].cpu().tolist()
            results[idx]["scores"] = results[idx]["scores"].cpu().numpy()
            results[idx]["labels"] = results[idx]["labels"].cpu().numpy()
        return results


class UltralyticsDetectionModel(weave.Model):
    model_name: str = "yolo11n"
    _model: YOLO

    def model_post_init(self, __context):
        self._model = YOLO(f"{self.model_name}.pt")

    @weave.op()
    def predict(self, image: Image):
        results = self._model(image)
        result_dict = []
        for idx, result in enumerate(results):
            result_dict.append(
                {
                    "box": result.boxes.xyxyn.tolist(),
                    "score": result.boxes.conf.tolist(),
                    "class": result.boxes.cls.tolist(),
                }
            )
            result.save(f"{self.model_name}-result-{idx}.jpg")

        annotated_images = []
        for image in glob(f"{self.model_name}-result-*.jpg"):
            annotated_images.append(Image.open(image))
            os.remove(image)
        return {
            "detection_results": result_dict,
            "annotated_images": annotated_images,
        }


class Florence2DetectionModel(weave.Model):
    _processor: AutoProcessor
    _model: AutoModelForCausalLM
    model_name: str = "microsoft/Florence-2-large"
    task_prompt: str = "<DENSE_REGION_CAPTION>"

    def model_post_init(self, __context):
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True
        ).eval()

    def plot_bbox(self, image: Image.Image, data: dict) -> Image.Image:
        fig, ax = plt.subplots()
        ax.imshow(image)
        for bbox, label in zip(data["bboxes"], data["labels"]):
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
            )
            ax.add_patch(rect)
            plt.text(
                x1,
                y1,
                label,
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5),
            )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        pil_image = Image.open(buf)
        plt.close(fig)
        return pil_image

    @weave.op()
    def predict(self, image: Image.Image, prompt: Optional[str] = None):
        prompt = self.task_prompt if prompt is None else prompt
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self._processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height),
        )
        return {
            "response": parsed_answer,
            "annotated_image": self.plot_bbox(
                image, parsed_answer[self.task_prompt]
            ),
        }
