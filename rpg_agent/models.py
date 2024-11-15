import torch
import weave
import os
from glob import glob
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
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
