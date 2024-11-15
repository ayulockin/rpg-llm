import torch
import weave
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection


class Owlv2DetectionModel(weave.Model):
    model_name: str = "google/owlv2-base-patch16-ensemble"
    _processor: AutoProcessor
    _model: Owlv2ForObjectDetection

    def model_post_init(self, __context):
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name)

    @weave.op()
    def predict(self, prompts: str[list[str]], image: Image, threshold: float = 0.5):
        prompts = list[prompts] if isinstance(prompts, str) else prompts
        inputs = self._processor(text=prompts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes
        )
        result_dict = []
        for prompt, result in zip(prompts, results):
            boxes = []
            for idx, box in enumerate(result["boxes"]):
                boxes.append(
                    {
                        "bbox": box[idx].tolist(),
                        "confidence": result["scores"][idx].item(),
                        "label": result["labels"][idx],
                    }
                )
            result_dict.append(
                {
                    "prompt": prompt,
                    "boxes": boxes,
                }
            )
        return result_dict
