import torch
import weave
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


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
