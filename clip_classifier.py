# clip_classifier.py
from typing import List, Tuple
import numpy as np
import cv2

import torch
import open_clip
from PIL import Image


class ClipClassifier:
    """
    Clasifica un recorte (crop) en una lista de labels usando CLIP (zero-shot).
    Útil para hackathon cuando YOLO COCO no tiene variedad o confunde objetos.
    """

    def __init__(
        self,
        labels: List[str],
        device: str = "cuda",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai"
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.labels = labels

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()

        tokenizer = open_clip.get_tokenizer(model_name)
        # Prompts simples (puedes mejorar los prompts después)
        texts = [f"a photo of a {lbl}" for lbl in labels]
        self.text_tokens = tokenizer(texts).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(self.text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def classify_bgr(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        # BGR (OpenCV) -> RGB (PIL)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_t)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            logits = (img_features @ self.text_features.T) * 100.0
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])
