# recognizer.py
import torch
import numpy as np
import cv2
import open_clip
from PIL import Image
from typing import Optional, Tuple, Dict, List

from learned_db import load_embeddings_grouped

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32", copy=False)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n

class Recognizer:
    """
    CLIP embeddings + prototypes (promedio por etiqueta) + fallback kNN.
    """

    def __init__(
        self,
        device: str = "cuda",
        threshold: float = 0.25,          # ✅ baja un poco (0.22-0.28 suele ir mejor)
        ambiguous_margin: float = 0.03,
        use_knn_fallback: bool = True,
        knn_topk: int = 5
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.threshold = float(threshold)
        self.ambiguous_margin = float(ambiguous_margin)
        self.use_knn_fallback = bool(use_knn_fallback)
        self.knn_topk = int(knn_topk)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()

        self.prototypes: Dict[str, np.ndarray] = {}
        self.samples: List[Tuple[str, np.ndarray]] = []

        self.reload()

    def reload(self):
        groups = load_embeddings_grouped()
        self.samples = []
        self.prototypes = {}

        for label, embs in groups.items():
            normed = [l2_normalize(np.asarray(e, dtype=np.float32)) for e in embs]
            for e in normed:
                self.samples.append((label, e))

            proto = np.mean(np.stack(normed, axis=0), axis=0)
            self.prototypes[label] = l2_normalize(proto)

        print(f"✅ Recognizer.reload(): labels={len(self.prototypes)} samples={len(self.samples)}")

    def embed_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        t = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.detach().cpu().numpy()[0].astype("float32")

    def _rank_prototypes(self, emb: np.ndarray) -> List[Tuple[str, float]]:
        ranked = []
        for label, proto in self.prototypes.items():
            ranked.append((label, float(np.dot(emb, proto))))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _knn_vote(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        if not self.samples:
            return None, 0.0

        sims = [(label, float(np.dot(emb, s))) for (label, s) in self.samples]
        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[: max(1, self.knn_topk)]

        score_by_label: Dict[str, float] = {}
        for lbl, sc in top:
            score_by_label[lbl] = score_by_label.get(lbl, 0.0) + sc

        best_lbl = max(score_by_label.items(), key=lambda x: x[1])[0]
        best_score = float(score_by_label[best_lbl] / len(top))
        return best_lbl, best_score

    def recognize_embedding(self, emb: np.ndarray, raw: bool = False):
        """
        raw=False -> si no pasa umbral devuelve label=None
        raw=True  -> devuelve best_label aunque no pase umbral (útil para debug / force_save)
        Returns: (label_or_none, best_score, second_score, ambiguous)
        """
        if not self.prototypes:
            return (None, 0.0, 0.0, False)

        emb = l2_normalize(emb)

        ranked = self._rank_prototypes(emb)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        ambiguous = (best_score - second_score) < self.ambiguous_margin

        if self.use_knn_fallback and ambiguous and self.samples:
            knn_label, knn_score = self._knn_vote(emb)
            if knn_label is not None and knn_score >= best_score:
                best_label, best_score = knn_label, knn_score

        if raw:
            return (best_label, float(best_score), float(second_score), bool(ambiguous))

        if best_score >= self.threshold:
            return (best_label, float(best_score), float(second_score), bool(ambiguous))

        return (None, float(best_score), float(second_score), bool(ambiguous))

    def predict_embedding(self, emb: np.ndarray, raw: bool = False) -> Tuple[Optional[str], float]:
        label, score, _, _ = self.recognize_embedding(emb, raw=raw)
        return label, float(score)

    def predict_bgr(self, img_bgr: np.ndarray, raw: bool = False) -> Tuple[Optional[str], float]:
        emb = self.embed_bgr(img_bgr)
        return self.predict_embedding(emb, raw=raw)
