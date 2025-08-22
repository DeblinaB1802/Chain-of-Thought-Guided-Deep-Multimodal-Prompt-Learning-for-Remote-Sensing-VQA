# RSVQA LR DATASET LOADER
import os
import json
from typing import Optional, List, Tuple, Dict

import numpy as np
from skimage import io
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------- Utilities ----------

CLASSES_9 = [
    "yes",
    "no",
    "rural",
    "urban",
    "0",
    "between 1 and 10",
    "between 11 and 100",
    "between 101 and 1000",
    "more than 1000",
]
CLASS_TO_INDEX_9 = {t: i for i, t in enumerate(CLASSES_9)}


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert HxW or HxWxC into uint8 RGB [0..255]."""
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 1.0), np.percentile(arr, 99.0)
        if hi <= lo:
            hi = max(arr.max(), lo + 1.0)
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr


def _normalize_to_9classes(raw_text: str) -> Optional[str]:
    """
    Map raw answer text to one of the 9 LR classes.
    Returns the class TEXT or None if it doesn't match the LR buckets.
    """
    t = (raw_text or "").strip().lower()

    # direct text classes
    if t in ("yes", "no", "rural", "urban"):
        return t

    # pure integer?
    if t.isdigit():
        n = int(t)
        if n == 0:
            return "0"
        if 1 <= n <= 10:
            return "between 1 and 10"
        if 11 <= n <= 100:
            return "between 11 and 100"
        if 101 <= n <= 1000:
            return "between 101 and 1000"
        return "more than 1000"

    # if you want to be extra safe, you can try to strip units/spaces:
    # e.g., " 0 ", "000", but LR dataset usually stores raw numerics cleanly.
    return None


# ---------- Dataset ----------

class RSVQADatasetClip(Dataset):
    """
    RSVQA Low-Resolution (LR) dataloader constrained to the 9 classes described in the paper:

    Classes:
      0: "yes"
      1: "no"
      2: "rural"
      3: "urban"
      4: "0"
      5: "between 1 and 10"
      6: "between 11 and 100"
      7: "between 101 and 1000"
      8: "more than 1000"

    Uses FIRST annotator's answer (answers_ids[0]) per question.
    Samples whose raw answer doesn't map to the 9 classes are skipped.

    Returns each item as:
      image_tensor: Tensor [C,H,W]
      question_str: str
      class_index:  int in [0..8]
      question_id:  int
      image_id:     int
    """

    def __init__(
        self,
        images_dir: str,
        all_questions_json: str,
        all_answers_json: str,
        transform: Optional[torch.nn.Module] = None,
        only_active: bool = True,
        strict_image_check: bool = True,
        verbose: bool = True,
        #max_answers: int = 9,
        fixed_class_to_answer_id: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        self.only_active = only_active
        self.strict_image_check = strict_image_check

        # Load global JSONs
        with open(all_questions_json, "r", encoding="utf-8") as f:
            qroot = json.load(f)
        with open(all_answers_json, "r", encoding="utf-8") as f:
            aroot = json.load(f)

        questions = qroot.get("questions", [])
        answers   = aroot.get("answers", [])

        # Map answer_id -> raw string
        self.answer_id_to_string: Dict[int, str] = {
            int(a["id"]): (a.get("answer", "") or "") for a in answers if "id" in a
        }

        if fixed_class_to_answer_id is not None:
            self.class_to_answer_id = fixed_class_to_answer_id
        else:
            self.class_to_answer_id = {cls: idx for idx, cls in enumerate(CLASSES_9)}

        # Build samples (qid, iid, question, cls)
        self.samples: List[Tuple[int, int, str, int]] = []
        skipped_inactive = skipped_no_imgid = skipped_no_ans = 0
        skipped_bad_types = skipped_unmapped = skipped_missing_img = 0

        for q in questions:
            if self.only_active and not q.get("active", True):
                skipped_inactive += 1
                continue

            qid_raw = q.get("id", None)
            iid_raw = q.get("image_id", q.get("img_id", None))  # support both keys
            qstr = (q.get("question", "") or "").strip()
            a_ids = q.get("answers_ids", []) or []

            if iid_raw is None:
                skipped_no_imgid += 1
                continue
            if not a_ids:
                skipped_no_ans += 1
                continue

            try:
                qid = int(qid_raw)
                iid = int(iid_raw)
                aid_first = int(a_ids[0])
            except (TypeError, ValueError):
                skipped_bad_types += 1
                continue

            raw_answer = self.answer_id_to_string.get(aid_first, "")
            mapped_text = _normalize_to_9classes(raw_answer)
            if mapped_text is None:
                skipped_unmapped += 1
                continue

            # ✅ Enforce consistent class index
            if mapped_text not in self.class_to_answer_id:
                skipped_unmapped += 1
                continue

            cls_idx = self.class_to_answer_id[mapped_text]


            # optionally skip if image missing
            img_path = os.path.join(self.images_dir, f"{iid}.tif")
            if self.strict_image_check and (not os.path.exists(img_path)):
                skipped_missing_img += 1
                continue

            self.samples.append((qid, iid, qstr, cls_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                "RSVQADatasetClip: no samples produced. "
                f"skipped_inactive={skipped_inactive}, "
                f"skipped_no_imgid={skipped_no_imgid}, "
                f"skipped_no_ans={skipped_no_ans}, "
                f"skipped_bad_types={skipped_bad_types}, "
                f"skipped_unmapped={skipped_unmapped}, "
                f"skipped_missing_img={skipped_missing_img}"
            )

        if verbose:
            print(
                f"[RSVQADatasetClip] samples={len(self.samples)} | "
                f"skipped inactive={skipped_inactive}, "
                f"no_img_id={skipped_no_imgid}, "
                f"no_answers={skipped_no_ans}, "
                f"bad_types={skipped_bad_types}, "
                f"unmapped_to_9={skipped_unmapped}, "
                f"missing_img={skipped_missing_img}"
            )

        # quick distribution
        self.num_classes = len(CLASSES_9)
        self.class_counts = self._compute_class_counts()
        self.num_answers = len(self.class_to_answer_id)

    def _compute_class_counts(self):
        counts = [0] * len(CLASSES_9)
        for _, _, _, c in self.samples:
            counts[c] += 1
        return counts

    def print_class_distribution(self):
        total = len(self.samples)
        print("Class distribution (9 classes):")
        for i, (name, cnt) in enumerate(zip(CLASSES_9, self.class_counts)):
            pct = 100.0 * cnt / max(1, total)
            print(f"  [{i:>2}] {name:<20s} : {cnt:>6}  ({pct:5.2f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        q_id, img_id, q_str, cls_idx = self.samples[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.tif")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        img_np = io.imread(img_path)
        img_np = _to_uint8_rgb(img_np)
        img_pil = Image.fromarray(img_np, mode="RGB")

        if self.transform is not None:
            img_t = self.transform(img_pil)
        else:
            img_t = T.ToTensor()(img_pil)

        return img_t, q_str, int(cls_idx), int(q_id), int(img_id)


