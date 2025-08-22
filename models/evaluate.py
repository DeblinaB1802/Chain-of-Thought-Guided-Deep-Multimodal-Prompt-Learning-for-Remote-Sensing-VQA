# evaluate.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import clip

from dataloader import RSVQADatasetClip
from text_prompt_learner import TextPromptLearner
from visual_prompt_learner import VisualPromptLearner
from fusionclassifier import FusionClassifier


@torch.no_grad()
def evaluate(
    data_root: str,
    checkpoint_path: str,
    images_subdir: str = "Images_LR/Images_LR",
    batch_size: int = 32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === Load CLIP ===
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # === Load checkpoint ===
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_answer_id = ckpt["class_to_answer_id"]
    prompt_len   = ckpt.get("prompt_len", 4)
    prompt_depth = ckpt.get("prompt_depth", 9)
    num_vprompts = ckpt.get("num_vprompts", 9)

    # === Dataset (GLOBAL FILES, mapping enforced from ckpt) ===
    images_dir = os.path.join(data_root, images_subdir)
    qjson = os.path.join(data_root, "all_questions.json")
    ajson = os.path.join(data_root, "all_answers.json")

    ds = RSVQADatasetClip(
        images_dir=images_dir,
        all_questions_json=qjson,
        all_answers_json=ajson,
        transform=preprocess,
        #max_answers=len(class_to_answer_id),   # ignored when fixed mapping provided
        only_active=True,
        fixed_class_to_answer_id=class_to_answer_id,  # enforce mapping
    )
    num_answers = ds.num_answers

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === Models ===
    text_prompt = TextPromptLearner(
        clip_model, prompt_len=prompt_len, prompt_depth=prompt_depth, num_vprompts=num_vprompts
    ).to(device)
    visual_prompt = VisualPromptLearner(
        clip_visual=clip_model.visual, num_vprompts=num_vprompts, prompt_depth=prompt_depth
    ).to(device)
    embed_dim = clip_model.text_projection.shape[1]
    fusion_head = FusionClassifier(
        embed_dim=embed_dim,
        num_answers=num_answers,
        fusion="gated_tanh",
        hidden_dims=(1024, 512),
        dropout=0.3,
        norm_inputs=True,
    ).to(device)

    text_prompt.load_state_dict(ckpt["text_prompt"], strict=True)
    visual_prompt.load_state_dict(ckpt["visual_prompt"], strict=True)
    fusion_head.load_state_dict(ckpt["fusion_head"], strict=True)

    text_prompt.eval(); visual_prompt.eval(); fusion_head.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, qstr, acls, _, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        acls = acls.to(device, non_blocking=True)
        tokenized = clip.tokenize(list(qstr)).to(device)

        img_feats_aux = clip_model.encode_image(images)
        text_feats, vprompts_list, _ = text_prompt(img_feats_aux, tokenized)
        image_feats = visual_prompt(images, vprompts_list)

        logits = fusion_head(text_feats, image_feats)
        loss = criterion(logits, acls)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == acls).sum().item()
        total += acls.numel()

    print(f"Eval Loss: {total_loss/total:.4f} | Eval Acc: {100.0*correct/total:.2f}%")

