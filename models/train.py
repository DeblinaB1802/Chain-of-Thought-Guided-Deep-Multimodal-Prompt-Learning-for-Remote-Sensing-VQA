# train.py
import os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

import clip

from dataloader import RSVQADatasetClip
from text_prompt_learner import TextPromptLearner
from visual_prompt_learner import VisualPromptLearner
from fusionclassifier import FusionClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_root: str,
    images_subdir: str = "Images_LR/Images_LR",
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_answers: int = 9,
    prompt_len: int = 4,
    prompt_depth: int = 9,
    num_vprompts: int = 9,
    val_ratio: float = 0.1,
    seed: int = 42,
    out_ckpt: str = "checkpoint_best.pth",
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === CLIP ===
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    images_dir = os.path.join(data_root, images_subdir)
    qjson = os.path.join(data_root, "all_questions.json")
    ajson = os.path.join(data_root, "all_answers.json")

    # === Dataset (GLOBAL FILES ONLY) ===
    full_ds = RSVQADatasetClip(
        images_dir=images_dir,
        all_questions_json=qjson,
        all_answers_json=ajson,
        transform=preprocess,
        #max_answers=max_answers,
        only_active=True,
        fixed_class_to_answer_id=None,  # build mapping from global files
        #seed=seed,
    )
    num_answers = full_ds.num_answers
    class_to_answer_id = full_ds.class_to_answer_id  # save for eval

    # random split
    val_size = max(1, int(val_ratio * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # === Models ===
    text_prompt = TextPromptLearner(
        clip_model, prompt_len=prompt_len, prompt_depth=prompt_depth, num_vprompts=num_vprompts
    ).to(device)
    visual_prompt = VisualPromptLearner(
        clip_visual=clip_model.visual, num_vprompts=num_vprompts, prompt_depth=prompt_depth
    ).to(device)

    # NOTE: your FusionClassifier in fusionclassifier.py expects (text_feats, image_feats) in SAME dim.
    # CLIP ViT-B/32 embed_dim is clip_model.text_projection.shape[1].
    embed_dim = clip_model.text_projection.shape[1]
    fusion_head = FusionClassifier(
        embed_dim=embed_dim,
        num_answers=num_answers,
        fusion="gated_tanh",
        hidden_dims=(1024, 512),
        dropout=0.3,
        norm_inputs=True,
    ).to(device)

    # === Optimizer / Loss ===
    params = list(text_prompt.parameters()) + list(visual_prompt.parameters()) + list(fusion_head.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # === Tracking ===
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        text_prompt.train(); visual_prompt.train(); fusion_head.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for images, qstr, acls, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            images = images.to(device, non_blocking=True)
            acls = acls.to(device, non_blocking=True)

            tokenized = clip.tokenize(list(qstr)).to(device)

            # auxiliary frozen image encoder for MetaNet features
            with torch.no_grad():
                img_feats_aux = clip_model.encode_image(images)

            text_feats, vprompts_list, _ = text_prompt(img_feats_aux, tokenized)
            image_feats = visual_prompt(images, vprompts_list)

            logits = fusion_head(text_feats, image_feats)
            loss = criterion(logits, acls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == acls).sum().item()
            total += acls.numel()

        train_loss = epoch_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- Validate ----
        text_prompt.eval(); visual_prompt.eval(); fusion_head.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, qstr, acls, _, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                images = images.to(device, non_blocking=True)
                acls = acls.to(device, non_blocking=True)
                tokenized = clip.tokenize(list(qstr)).to(device)

                img_feats_aux = clip_model.encode_image(images)
                text_feats, vprompts_list, _ = text_prompt(img_feats_aux, tokenized)
                image_feats = visual_prompt(images, vprompts_list)

                logits = fusion_head(text_feats, image_feats)
                loss = criterion(logits, acls)

                v_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == acls).sum().item()
                v_total += acls.numel()

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f} Acc={train_acc*100:.2f}% | "
              f"ValLoss={val_loss:.4f} Acc={val_acc*100:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "text_prompt": text_prompt.state_dict(),
                "visual_prompt": visual_prompt.state_dict(),
                "fusion_head": fusion_head.state_dict(),
                "class_to_answer_id": class_to_answer_id,  # for consistent mapping at eval
                "max_answers": max_answers,
                "prompt_len": prompt_len,
                "prompt_depth": prompt_depth,
                "num_vprompts": num_vprompts,
            }, out_ckpt)

    # ---- Plot curves ----
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    print("Saved curves to training_curves.png")
