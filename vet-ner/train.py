# train.py
# Treino do modelo para NER veterinário.
# Implementação manual para controle total do loop

import torch
import json
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from data import VetNERDataset, get_splits, LABELS
from model import BertNER
from evaluate import evaluate

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 30
BATCH_SIZE  = 2 
LR          = 3e-5   # learning rate padrão 
WARMUP_FRAC = 0.1    # 10% dos steps como warmup
PATIENCE    = 4     # para se não melhorar por 4 épocas
CKPT_DIR    = Path("checkpoints")
MODEL_NAME  = "neuralmind/bert-base-portuguese-cased"


def train():
    print(f"Device: {DEVICE}")
    CKPT_DIR.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # 36 exemplos para validação
    train_data, val_data, test_data = get_splits(val_size=36)

    train_ds = VetNERDataset(train_data, tokenizer)
    val_ds   = VetNERDataset(val_data, tokenizer)
    test_ds  = VetNERDataset(test_data, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BertNER(model_name=MODEL_NAME, use_class_weights=True).to(DEVICE)
    model.count_params()

    # weight decay 
    # bias e LayerNorm sem weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_params, lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1    = 0.0
    patience   = 0
    history    = []

    print(f"\nIniciando treino — {len(train_ds)} exemplos treino | {len(val_ds)} validação")
    print(f"Steps por época: {len(train_loader)} | Warmup: {warmup_steps} steps\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, token_type_ids, labels)
            loss = out["loss"]
            loss.backward()

            # Evita explosão de gradiente em embeddings BERT
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        metrics = evaluate(model, val_loader, DEVICE)
        f1      = metrics["f1_macro"]

        print(f"\nÉpoca {epoch} — Loss médio: {avg_loss:.4f} | "
              f"F1 macro: {f1:.4f} | Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f}")

        history.append({"epoch": epoch, "loss": avg_loss, **metrics})

        # Salva checkpoint se melhorou
        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")
            tokenizer.save_pretrained(CKPT_DIR)
            print(f"  ✓ Novo melhor F1: {best_f1:.4f} — checkpoint salvo\n")
        else:
            patience += 1
            print(f"  Sem melhora ({patience}/{PATIENCE})\n")
            if patience >= PATIENCE:
                print(f"Early stopping na época {epoch}.")
                break

    # Salva histórico de treino
    with open("outputs/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTreino finalizado. Melhor F1 macro: {best_f1:.4f}")
    print(f"Checkpoint em: {CKPT_DIR}/best_model.pt")

    # Avaliação final no conjunto de teste usando o melhor checkpoint salvo
    best_ckpt = CKPT_DIR / "best_model.pt"
    if best_ckpt.exists():
        best_model = BertNER(model_name=MODEL_NAME, use_class_weights=True).to(DEVICE)
        best_model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
        test_metrics = evaluate(best_model, test_loader, DEVICE)
        with open("outputs/test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nAvaliação final (teste) — F1: {test_metrics['f1_macro']:.4f} | "
              f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f}")
        print("Métricas de teste salvas em outputs/test_metrics.json")
    else:
        print("Aviso: melhor checkpoint não encontrado — pulando avaliação final no teste.")


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    train()
