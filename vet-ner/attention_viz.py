# attention_viz.py
# Mostra gráfico de atenção e entidades detectadas para uma frase.

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer
from model import BertNER
from data import LABELS, LABEL2ID, ID2LABEL
from matplotlib.patches import Rectangle

CKPT_DIR   = Path("checkpoints")
OUTPUT_DIR = Path("outputs")
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


def load_model(device="cpu"):
    model = BertNER()
    ckpt  = CKPT_DIR / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("Modelo carregado do checkpoint.")
    else:
        print("Checkpoint não encontrado — usando pesos aleatórios (só para teste de viz).")
    model.to(device)
    model.eval()
    return model


def predict_sentence(sentence: str, model, tokenizer, device="cpu"):
    tokens = sentence.split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)

    preds, attentions = model.predict(input_ids, attention_mask, token_type_ids)

    # Alinha predições de volta com os tokens originais
    word_ids     = encoding.word_ids()
    sub_tokens   = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    pred_labels  = [ID2LABEL.get(p.item(), "O") for p in preds[0]]

    # Mapear primeiro índice de sub-token
    first_subtoken = {}
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in first_subtoken:
            first_subtoken[wid] = i

    # Garante que cada palavra original
    token_preds = []
    for wid in range(len(tokens)):
        if wid in first_subtoken:
            idx = first_subtoken[wid]
            token_preds.append((tokens[wid], pred_labels[idx]))
        else:
            token_preds.append((tokens[wid], "O"))

    return token_preds, attentions, sub_tokens


def plot_attention(attentions, sub_tokens, layer=-1, head=0,
                   save_path=None, title="Attention Weights"):
    """
    Plota o mapa de atenção de uma camada e cabeça específicas.

    layer=-1 usa a última camada .
    """
    # tuple de 12 tensores (batch, n_heads, seq, seq)
    attn = attentions[layer][0, head].detach().cpu().numpy()

    # Limita tokens 
    max_tokens = min(20, len(sub_tokens))
    attn       = attn[:max_tokens, :max_tokens]
    labels     = sub_tokens[:max_tokens]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn, cmap="Blues", aspect="auto")

    ax.set_xticks(range(max_tokens))
    ax.set_yticks(range(max_tokens))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Valores nas células
    for i in range(max_tokens):
        for j in range(max_tokens):
            val = attn[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title}\nCamada {layer} | Cabeça {head}", fontsize=11)
    ax.set_xlabel("Key (token atendido)")
    ax.set_ylabel("Query (token que atende)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_entity_highlights(token_preds: list, save_path=None):
    """
    Plota os tokens coloridos por tipo de entidade predita.
    Mais intuitivo para mostrar o resultado do NER.
    """
    ENTITY_COLORS = {
        "DOENCA":      "#ff6b6b",
        "SINTOMA":     "#ffd93d",
        "MEDICAMENTO": "#6bcb77",
        "ESPECIE":     "#4d96ff",
        "TRATAMENTO":  "#c77dff",
        "O":           "#e0e0e0",
    }

    tokens = [t for t, _ in token_preds]
    labels = [l for _, l in token_preds]

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.9), 2.5))
    ax.set_ylim(0, 1)
    ax.axis("off")

    x = 0.0
    for token, label in zip(tokens, labels):
        entity = label[2:] if label != "O" else "O"
        color  = ENTITY_COLORS.get(entity, "#e0e0e0")
        width  = len(token) * 0.12 + 0.3

        rect = Rectangle((x, 0.25), width, 0.5,
                              color=color, alpha=0.85, linewidth=1,
                              edgecolor="gray")
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.5, token, ha="center", va="center",
                fontsize=9, fontweight="bold" if entity != "O" else "normal")

        if entity != "O":
            ax.text(x + width / 2, 0.15, entity, ha="center", va="center",
                    fontsize=6, color="gray")
        x += width + 0.05

    ax.set_xlim(0, x)

    # Legenda
    legend_x = 0
    for entity, color in ENTITY_COLORS.items():
        if entity == "O":
            continue
        ax.add_patch(Rectangle((legend_x, 0.82), 0.15, 0.12,
                                   color=color, alpha=0.85))
        ax.text(legend_x + 0.17, 0.88, entity, fontsize=7, va="center")
        legend_x += len(entity) * 0.1 + 0.3

    ax.set_title("Entidades Identificadas pelo Modelo", fontsize=11, pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva: {save_path}")
    else:
        plt.show()
    plt.close()


def run(sentence=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        CKPT_DIR if (CKPT_DIR / "vocab.txt").exists() else MODEL_NAME,
        use_fast=True
    )
    model = load_model(device)

    if sentence is None:
        sentence = ("O gato apresentava sintomas de sarna, com lesões cutâneas e coceira intensa, mas melhorou após tratamento com ivermectina, indicando um caso"
                    "sugestivos de lesões cutâneas")

    print(f"\nSentença: {sentence}\n")
    token_preds, attentions, sub_tokens = predict_sentence(
        sentence, model, tokenizer, device
    )

    print("Entidades identificadas:")
    for token, label in token_preds:
        if label != "O":
            print(f"  {token:<20} → {label}")

    # Gera visualizações
    plot_entity_highlights(
        token_preds,
        save_path=OUTPUT_DIR / "entity_highlights.png"
    )
    plot_attention(
        attentions, sub_tokens,
        layer=-1, head=0,
        save_path=OUTPUT_DIR / "attention_last_layer.png",
        title="Atenção — Última Camada"
    )
    plot_attention(
        attentions, sub_tokens,
        layer=0, head=0,
        save_path=OUTPUT_DIR / "attention_first_layer.png",
        title="Atenção — Primeira Camada"
    )
    print(f"\nVisualizações salvas em outputs/")


if __name__ == "__main__":
    run()
