# evaluate.py
# Avaliação do modelo NER no nível de spans (CoNLL-style).
# Retorna precision/recall/f1 geral e por classe.

import torch
from data import ID2LABEL, LABELS


def extract_spans(label_ids: list) -> set:
    """
    Extrai spans de entidades a partir de uma lista. 
    Retorna um conjunto de tuplas. Só conta a entidade se o span estiver completo."""

    spans = set()
    current_start = None
    current_type  = None

    for i, lid in enumerate(label_ids):
        if lid == -100:
            continue
        label = ID2LABEL.get(lid, "O")

        if label.startswith("B-"):
            # Fecha span anterior se existir
            if current_start is not None:
                spans.add((current_start, i - 1, current_type))
            current_start = i
            current_type  = label[2:]

        elif label.startswith("I-"):
            # I- sem B- precedente é ignorado (erro de anotação)
            if current_type is None or label[2:] != current_type:
                if current_start is not None:
                    spans.add((current_start, i - 1, current_type))
                current_start = None
                current_type  = None

        else:  # "O"
            if current_start is not None:
                spans.add((current_start, i - 1, current_type))
            current_start = None
            current_type  = None

    if current_start is not None:
        spans.add((current_start, len(label_ids) - 1, current_type))

    return spans


def compute_metrics(true_spans: set, pred_spans: set, entity_types: list) -> dict:
    """Calcula precision, recall e F1 global e por tipo de entidade."""
    # Global
    tp = len(true_spans & pred_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Por entidade
    per_entity = {}
    for etype in entity_types:
        t = {s for s in true_spans if s[2] == etype}
        p = {s for s in pred_spans if s[2] == etype}
        e_tp = len(t & p)
        e_fp = len(p - t)
        e_fn = len(t - p)
        e_p  = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
        e_r  = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
        e_f1 = (2 * e_p * e_r / (e_p + e_r)) if (e_p + e_r) > 0 else 0.0
        per_entity[etype] = {"precision": e_p, "recall": e_r, "f1": e_f1}

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "per_entity": per_entity,
    }


def evaluate(model, dataloader, device="cpu") -> dict:
    """Roda o modelo no dataloader e retorna métricas de NER por entidade."""
    model.eval()
    entity_types = [l[2:] for l in LABELS if l.startswith("B-")]

    all_true_spans = set()
    all_pred_spans = set()
    offset = 0  # acumula offset de posição entre batches

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            preds, _ = model.predict(input_ids, attention_mask, token_type_ids)

            for i in range(labels.shape[0]):
                true_ids = labels[i].cpu().tolist()
                pred_ids = preds[i].cpu().tolist()

                # posições ignoradas
                masked_pred = [
                    p if t != -100 else -100
                    for p, t in zip(pred_ids, true_ids)
                ]

                t_spans = {(s + offset, e + offset, et)
                           for s, e, et in extract_spans(true_ids)}
                p_spans = {(s + offset, e + offset, et)
                           for s, e, et in extract_spans(masked_pred)}

                all_true_spans |= t_spans
                all_pred_spans |= p_spans
                offset += len(true_ids)

    return compute_metrics(all_true_spans, all_pred_spans, entity_types)


if __name__ == "__main__":
    # Teste 
    from data import LABEL2ID
    ids = [LABEL2ID[l] for l in
           ["O", "B-DOENCA", "I-DOENCA", "O", "B-SINTOMA", "O"]]
    spans = extract_spans(ids)
    print("Spans extraídos:", spans)
    # Esperado: {(1, 2, 'DOENCA'), (4, 4, 'SINTOMA')}
