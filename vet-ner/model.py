# model.py
# BERTimbau + camada simples para NER (token classification)
# Um encoder + um linear para predição por token.

import torch
import torch.nn as nn
from transformers import BertModel
from data import LABELS


class BertNER(nn.Module):
    def __init__(self, model_name="neuralmind/bert-base-portuguese-cased",
                 num_labels=len(LABELS), dropout=0.1):

        super().__init__()
        self.num_labels = num_labels

        # Uversão padrão do BERTimbau com saída de atenção habilitada.
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)
        self.dropout = nn.Dropout(dropout)

        # Projetando para as classes 
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,  # visualização
        )

        sequence_output = self.dropout(output.last_hidden_state)
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            # -100 é ignorado 
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "attentions": output.attentions,  # (batch, heads, seq, seq)
        }

    def predict(self, input_ids, attention_mask, token_type_ids):
        self.eval()

        with torch.no_grad():
            out = self.forward(input_ids, attention_mask, token_type_ids)
        preds = torch.argmax(out["logits"], dim=-1)
        return preds, out["attentions"]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parâmetros: {total:,} total | {trainable:,} treináveis")


if __name__ == "__main__":
    model = BertNER()
    model.count_params()

    # smoke test
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones(2, 128, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 128, dtype=torch.long),
        "labels": torch.randint(-100, len(LABELS), (2, 128)),
    }
    out = model(**batch)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Logits shape: {out['logits'].shape}")  # (2, 128, 11)
    print(f"Attention layers: {len(out['attentions'])}")  # 12 camadas
