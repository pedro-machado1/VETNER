#!/usr/bin/env python3
"""Calcular class weights para balancear TRATAMENTO"""

import torch
from collections import Counter
from data import RAW_DATA, LABELS, LABEL2ID

# Contar frequência de cada label
label_counts = Counter()
for sentence_pairs in RAW_DATA:
    for word, label in sentence_pairs:
        label_counts[label] += 1

# inverso da frequência
total = sum(label_counts.values())
weights = {}
for label in LABELS:
    count = label_counts.get(label, 1)
    weights[label] = total / (len(LABELS) * count)

# Normalizar
avg_weight = sum(weights.values()) / len(weights)
for label in weights:
    weights[label] /= avg_weight

weights['B-TRATAMENTO'] *= 2.0
weights['I-TRATAMENTO'] *= 2.0

print("Class Weights:")
print("-" * 50)
for label in LABELS:
    w = weights[label]
    bar = "█" * int(w * 5)
    print(f"{label:20} {w:6.2f}  {bar}")

weight_tensor = torch.tensor([weights[LABELS[i]] for i in range(len(LABELS))], 
                             dtype=torch.float32)

print("\n" + "=" * 50)
print("Use isso no model.py:")
print("=" * 50)
print("\nnn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)")
print(f"\nclass_weights = torch.tensor({weight_tensor.tolist()})")
