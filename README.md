# VetNER — NER Clínico Veterinário (BERTimbau)

Projeto para reconhecer entidades em textos clínicos veterinários em português, com fine-tuning do modelo BERTimbau usando PyTorch. A ideia é simples: treinar um classificador token-level (esquema BIO) para extrair entidades como doenças, sintomas e medicamentos em textos de anotações clínicas.

Entidades reconhecidas: `DOENCA`, `SINTOMA`, `MEDICAMENTO`, `ESPECIE`, `TRATAMENTO`.

---

## O que é este repositório

Este repositório contém código para treinar, avaliar e inspecionar um modelo de NER voltado ao domínio veterinário em português. Além do pipeline de treino, há ferramentas para visualizar atenção do modelo e uma demo simples para testar sentenças.

Principais características:
- Treinamento com PyTorch e otimização com `AdamW`.
- Scheduler linear e early stopping para evitar overfitting.
- Avaliação no nível de spans (CoNLL-style): precision / recall / F1.
- Visualizações: heatmaps de atenção e destaques das entidades preditas.

---

## Rápido — como começar

Recomendo usar um ambiente virtual (venv) e Python 3.8+.

```bash
git clone https://github.com/pedro-machado1/vet-ner
cd vet-ner
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\\Scripts\\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

Exemplos úteis:

- Treinar o modelo (salva checkpoints em `checkpoints/`):

```bash
python train.py
```

- Avaliar um checkpoint ou rodar a avaliação padrão:

```bash
python evaluate.py
```

- Gerar visualizações de atenção e destaques de entidade:

```bash
python attention_viz.py
```

- Demo rápida (linha de comando):

```bash
# testar uma única sentença e sair
python demo.py --sentence "O gato apresentou tosse e secreção nasal"

# exemplos prontas
python demo.py --examples

# salvar visualizações com --viz
python demo.py --sentence "O cão foi tratado com ivermectina" --viz
```

---

## Rodando com Docker

Você pode buildar a imagem e rodar em container. Os volumes `checkpoints` e `outputs` são montados para persistência local:

```bash
docker build -t vet-ner:latest .
docker run --rm -it -v $(pwd)/checkpoints:/app/checkpoints -v $(pwd)/outputs:/app/outputs vet-ner:latest

# ou com docker-compose
docker-compose up --build
```

---

## Estrutura do projeto

```
vet-ner/
├── data.py          # Dataset BIO + VetNERDataset (PyTorch)
├── model.py         # BertNER: BERTimbau + classification head
├── train.py         # Loop de treino (AdamW, scheduler, early stopping)
├── evaluate.py      # Métricas por span (precision/recall/F1 por entidade)
├── attention_viz.py # Heatmaps de atenção + entity highlights
├── demo.py          # Interface interativa / CLI
├── requirements.txt
├── checkpoints/     # modelos salvos (gitignored)
└── outputs/         # visualizações e logs
```

---

## Dicas rápidas para treino e avaliação

- Ajuste o `batch_size` e `learning_rate` conforme memória da GPU.
- Use `early stopping` para evitar overfitting; monitorar F1 por entidade ajuda.
- Salve checkpoints com timestamps para facilitar reprodutibilidade.

---

## Problemas comuns

- Erro de memória CUDA: reduza `batch_size` ou use `gradient_accumulation`.
- Tokenização inesperada: verifique o `tokenizer.json` e o alinhamento BIO no `data.py`.

---
