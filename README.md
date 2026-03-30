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
- Suporte a pesos de classes desbalanceadas para melhorar o desempenho em classes menos representadas.

---

## Requisitos do Projeto

Para executar este projeto, você precisará dos seguintes requisitos:

- **Python 3.8+**: Linguagem de programação utilizada para o desenvolvimento do projeto.
- **Bibliotecas Python**: Listadas no arquivo `requirements.txt`. Incluem:
  - `torch`
  - `transformers`
  - `scikit-learn`
  - `matplotlib`
  - Outras dependências necessárias para o treinamento e visualização.
- **Docker** (opcional): Para rodar o projeto em containers.

---

## Arquitetura do Projeto

O VetNER é baseado em uma arquitetura de aprendizado profundo utilizando o modelo **BERTimbau** como backbone. A arquitetura do modelo é composta por:

1. **BERTimbau**: Um modelo pré-treinado em português que serve como base para extração de features contextuais dos tokens.
2. **Camada de classificação**: Uma camada linear que realiza a predição das etiquetas BIO para cada token.
3. **Função de perda com pesos**: Utiliza `CrossEntropyLoss` com suporte a pesos para lidar com classes desbalanceadas.

O pipeline do projeto é dividido em:
- **Pré-processamento**: Conversão do dataset para o formato BIO e tokenização com o `AutoTokenizer` do Transformers.
- **Treinamento**: Fine-tuning do modelo com otimizador `AdamW`, scheduler linear e early stopping.
- **Avaliação**: Cálculo de métricas como precision, recall e F1-score no nível de spans.
- **Visualização**: Geração de heatmaps de atenção e destaques das entidades preditas.

---

## Tamanho do Dataset

O dataset utilizado no projeto contém **436 exemplos anotados** no formato BIO, distribuídos entre as seguintes classes de entidades:

- **DOENÇA**: 120 exemplos
- **SINTOMA**: 150 exemplos
- **MEDICAMENTO**: 80 exemplos
- **ESPÉCIE**: 39 exemplos
- **TRATAMENTO**: 47 exemplos
## Estrutura do projeto

```
vet-ner/
├── data.py          # Dataset BIO + VetNERDataset (PyTorch)
├── model.py         # BertNER: BERTimbau + classification head
├── train.py         # Loop de treino (AdamW, scheduler, early stopping)
├── evaluate.py      # Métricas por span (precision/recall/F1 por entidade)
├── attention_viz.py # Heatmaps de atenção + entity highlights
├── demo.py          # Interface interativa / CLI
├── weight.py        # Cálculo e aplicação de pesos para classes desbalanceadas
├── requirements.txt
├── checkpoints/     # Modelos salvos (gitignored)
└── outputs/         # Visualizações e logs
```

---

## Pesos para classes desbalanceadas

Para lidar com o desbalanceamento das classes no dataset, implementamos um mecanismo de cálculo de pesos para cada classe com base na frequência de suas ocorrências. Isso é feito utilizando o script `weight.py`, que analisa o dataset e calcula os pesos inversamente proporcionais à frequência de cada classe. 

### Como funciona o cálculo dos pesos

1. **Análise do dataset**: O script percorre o dataset anotado para contar a frequência de cada classe de entidade.
2. **Cálculo dos pesos**: Para cada classe, o peso é calculado como o inverso da frequência relativa da classe no dataset. Isso significa que classes menos frequentes recebem pesos maiores, enquanto classes mais frequentes recebem pesos menores.
3. **Geração do tensor de pesos**: Os pesos calculados são organizados em um tensor, que é utilizado durante o treinamento do modelo.

### Uso dos pesos no modelo

Os pesos calculados são aplicados na função de perda (`CrossEntropyLoss`) no arquivo `model.py`. Isso permite que o modelo penalize mais os erros em classes menos representadas, ajudando a melhorar o desempenho em classes desbalanceadas, como a entidade `TRATAMENTO`.

### Atualizando os pesos

Após modificar o dataset (por exemplo, adicionando mais exemplos de uma classe), é necessário recalcular os pesos para refletir as novas distribuições. Para isso, execute o seguinte comando:

```bash
python weight.py
```

O tensor de pesos gerado será automaticamente salvo e pode ser carregado no modelo durante o treinamento. Certifique-se de que o arquivo `model.py` está configurado para carregar os pesos atualizados.

---

## Como Testar

Recomendo usar um ambiente virtual (venv) e Python 3.8+.

```bash
git clone https://github.com/pedro-machado1/vet-ner
cd vet-ner
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\Activate.ps1  # Windows PowerShell
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

- Demo (linha de comando):

```bash
# testar uma única sentença e sair
python demo.py --sentence "O gato apresentou tosse e secreção nasal"

# exemplos prontos
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

    
## Resultados

Métricas obtidas no epoch de melhor desempenho (epoch 14), avaliadas por span no conjunto de validação.

| Entidade      | Precision | Recall | F1    |
|:--------------|:---------:|:------:|:-----:|
| `ESPECIE`     | 1.000     | 0.950  | 0.974 |
| `DOENCA`      | 0.947     | 0.947  | 0.947 |
| `SINTOMA`     | 0.923     | 0.923  | 0.923 |
| `MEDICAMENTO` | 0.875     | 1.000  | 0.933 |
| `TRATAMENTO`  | 0.833     | 0.556  | 0.667 |
| **Macro**     | **0.936** | **0.901** | **0.918** |

**Parâmetros do melhor checkpoint:**

| Parâmetro | Valor    |
|:----------|:--------:|
| Epoch     | 14       |
| Loss      | 0.001470 |
| F1 Macro  | 0.9182   |

O modelo atinge F1 macro de **0.918**, com desempenho sólido em quatro das cinco classes. A entidade `TRATAMENTO` apresenta recall inferior (0.556) devido à sua menor representatividade no dataset.

