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

O dataset contém **187 frases anotadas manualmente** no formato BIO, divididas em:

| Split      | Frases |
|:-----------|:------:|
| Treino     | 135    |
| Validação  | 36     |
| Teste      | 16     |

Distribuição de entidades (contagem de spans `B-`):

| Entidade      | Spans |
|:--------------|:-----:|
| `DOENCA`      | 171   |
| `ESPECIE`     | 167   |
| `SINTOMA`     | 155   |
| `TRATAMENTO`  | 76    |
| `MEDICAMENTO` | 70    |
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

### Evolução do treino

A cada época, o modelo é avaliado no conjunto de **validação (36 frases)**, que é usado para monitorar o progresso, salvar o melhor checkpoint e acionar o early stopping — portanto essas frases fazem parte do processo de treino. O treino parou na época 12 por early stopping (patience=4 sem melhora após a época 8).

| Época | Loss     | Precision | Recall | F1 Macro |
|:-----:|:--------:|:---------:|:------:|:--------:|
| 1     | 2.2056   | 0.000     | 0.000  | 0.000    |
| 2     | 1.1831   | 0.200     | 0.015  | 0.028    |
| 3     | 0.3048   | 0.787     | 0.828  | 0.807    |
| 4     | 0.0832   | 0.823     | 0.866  | 0.844    |
| 5     | 0.0417   | 0.873     | 0.873  | 0.873    |
| 6     | 0.0387   | 0.852     | 0.858  | 0.855    |
| 7     | 0.0083   | 0.892     | 0.866  | 0.879    |
| **8** | **0.0057** | **0.930** | **0.896** | **0.913** ← melhor |
| 9     | 0.0052   | 0.895     | 0.888  | 0.891    |
| 10    | 0.0041   | 0.877     | 0.903  | 0.890    |
| 11    | 0.0023   | 0.908     | 0.888  | 0.898    |
| 12    | 0.0019   | 0.902     | 0.888  | 0.895    |

### Validação — melhor checkpoint (época 8)

Métricas por span no conjunto de validação (36 frases usadas durante o treino para monitoramento e early stopping).

| Entidade      | Precision | Recall | F1    |
|:--------------|:---------:|:------:|:-----:|
| `ESPECIE`     | 1.000     | 0.970  | 0.985 |
| `SINTOMA`     | 0.978     | 0.936  | 0.957 |
| `DOENCA`      | 0.839     | 0.897  | 0.867 |
| `MEDICAMENTO` | 0.909     | 0.833  | 0.870 |
| `TRATAMENTO`  | 0.800     | 0.615  | 0.696 |
| **Macro**     | **0.930** | **0.896** | **0.913** |

**Parâmetros do melhor checkpoint:**

| Parâmetro | Valor    |
|:----------|:--------:|
| Época     | 8        |
| Loss      | 0.005725 |
| F1 Macro  | 0.9125   |

### Teste — avaliação final (frases nunca vistas)

Após o treino, o melhor checkpoint é carregado e avaliado no conjunto de **teste (16 frases)**, que nunca foram usadas em nenhuma etapa do treino. Essa é a medida mais honesta do desempenho real do modelo.

| Entidade      | Precision | Recall | F1    |
|:--------------|:---------:|:------:|:-----:|
| `ESPECIE`     | 1.000     | 1.000  | 1.000 |
| `MEDICAMENTO` | 1.000     | 1.000  | 1.000 |
| `DOENCA`      | 0.875     | 0.875  | 0.875 |
| `TRATAMENTO`  | 0.625     | 0.833  | 0.714 |
| `SINTOMA`     | 0.722     | 0.813  | 0.765 |
| **Macro**     | **0.831** | **0.891** | **0.860** |

O modelo atinge F1 macro de **0.913** na validação e **0.860** no teste. A entidade `TRATAMENTO` é a mais desafiadora, com menor representatividade no dataset.

