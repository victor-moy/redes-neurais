# Classificação de Dígitos Manuscritos com MLP

Trabalho de **Introdução a Redes Neurais Artificiais** — classificação do dataset MNIST usando uma rede Multilayer Perceptron (MLP) implementada em PyTorch, com interface web interativa para desenhar e classificar dígitos em tempo real.

**Acurácia no conjunto de teste: 98.98%**

---

## Demonstração

Desenhe um dígito com o mouse e classifique em tempo real:

- Probabilidades para cada classe (0–9)
- Preview da imagem 28×28 enviada ao modelo
- Atalhos: `Espaço` para classificar, `Esc` para limpar

---

## Estrutura do Projeto

```
redes-neurais/
├── model.py          # Arquitetura MLP (784→512→256→128→10)
├── train.py          # Script de treinamento
├── app.py            # Servidor FastAPI (backend + API)
├── static/
│   └── index.html    # Interface web interativa
├── models/
│   └── mlp_mnist.pth # Pesos do modelo treinado
├── analise.ipynb     # Notebook com análise completa
└── requirements.txt
```

---

## Como Rodar

### Pré-requisitos

- Python 3.9+

### 1. Clone o repositório

```bash
git clone https://github.com/victor-moy/redes-neurais.git
cd redes-neurais
```

### 2. Crie o ambiente virtual e instale as dependências

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Suba o servidor

O modelo já está treinado (`models/mlp_mnist.pth`). Basta iniciar o servidor:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Acesse **http://localhost:8000** no navegador.

### (Opcional) Re-treinar o modelo

```bash
python train.py
```

O dataset MNIST é baixado automaticamente na pasta `data/`. O treinamento leva ~5 minutos em CPU e salva o melhor modelo em `models/mlp_mnist.pth`.

---

## Análise Completa

O notebook `analise.ipynb` contém:

- Exploração e visualização do dataset
- Arquitetura e número de parâmetros do modelo
- Curvas de loss e acurácia por época
- Matriz de confusão (absoluta e normalizada)
- Relatório de métricas por classe (precision, recall, F1)
- Análise qualitativa de acertos e erros

Para abrir:

```bash
jupyter notebook analise.ipynb
```

---

## Arquitetura do Modelo

```
Input (784)
  └─ Linear(512) → BatchNorm → ReLU → Dropout(0.3)
  └─ Linear(256) → BatchNorm → ReLU → Dropout(0.3)
  └─ Linear(128) → BatchNorm → ReLU → Dropout(0.2)
  └─ Linear(10)  ← logits
```

| Hiperparâmetro | Valor |
|---|---|
| Parâmetros totais | ~567 mil |
| Otimizador | Adam (lr=1e-3, weight decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Épocas | 20 |
| Batch size | 256 |
| Data augmentation | Rotação ±10°, Translação ±10%, Escala 90–110% |

---

## Referências

- [MNIST Classification using MLP — Kaggle](https://www.kaggle.com/code/jonathankristanto/mnist-classification-using-multilayer-perceptron)
- [Digit Recognizer — Kaggle](https://www.kaggle.com/c/digit-recognizer/data)
- [Digits Recognition MLP Demo](https://trekhleb.dev/machine-learning-experiments/#/experiments/DigitsRecognitionMLP)
