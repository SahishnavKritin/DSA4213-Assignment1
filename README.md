# DSA4213 Assignment 1
## Language Models, Embeddings, and Representations

### Project Overview

This project compares four next-token language models on WikiText-2, investigates the effect of different word embedding strategies, and evaluates learned representations on IMDb sentiment classification.

### Part I: Language Model Comparison

Four models were trained from scratch:

- Interpolated trigram with smoothing
- Vanilla RNN
- LSTM
- Causal Transformer

**Dataset:** WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`)

**Preprocessing:** Lowercase conversion, whitespace tokenization, removal of empty rows, and a 10,000-token vocabulary built from the training split.

Neural-model configuration:
- Context length: 32
- Batch size: 32
- Embedding dimension: 100
- Optimizer: Adam
- Learning rate: 0.001
- Training epochs: 5
- Random seed: 42

### Part II: Embedding Ablation

The LSTM architecture was evaluated using three input-embedding strategies:

1. Randomly initialized, trainable embeddings.
2. Self-trained Skip-gram embeddings, frozen during language-model training.
3. Public GloVe 100-dimensional embeddings, frozen during language-model training.

Skip-gram was trained using Gensim on the WikiText-2 training corpus. Public GloVe vectors were obtained using `glove-wiki-gigaword-100`.

### Part III: Downstream Classification

**Dataset:** IMDb movie reviews (`stanfordnlp/imdb`)

Balanced subsets:
- Training: 4,000 reviews
- Validation: 1,000 reviews
- Test: 1,000 reviews

Representations were extracted from the frozen LSTM and Transformer language models.

Reviews were truncated to 256 tokens, processed in chunks of up to 32 tokens, and represented using mean pooling.

Separate logistic regression classifiers were trained using the same hyperparameters and evaluated using accuracy and macro-F1.

### Environment and Dependencies

- Python 3.12
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Hugging Face Datasets
- Gensim
- tqdm
- Jupyter

Experiments were run on a Windows laptop using CPU computation.

### Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install torch numpy pandas matplotlib scikit-learn datasets gensim tqdm jupyter ipykernel joblib
```

### Reproducing the Experiments

Run the notebooks in numerical order, beginning with data preparation.

The notebooks cover:
1. WikiText-2 preprocessing and vocabulary construction.
2. Trigram training and evaluation.
3. RNN training and evaluation.
4. LSTM training and evaluation.
5. Transformer training and evaluation.
6. Model comparison.
7. Skip-gram embedding training.
8. Frozen Skip-gram LSTM.
9. GloVe preparation and frozen GloVe LSTM.
10. IMDb downstream representation analysis.
11. Embedding-generation comparison.

Some experiments are combined into the same notebook.

Trained checkpoints are saved in `models/`, and evaluation metrics and figures are saved in `results/`.

### Reproducibility Notes

- Random seed: 42
- Vocabulary constructed only from WikiText-2 training data.
- Validation data used for model selection.
- Test data reserved for final evaluation.
- Frozen embedding weights verified to remain unchanged.
- Language models frozen during downstream classification.
- All models trained locally using CPU computation.

### Limitations

- Vocabulary restricted to 10,000 tokens.
- Unknown-token replacement reduces lexical information.
- Neural language models use 32-token context windows.
- IMDb reviews differ from the Wikipedia training corpus.
- Downstream reviews are truncated to 256 tokens.
- Experiments use a limited training budget and one primary random seed.

### External Resources and Tool Use

- WikiText-2 dataset
- IMDb movie-review dataset
- GloVe pretrained word embeddings
- PyTorch, Gensim, Hugging Face Datasets, and Scikit-learn documentation
- ChatGPT assistance for implementation guidance, debugging, and experiment organization

All reported numerical results are obtained from locally executed experiments.