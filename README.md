# Phishing Email Detection with DeBERTa, LoRA and Graph Attention

A notebook-based binary classifier that combines contextual email representations with part-of-speech (POS) features to distinguish **safe emails** from **phishing emails**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RuleOfNight/phishing-detector/blob/main/98_55_20%281%29.ipynb)

The saved notebook reports **98.55% test accuracy**. This is a recorded experimental result, not an independently reproduced benchmark. See the evaluation and reproducibility notes below.

## Approach

The model uses two branches:

- **Text branch:** `microsoft/deberta-v3-base` with LoRA adapters on attention query and value projections.
- **POS branch:** learned embeddings of spaCy's fine-grained POS tags, aligned to the text's subword tokens.

Each branch is projected to 256 dimensions and receives a learned type embedding. A custom graph-attention layer combines text and POS nodes using three learned biases: node-type interactions, cross-branch token alignment, and relative positions within each branch. The text branch's final `[CLS]` representation feeds a two-class classification head.

```text
Email text
  ├── DeBERTa-v3 + LoRA → text adapter ───┐
  └── spaCy POS tags → embeddings        │
                       → POS adapter ────┤
                                        ↓
                         Type embeddings + node concatenation
                                        ↓
                         Graph attention with learned biases
                                        ↓
                         Text [CLS] → classification head
                                        ↓
                              Safe / Phishing
```

The instantiated classifier uses **one graph layer with four attention heads**, even though the model class defaults to two layers. Attention is dense over the combined text/POS nodes; the notebook does not construct a dependency-parse graph or use a separate graph library.

## Dataset

Upload a file named `phishing_email.csv` to the notebook's working directory **before the first CSV-reading cell**. The file is not included in this repository.

Required columns:

| Column | Expected content |
| --- | --- |
| `Email Text` | Email text |
| `Email Type` | Exactly `Safe Email` or `Phishing Email` |

Labels map to `0` for safe and `1` for phishing. Rows with missing text or unmapped labels are dropped. The saved run reports **18,634 usable examples**, after removing 16 rows.

The notebook splits row indices into approximately **80% training, 10% validation and 10% test**, using `random_state=42` for both split operations. Splits are not stratified, and no duplicate-removal or sender/domain grouping step is implemented.

The committed notebook does not identify the dataset's original source or license. To reproduce the recorded result, obtain the same dataset and verify its provenance and usage terms; a different CSV with the same columns is not an equivalent benchmark.

## Recorded Results

These values come from saved notebook outputs. Stage names describe notebook sections rather than a verified uninterrupted training history.

| Evaluation stage | Test loss | Test accuracy |
| --- | ---: | ---: |
| Additional joint-training cell | 0.0472 | 98.23% |
| After the two-epoch continuation section | 0.0573 | 98.28% |
| After the five-epoch continuation section | 0.0724 | **98.55%** |

The final continuation section selects a checkpoint by validation accuracy; its best recorded validation accuracy is **98.28%**. Test accuracy is then measured after loading the selected checkpoint.

**Evaluation limits:** the same test split is evaluated at multiple stages, so 98.55% should not be presented as a single untouched final-test result. The notebook does not report precision, recall, F1, a confusion matrix, multiple-seed results, a plain DeBERTa baseline or component ablations. Accuracy alone does not establish the phishing miss rate or performance on new email sources.

## Model and Training Settings

| Setting | Value in the notebook |
| --- | --- |
| Backbone | `microsoft/deberta-v3-base` |
| LoRA targets | `query_proj`, `value_proj` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.1 |
| POS embedding size | 128 |
| Shared hidden size | 256 |
| Graph layers / heads | 1 / 4 |
| Maximum sequence length | 256, including special tokens |
| Text preprocessing limit | First 1,500 characters |
| Batch size | 16 |
| Loss | Cross-entropy |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |

Warm-up freezes both DeBERTa and its LoRA parameters while training the POS embeddings, projection adapters, type embeddings, graph layer and classifier at `1e-3`. Joint training unfreezes LoRA while keeping the base DeBERTa weights frozen.

The initial joint-training code sets LoRA learning rate to `5e-3` and other trainable parameters to `5e-4`. A later cell recreates the optimizer with LoRA learning rate `3e-4` and other trainable parameters `5e-4`; continuation loops use that `optimizer_joint`.

The printed **589,824 trainable parameters** refer only to the LoRA-equipped DeBERTa component, not the full classifier. The graph layer, POS branch, adapters and classification head add trainable parameters.

## Run in Google Colab

1. Open the notebook using the badge above and save a working copy.
2. Select a GPU runtime. The saved metadata identifies a T4 GPU; memory usage and training time have not been independently measured.
3. Upload `phishing_email.csv` with the required columns before executing `pd.read_csv(...)`.
4. Run the installation cells, dataset definition, encoder and graph-layer cells in order. The notebook downloads DeBERTa and spaCy's `en_core_web_sm`, so internet access is required.
5. Review the execution-order notes below before running training and checkpoint selection.
6. Save `best_phishing_model.pt` outside temporary Colab storage. For later inference, also preserve the tokenizer, POS label mapping, architecture settings and package versions.

The notebook installs `transformers`, `peft`, `pandas` and `spacy`. It also uses PyTorch, NumPy, scikit-learn and tqdm from the runtime. Dependency versions are not pinned. If the runtime reports a missing dependency, install it before proceeding; DeBERTa tokenizer setup may also require SentencePiece in a fresh environment.


## Resources

- [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [spaCy English models](https://spacy.io/models/en)
