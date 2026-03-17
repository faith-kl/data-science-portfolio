# AI Industry Impact Analysis
### Natural lnaguage Processing Project — Analysis of 200K AI/ML News Articles

---

## Project Overview

This project analyzes approximately **196,629 AI and machine learning news articles** to answer three core business questions:

1. **Who** — Which industries and companies are most impacted by AI?
2. **How** — Are they impacted positively or negatively, and by what AI mechanisms?
3. **Why** — What separates successful AI adoption from unsuccessful adoption?

The full pipeline runs from raw web-crawled text through entity extraction, topic modeling, custom sentiment model training, and synthesis visualization — producing a data-driven view of AI's impact across 16 industries and thousands of companies.

---

## Repository Structure

```
nlp_final_project/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Raw text cleaning and filtering
│   ├── 02_ner_extraction.ipynb         # spaCy organization entity extraction
│   ├── 02b_industry_classification.ipynb  # Gemini API industry labeling
│   ├── 03_topic_modeling.ipynb         # BERTopic topic discovery
│   ├── 04_sentiment_training.ipynb     # RoBERTa fine-tuning
│   ├── 04b_sentiment_inference.ipynb   # Full corpus sentiment scoring
│   └── 06_synthesis_visualization.ipynb  # Analysis and figures
│
├── data/                               # Intermediate artifacts (not tracked in git)
│   ├── cleaned_articles.parquet
│   ├── ner_results.parquet
│   ├── topic_assignments.parquet
│   ├── sentiment_scores.parquet
│   └── company_industry_lookup.csv
│
├── figures/                            # All output visualizations (dpi=300)
│
└── README.md
```

> **Note:** Raw data and large parquet files are not tracked in this repository due to file size. The pipeline reads from and writes to Google Drive. See notebook headers for artifact dependencies.

---

## Pipeline

```
Raw Articles (199,989)
        │
        ▼
01 Data Cleaning ──────────────────────────────► cleaned_articles.parquet (196,629)
        │
        ├──► 02 NER Extraction ─────────────────► ner_results.parquet
        │         └──► 02b Industry Classification    company_industry_lookup.csv
        │
        ├──► 03 Topic Modeling ────────────────► topic_assignments.parquet
        │                                         topic_labels.csv
        │
        ├──► 04 Sentiment Training ───────────► sentiment_model/
        │         └──► 04b Inference ──────────► sentiment_scores.parquet
        │
        └──► 06 Synthesis & Visualization ───► figures/ + summary CSVs
```

All notebooks join on `article_id`.

---

## Notebooks

### 01 — Data Cleaning
Processes raw web-crawled articles through a multi-stage pipeline:
- Exact duplicate removal (494 duplicates dropped)
- Minimum length filtering (200 character threshold)
- Language detection via `pycld2` — non-English articles removed
- Relevance keyword filtering (AI/ML terms) — 1,783 irrelevant articles dropped
- HTML and boilerplate removal via `BeautifulSoup` and regex
- Date parsing and `year_month` feature creation

**Output:** 196,629 clean English AI/ML articles

---

### 02 — Named Entity Recognition
Extracts organization mentions across the full corpus using `spaCy en_core_web_sm`:
- Pipeline restricted to `tok2vec` and `ner` components for efficiency
- Text truncated to first 3,000 characters per article
- Multi-layer noise filtering: boundary words, stock tickers, blocklist (100+ terms), AI noise pattern, structural artifacts
- Batched processing at 25.33 articles/second over ~2.5 hours

**Output:** 194,652 articles with at least one organization extracted

---

### 02b — Industry Classification
Maps extracted companies to industries using the **Gemini API**:
- Articles with >10 org mentions filtered (75th percentile threshold) to remove directory pages
- Companies with fewer than 5 corpus mentions excluded
- Remaining companies classified via Gemini into 13 preset industries + Unknown
- Batched at 50 companies per API call with checkpoint saving
- Model: `gemini-3.1-flash-lite-preview` (switched from `gemini-2.5-flash` due to RPM limits)

**Industries:** Technology, Healthcare, Finance, Legal, Manufacturing, Retail, Energy, Media, Transportation, Education, Government, Real Estate, Telecommunications

---

### 03 — Topic Modeling (BERTopic)
Discovers AI technology themes using **BERTopic** on the full 196,629 article corpus:

| Component | Configuration |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` |
| Dimensionality Reduction | UMAP (n_neighbors=30, n_components=5) |
| Clustering | HDBSCAN (min_cluster_size=0.3% of corpus) |
| Vectorizer | CountVectorizer (1-2 ngrams, min_df=0.001) |
| Topic Reduction | Auto |

- 33 topics discovered; 5 excluded as noise (HDBSCAN outliers, German-language cluster, stock listing pages, EIN Presswire artifacts)
- 28 final topics labeled via Gemini API (4-6 word descriptive labels)
- Topics serve as AI technology/mechanism proxies for downstream industry-level analysis
- Embeddings saved to avoid recompute (~2GB)

---

### 04 — Sentiment Model Training
Fine-tunes `roberta-base` for 3-class sentiment classification:

**Training Data:** Financial PhraseBank — `Sentences_AllAgree` subset (2,264 sentences, unanimous annotator agreement)

| Split | Records |
|---|---|
| Train | 1,635 |
| Validation | 289 |
| Test | 340 |

**Training Configuration:** 4 epochs, lr=2e-5, batch_size=16, FP16, best model selected on F1 Macro

**Test Results:**

| Metric | Score |
|---|---|
| Accuracy | 98.8% |
| F1 Macro | 0.98 |
| F1 Weighted | 0.99 |

> High performance is expected given the `Sentences_AllAgree` subset contains only unambiguous labels. Generalization to noisier news text may be more modest.

---

### 04b — Sentiment Inference
Runs the fine-tuned model across all 196,629 articles on GPU:

- **Token analysis:** 92% of articles exceed 512-token limit (mean: 1,707 tokens)
- **Strategy:** Sliding window chunking — 512-token windows with 128-token stride, max 10 chunks per article
- **Aggregation:** Inverse-frequency weighted mean to correct for neutral-heavy training distribution
- **Checkpointing:** Every 10,000 articles to guard against session interruption

**Final Sentiment Distribution:**

| Label | Count | Share |
|---|---|---|
| Neutral | 135,886 | 69.1% |
| Positive | 54,198 | 27.6% |
| Negative | 6,545 | 3.3% |

---

### 06 — Synthesis & Visualization
Joins all four artifacts on `article_id` to produce cross-dimensional analysis:

- **Industry coverage** — which industries dominate AI discourse
- **Entity-level sentiment** — company and industry net sentiment scores
- **Topic-mechanism analysis** — industry × topic heatmap identifying which AI technologies drive sentiment in which industries
- **Adoption insights** — success and failure signals via TF-IDF keyword extraction
- **Sentiment over time** — stacked area charts, net sentiment trend lines, calendar heatmaps
- **Company deep dive** — bubble charts, top/bottom company rankings, trend lines for top 5 companies

All figures saved at dpi=300 for presentation use.

---

## Key Findings

- No industry carries a net negative AI sentiment score — all 16 industries are net positive on balance
- Average net sentiment across all industries: **+0.31**
- **Healthcare** leads sentiment at +0.44 across ~14,000 articles
- **Transportation** and **Legal** show the most cautious sentiment (+0.18, +0.21) with neutral coverage exceeding 73%
- **Technology** dominates by volume (103,078 articles, 52% of corpus) but carries only moderate sentiment (+0.27) reflecting a maturing narrative
- **AI Advancements in Healthcare Research** is the most broadly positive mechanism across industries
- **AI Copyright and Legal Disputes** frames AI impact in Legal through regulatory disruption rather than operational benefit

---

## Tools & Technologies

| Category | Tools |
|---|---|
| Data Processing | `pandas`, `numpy`, `BeautifulSoup`, `regex`, `pycld2` |
| NLP / NER | `spaCy` (en_core_web_sm) |
| Topic Modeling | `BERTopic`, `sentence-transformers`, `UMAP`, `HDBSCAN` |
| Sentiment Model | `transformers` (RoBERTa), `PyTorch`, `scikit-learn` |
| Generative AI | Gemini API (`gemini-3.1-flash-lite-preview`) |
| Visualization | `matplotlib`, `seaborn` |
| Infrastructure | Google Colab (T4/L4 GPU), Google Drive |
| Data Format | Parquet (via `pyarrow`) |

---

## Dataset Citation

**Financial PhraseBank (Malo et al., 2014)**
```
@article{Malo2014GoodDO,
  title={Good debt or bad debt: Detecting semantic orientations in economic texts},
  author={P. Malo and A. Sinha and P. Korhonen and J. Wallenius and P. Takala},
  journal={Journal of the Association for Information Science and Technology},
  year={2014},
  volume={65}
}
```

---

## Skills Demonstrated

- End-to-end NLP pipeline design and execution at scale (196K+ documents)
- Named entity recognition with custom noise filtering
- Unsupervised topic modeling with BERTopic on large corpora
- Transformer fine-tuning for domain-specific classification
- Long-document inference with sliding window chunking
- Generative AI API integration for structured classification tasks
- Multi-source data fusion and cross-dimensional analysis
- Data visualization for business and technical audiences
