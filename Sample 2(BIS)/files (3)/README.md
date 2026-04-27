# BIS Standards Recommendation Engine
### BIS × Sigma Squad AI Hackathon — Track: AI / RAG

> **Theme:** Accelerating MSE Compliance — Automating BIS Standard Discovery  
> **Focus:** Building Materials (BIS SP 21)

---

## What It Does

Indian Micro and Small Enterprises (MSEs) often spend weeks identifying which Bureau of Indian Standards (BIS) regulations apply to their products. This engine reduces that to **under 100ms** — no internet required.

Given a plain-language product description, the pipeline returns the **top 3–5 ranked BIS standards** with relevance scores and rationale, covering all major Building Materials categories from BIS SP 21.

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-team/bis-rag-engine
cd bis-rag-engine
pip install -r requirements.txt   # No mandatory deps — stdlib only
```

### 2. Run Inference (as judges will)

```bash
python inference.py --input data/public_test_set.json --output data/my_results.json
```

### 3. Evaluate Against Ground Truth

```bash
python eval_script.py \
  --predictions data/my_results.json \
  --ground_truth data/ground_truth.json
```

---

## Repository Structure

```
bis-rag-engine/
├── inference.py              # ← Mandatory judge entry point
├── eval_script.py            # ← Mandatory evaluation script
├── requirements.txt
├── README.md
├── src/
│   ├── rag_engine.py         # Core retrieval: TF-IDF + keyword boosting
│   └── llm_rationale.py      # Optional: Claude API rationale generation
└── data/
    ├── public_test_set.json  # 10 sample queries for local validation
    └── results_public.json   # Results on public test set
```

---

## System Architecture

```
Product Description (text)
        │
        ▼
┌─────────────────────────┐
│   Query Preprocessing   │  tokenize → unigrams + bigrams + trigrams
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   TF-IDF Retrieval      │  field-weighted scoring across:
│   + Keyword Boosting    │  keywords (3.5x), title (2.5x), IS code (4.0x),
│                         │  applications (2.0x), description (1.5x)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Ranked Results        │  top-k standards, relevance % normalised 60–99
│   (IS Code + Title)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Output JSON           │  id, retrieved_standards[], latency_seconds
└─────────────────────────┘
```

---

## Retrieval Strategy

### Chunking & Indexing
Each BIS standard from SP 21 is represented as a structured document with:
- **IS Code** — highest weight field (4.0×)
- **Title** — high weight (2.5×)  
- **Keywords** — curated domain terms (3.5×)
- **Description** — full specification summary (1.5×)
- **Applications** — use-case tags (2.0×)
- **Category** — material category (1.0×)

### Scoring
`score = Σ (field_weight × TF × IDF) + keyword_exact_match_boost + category_boost`

### Why No Embeddings?
- **Zero latency overhead** — pure Python, no GPU, no model loading
- **No hallucination risk** — only returns IS codes from the known BIS SP 21 corpus
- **Avg latency < 5ms** — easily beats the 5-second target
- **Fully reproducible** — no external API dependencies for core inference

---

## Knowledge Base Coverage (BIS SP 21)

| Category | Standards Covered |
|---|---|
| Cement | IS 269, IS 8112, IS 12269, IS 1489 (Pt 1&2), IS 455, IS 12600, IS 8041, IS 6452, IS 3466 |
| Steel | IS 1786, IS 432 (Pt 1), IS 2062, IS 808, IS 1161 |
| Concrete | IS 456, IS 10262, IS 4926, IS 9103, IS 516, IS 1199, IS 3812, IS 15388, IS 1343, IS 13920 |
| Aggregates | IS 383, IS 2386, IS 515, IS 9142 |
| Bricks & Masonry | IS 1077, IS 3495, IS 12894, IS 13757, IS 2185, IS 6441 |
| Tiles & Flooring | IS 1237, IS 654, IS 15622, IS 777 |
| Pipes & Precast | IS 458 |
| Testing | IS 4031, IS 650 |

---

## Input / Output Schema

### Input (`--input`)
```json
[
  {"id": "q01", "query": "Ordinary Portland Cement 53 grade for structural concrete"}
]
```

### Output (`--output`) — **strict schema, do not change key names**
```json
[
  {
    "id": "q01",
    "retrieved_standards": ["IS 12269", "IS 8112", "IS 456"],
    "latency_seconds": 0.0031
  }
]
```

---

## Performance (Public Test Set)

| Metric | Score | Target |
|---|---|---|
| Hit Rate @3 | >80% | >80% |
| MRR @5 | >0.70 | >0.70 |
| Avg Latency | <0.01s | <5s |

---

## Environment

- **Python**: 3.8+
- **Dependencies**: None (stdlib only for inference)
- **Hardware**: Any — no GPU required, runs on CPU
- **OS**: Linux / macOS / Windows

---

## Team & Acknowledgements

Built for the **BIS × Sigma Squad AI Hackathon 2026**.  
Dataset: BIS SP 21 — Summaries of Indian Standards for Building Materials.
