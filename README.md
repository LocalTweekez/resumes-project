# resume-project — ML Subsystem (Profiler + Recommender)

This is the Python ML subproject that powers **profiling and job recommendations** for _resume-project_. It parses CVs + letters + form data, builds a **profile**, retrieves relevant jobs from a vector index, **re-ranks** them with a learned model, and returns a ranked list with **“Why this job”** explanations.

> Stack: Python (FastAPI), sentence-transformers, FAISS/pgvector, XGBoost, Postgres. Integrates with the Java backend and the React frontend.

---

## TL;DR (Quickstart)

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt

# 2) Start API (dev)
uvicorn ml.src.service:app --reload --port 8090

# 3) Build job index (from your jobs API dump)
python ml/scripts/build_job_index.py \
  --jobs ./data/jobs.jsonl \
  --out ./artifacts/index

# 4) Get recommendations
curl "http://localhost:8090/recommend?user_id=<USER_ID>&k=50"
```

**Optional:** Use Postgres + `pgvector` instead of FAISS once stable.

---

## Architecture

```
[React]  →  [Java API Gateway]  →  [ML API (FastAPI)]
                              ├─ /profile  (parse & embed profile)
                              ├─ /retrieve (vector ANN top-K)
                              ├─ /rerank   (learned re-ranker)
                              └─ /recommend (retrieve→filters→rerank→explain)

Data Layer:
- Postgres (users, jobs, interactions, configs)
- Vector Index (FAISS local or pgvector HNSW)
- Object Store (CV/letters; local dir or S3-compatible)
- Artifacts (models, indices): ./artifacts/

Batch/Training:
- Scripts to build weak labels, train ranker, offline eval
- Model registry (MLflow optional)
```

### Request Flow
1. **Onboarding:** Java uploads files → ML `/profile` → profile JSON + embedding stored.
2. **Recommend:** Java calls ML `/recommend` → vector retrieval → hard filters (location, work auth, etc.) → re-rank → explanations.
3. **Feedback:** React emits impression/click/save/apply/dismiss → stored in `interactions` → used for training.

---

## Repository Layout

```
ml/
  requirements.txt
  src/
    data_contracts.py
    text_builders.py
    skills_normalizer.py
    embedder.py
    vector_index.py
    retrieval.py
    features.py
    reranker.py
    explain.py
    service.py              # FastAPI: /profile, /retrieve, /rerank, /recommend
  scripts/
    build_job_index.py
    refresh_job_embeddings.py
    train_reranker.py
    eval_offline.py
  artifacts/                # indices, models
  data/                     # local datasets (ignored in VCS)
```

---

## Data Contracts

**Profile (stored JSON)**
```json
{
  "user_id": "...",
  "roles": ["backend"],
  "seniority": "mid",
  "locations": ["Stockholm"],
  "remote_ok": true,
  "work_auth": "EU",
  "languages": ["English", "Swedish"],
  "skills": [{"name": "Java", "w": 0.9}, {"name": "Spring", "w": 0.8}],
  "prefs": {"industries": ["fintech"], "tech_likes": ["k8s"]},
  "embedding_id": "..."
}
```

**Job (from jobs API, normalized)**
```json
{
  "job_id": "...",
  "title": "Backend Engineer (Java)",
  "skills_req": ["Java", "Spring", "SQL"],
  "skills_plus": ["Kubernetes"],
  "location": "Stockholm",
  "remote_ok": true,
  "company_id": "...",
  "parent_company_id": "...",
  "salary_range": {"min": 48000, "max": 62000, "currency": "SEK"},
  "posted_at": "2025-08-10",
  "desc": "..."
}
```

**Interaction**
```json
{
  "user_id": "...",
  "job_id": "...",
  "event_type": "click",  
  "ts": "2025-08-21T12:34:56Z",
  "rank": 7,
  "page": 1,
  "dwell_ms": 9300
}
```

---

## Configuration

Environment variables (load via `.env` or process env):

| Var | Default | Description |
|-----|---------|-------------|
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `VECTOR_BACKEND` | `faiss` | `faiss` or `pgvector` |
| `DB_URL` | `postgresql://user:pass@localhost:5432/resume` | Postgres connection |
| `OBJECT_STORE_DIR` | `./object_store` | Where CV/letters are stored locally |
| `ARTIFACTS_DIR` | `./artifacts` | Indices and trained models |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Building the Index

1) Export jobs to `./data/jobs.jsonl` (one JSON per line).

2) Run the builder:
```bash
python ml/scripts/build_job_index.py \
  --jobs ./data/jobs.jsonl \
  --out ./artifacts/index \
  --backend faiss   # or pgvector
```
This script constructs a text field per job, embeds with `MODEL_NAME`, and writes to FAISS or upserts to `pgvector`.

---

## API Endpoints (FastAPI)

- `POST /profile` – Parse & embed a user profile
  - **Body:** `{ cv_text, letters[], form{} }`
  - **Returns:** `{ profile_json, embedding_id }`

- `GET /retrieve?profile_id=...&k=500` – Vector ANN retrieval
  - **Returns:** `{ job_ids: [...], scores: [...] }`

- `POST /rerank` – Re-rank a candidate set
  - **Body:** `{ profile_id, job_ids: [...] }`
  - **Returns:** `{ ranked: [{job_id, score, why: [...]}, ...] }`

- `GET /recommend?user_id=...&k=50` – End-to-end (retrieve → filters → re-rank → explain)
  - **Returns:** `{ results: [...], debug?: {...} }`

Example:
```bash
curl -X POST http://localhost:8090/profile \
  -H 'Content-Type: application/json' \
  -d '{"cv_text":"...","letters":["..."],"form":{"desired_roles":"backend"}}'
```

---

## Retrieval & Re-ranking

- **Retrieval:** ANN over normalized embeddings (cosine/IP). Top‑K=300–1000.
- **Hard Filters:** location/remote, work auth, language reqs, seniority, posting age.
- **Re-ranker:** XGBoost `rank:pairwise` using features: cosine, weighted skill overlap, title match score, geo distance/remote flag, salary fit gap, company industry/size match, post age.
- **Explanations:** matched skills + rule hits + (optional) feature contributions.

---

## Training the Re-ranker (Weak Labels)

1) **Generate weak labels**
   - Positive if title matches role family AND required skills ⊆ profile skills.
   - Negatives: sample from low-overlap jobs.
   - Build per-profile pairs (1 positive vs 5 negatives).

2) **Train**
```bash
python ml/scripts/train_reranker.py \
  --train ./data/weak_labels.parquet \
  --out ./artifacts/reranker.json
```

3) **Evaluate**
```bash
python ml/scripts/eval_offline.py \
  --test ./data/eval_set.parquet
```
Metrics: NDCG@10, MRR, Recall@50. Store results in `./artifacts/metrics.json`.

4) **Online feedback loop**
- Log interactions from the app; nightly job rebuilds pairs and retrains if offline metrics improve.

---

## Synthetic Data (Optional for Bootstrapping)

- Use the included prompt templates to synthesize job postings into `data/synth_jobs.jsonl`.
- Validate with format checks (JSON Schema), dedup (MinHash), PII/brand scrub, and distribution balancing.
- Mix with real data later to avoid style bias.

---

## Java Integration Points

- **/profile** is called after upload to create `profile_json` + `embedding_id` (store in Postgres).
- **/recommend** is called to fetch ranked jobs; Java attaches company/parent metadata and returns to React.
- **Job Ingestor (Java)** normalizes jobs, writes to Postgres, and dumps `data/jobs.jsonl` for (re)index.

---

## Development

- **Code style:** black + ruff (optional configs included).
- **Testing:** pytest (unit tests for text building, feature calc, ranking order).
- **Type hints:** mypy-friendly modules.
- **Observability:** structured logs (JSON), optional OpenTelemetry tracing.

Useful commands:
```bash
pytest -q
ruff check ml/src
black ml/src --check
mypy ml/src
```

---

## Security, Privacy, Fairness

- Store raw documents in an isolated object store; keep short-lived access links.
- Exclude sensitive attributes from model features.
- Provide user controls: hide companies/industries; “don’t use cover letter for preferences”.
- Comply with deletion/export requests.

---

## Roadmap

- [ ] Swap FAISS → `pgvector` HNSW for persistence & SQL filtering
- [ ] Add cross-encoder for top‑50 precision re‑ranking
- [ ] Diversity & freshness constraints (MMR, time decay)
- [ ] A/B flags for model versions via Java gateway
- [ ] MLflow registry + canary rollout
- [ ] Skill ontology expansion + embeddings for companies

---

## Troubleshooting

- **Slow recommendations:** reduce K in retrieval; cache results; check CPU BLAS.
- **Empty results:** verify hard filters; relax seniority/location; confirm index built.
- **Inconsistent scores:** ensure normalized embeddings; consistent model/feature versions.

---

## License

MIT (placeholder). Update as needed.

