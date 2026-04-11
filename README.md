# BindHard
A structure-based drug discovery platform that generates candidate small-molecule binders for a target protein pocket using a flow matching model built on an equivariant graph neural network (EGNN). Users upload a PDB pocket file, specify a sample count, and receive ranked ligand candidates with 3D binding trajectories, docking scores, and drug-likeness metrics.

---

## Deliverables
| Feature | Status | Notes |
|---|---|---|
| PDB pocket upload (drag-and-drop or file picker) | ✅ Done | Parses ATOM/HETATM counts, chain IDs from uploaded file |
| Configurable sample count (4, 8, 16, 32, or 1–64 custom) | ✅ Done | Validated client-side and server-side (max 64) |
| Cmd+Enter keyboard shortcut to submit | ✅ Done | |
| Serverless GPU inference via Modal | ✅ Done | L4 / A10 / T4 GPU pool; auto-scales to zero |
| Flow matching + EGNN ligand generation | ✅ Done | 100-step ODE sampler, 9-layer attention EGNN |
| Atom count prior (learned distribution) | ✅ Done | Sampled from checkpoint-stored prior at inference time |
| Bond inference via OpenBabel | ✅ Done | `ConnectTheDots` + `PerceiveBondOrders` |
| SMILES generation via RDKit | ✅ Done | Canonical SMILES from sanitized mol |
| AutoDock Vina docking score | ✅ Done | Box computed automatically from pocket atom span |
| QED drug-likeness score | ✅ Done | RDKit `QED.qed` |
| SA (synthetic accessibility) score | ✅ Done | Morgan fingerprint-based, scaled 1–10 |
| Job + sample persistence (SQLite) | ✅ Done | Jobs and samples tables with indexes |
| Results page with sample selector | ✅ Done | Samples sorted by Vina score (lower = better) |
| Molstar 3D trajectory viewer | ✅ Done | Frame-by-frame scrub and animated playback |
| Metrics panel with bar charts vs. baseline | ✅ Done | Vina / QED / SA with CrossDocked dataset baseline |
| SDF export (single sample or batch) | ✅ Done | Filterable by validity, affinity, atom count |
| Landing page with FAQ | ✅ Done | |

### Current Project Status
The end-to-end pipeline is functional. You upload, process to GPU inference, see the results with 3D viewer. The platform is in a research/demo state deployed via Modal and a separate middleware API server.

### Backlog

| Item | Priority | Description |
|---|---|---|
| **Return N valid molecules** | High | Currently returns up to N samples including invalids. Should loop generation until N valid molecules are produced. |
| **Whole-protein pocket extraction** | High | Users currently must pre-extract the binding pocket. Automating pocket detection (e.g. fpocket, P2Rank) as step 1 would remove a key friction point. |
| **RL fine-tuning** | Medium | Define reward function (Vina + SA + QED), implement PPO or GRPO to fine-tune toward higher-quality candidates. |
| **Authentication / user accounts** | Medium | `user_id` column exists in DB schema but is currently unused; no login system is wired up. |
| **Production database** | Low | SQLite stores trajectory JSON inline as text blobs. At scale, migrate to PostgreSQL + object storage for trajectories. |
| **Unit and integration test coverage** | Medium | Only one integration test script exists (`inference_test.py`). |

### Known Issues
- Some fraction of generated samples fail RDKit sanitization and are returned as `status: "failed"`. The generation loop does not retry to compensate.
- Vina scoring requires OpenBabel to convert the RDKit mol to PDBQT format; partial charges are not assigned (rigid receptor scoring only).
- The Modal GPU pool cold-start latency can be 30–90 seconds when the container has scaled to zero.
- Large PDB files (>500 residues) may produce very large `trajectory_json` blobs stored in SQLite.
- The model was trained on CrossDocked v1.1 filtered to strong binders; performance on novel targets outside that distribution is unknown.
---

## Technical Details
### Codebase Overview

```
BindHard/
├── app/
│   ├── backend/
│   │   ├── serve_inference.py       # Modal GPU app + FastAPI inference endpoint
│   │   ├── config/
│   │   │   ├── config.py            # InferenceConfig dataclass
│   │   │   └── inference/
│   │   │       └── flow_matching.yaml  # Active inference hyperparameters
│   │   ├── model/
│   │   │   ├── common.py            # AtomCountPrior, GaussianSmearing, MLP, helpers
│   │   │   ├── egnn.py              # Equivariant GNN layers (MLP + Attention variants)
│   │   │   └── flow_matching.py     # LigandFlowMatching training + sampling
│   │   ├── tests/
│   │   │   └── inference_test.py    # Integration smoke test against live endpoint
│   │   ├── aa_to_index.json         # Amino acid → integer index mapping (20 canonical AAs)
│   │   ├── fpscores.pkl             # SA score fingerprint scores (RDKit contrib)
│   │   └── requirements.modal.txt   # Python dependencies for Modal container
│   ├── frontend/
│   │   ├── src/
│   │   │   ├── App.jsx              # Router setup (landing, dashboard)
│   │   │   ├── pages/
│   │   │   │   ├── Upload.jsx       # PDB upload + generation trigger
│   │   │   │   └── Results.jsx      # Sample viewer, metrics, export
│   │   │   ├── components/
│   │   │   │   ├── DashboardLayout.jsx   # Shared dashboard shell + view mode toggle
│   │   │   │   ├── TrajectoryViewer.jsx  # Molstar 3D renderer + trajectory playback
│   │   │   │   ├── MetricsPanel.jsx      # Bar chart metrics vs. CrossDocked baseline
│   │   │   │   ├── ExportPanel.jsx       # SDF export with filters
│   │   │   │   ├── Navbar.jsx
│   │   │   │   ├── Hero.jsx
│   │   │   │   ├── WhatIsIt.jsx
│   │   │   │   └── FAQ.jsx
│   │   │   └── lib/
│   │   │       ├── trajectoryApi.js  # Fetch wrappers for /api/* endpoints
│   │   │       └── sdfExport.js      # Client-side SDF file generation
│   │   ├── package.json
│   │   └── index.html
│   └── db/
│       ├── schema.sql               # Canonical DB schema
│       └── migrations/
│           └── 0001_init.sql        # Initial migration (identical to schema.sql)
└── README.md
```

**Repositories and Branches**
| Branch | Purpose |
|---|---|
| `master` | Main integration branch |
| `ligand` | Active development branch (current) |
| `features/learnedAtomCount_bondType_samplingGuidance` | Experimental: learned atom count prior + bond type guidance |
| `feature/paper-implementation` | Paper-faithful baseline implementation |

**Version Control:** Git, hosted on GitHub (`ryanhulke/BindHard`). Experiments can be tracked at [wandb.ai/rshulke-university-of-florida/bindhard](https://wandb.ai/rshulke-university-of-florida/bindhard/).

### Dependencies and Libraries
#### Backend

| Library | Version | Purpose |
|---|---|---|
| `modal` | latest | Serverless GPU deployment |
| `torch` | 2.8.0 (cu128) | Deep learning framework |
| `torch-geometric` | 2.7.0 | Graph neural network primitives (knn_graph, radius_graph) |
| `torch_scatter`, `torch_sparse`, `torch_cluster` | pyg wheels | PyG sparse ops |
| `fastapi` | latest | HTTP endpoint (via Modal) |
| `pydantic` | >=2,<3 | Request/response model validation |
| `rdkit` | latest | Cheminformatics: mol sanitization, SMILES, QED, SA, fingerprints |
| `openbabel-wheel` | latest | Geometry → bonded molecule + PDBQT conversion |
| `vina` | latest | AutoDock Vina docking score computation |
| `pyyaml` | latest | Inference config loading |

#### Frontend
| Library | Version | Purpose |
|---|---|---|
| `react` | ^19.2.0 | UI framework |
| `react-router-dom` | ^7.13.0 | Client-side routing |
| `molstar` | ^5.7.0 | 3D molecular structure / trajectory viewer |
| `framer-motion` | ^12.38.0 | Animations (drag-over effects, error shake) |
| `recharts` | ^3.7.0 | Metrics bar charts |
| `tailwindcss` | ^4.2.0 | Utility CSS |
| `lucide-react` | ^1.8.0 | Icons |
| `gsap` | ^3.14.2 | Advanced animation |
| `vite` | ^7.3.1 | Build tool / dev server |

### Deployment Instructions
#### Prerequisites
- Python 3.11
- Node.js 20+
- A [Modal](https://modal.com) account with a secret named `bindhard-inference` containing `INFER_BEARER_TOKEN`
- A Modal volume named `bindhard-model-weights` containing the model checkpoint at `/models/graphAttn_flow_matching_best.pt`

#### Backend
```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Deploy the GPU inference endpoint
cd app/backend
modal deploy serve_inference.py
```

The deploy prints the HTTPS endpoint URL for `LigandGenerator.generate`. Set this as the upstream for your middleware API.

#### Middleware API
A separate API server (not included) handles:
- Receiving multipart form uploads from the frontend (`POST /api/inference`)
- Forwarding PDB text + params to the Modal endpoint with the bearer token
- Writing job and sample rows to SQLite
- Serving `GET /api/jobs` and `GET /api/samples`

#### Frontend
```bash
cd app/frontend
npm install

# Development
VITE_API_BASE=http://localhost:3001 npm run dev

# Production build
npm run build
# Serve the dist/ directory with any static host or reverse proxy
```

#### Environment Variables

| Variable | Where | Description |
|---|---|---|
| `VITE_API_BASE` | Frontend build | Base URL of the middleware API (e.g. `https://api.bindhard.io`) |
| `INFER_BEARER_TOKEN` | Modal secret `bindhard-inference` | Bearer token required to call the GPU inference endpoint |
| `BINDHARD_ENDPOINT` | Test script env | URL of Modal endpoint (for `inference_test.py`) |
| `BINDHARD_TOKEN` | Test script env (optional) | Bearer token for test script |

#### Model Training (HiPerGator / local)

```bash
conda create -n bindhard python=3.11
conda activate bindhard
pip install -r requirements.txt

# Download SA score fingerprints
curl -L https://github.com/rdkit/rdkit/raw/master/Contrib/SA_Score/fpscores.pkl.gz \
  -o data/fpscores.pkl.gz

# Prepare dataset (see research/create_dataset.ipynb)

# Train locally
python research/train.py

# Train on HiPerGator via SLURM
sbatch research/scripts/train.sh
```

### Database Schema
BindHard uses **SQLite**. The schema lives in `app/db/schema.sql`; migrations live under `app/db/migrations/`.

#### `jobs` Table
| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | UUID for the job |
| `user_id` | TEXT | Reserved for future auth (currently NULL) |
| `filename` | TEXT | Original uploaded filename |
| `pdb_sha256` | TEXT | SHA-256 of the uploaded PDB for deduplication |
| `status` | TEXT | `pending` / `running` / `completed` / `failed` |
| `samples_requested` | INTEGER | How many ligands were requested |
| `samples_completed` | INTEGER | How many finished (valid + invalid) |
| `samples_valid` | INTEGER | How many passed RDKit sanitization |
| `samples_invalid` | INTEGER | How many failed |
| `return_trajectory` | INTEGER | 0 or 1 |
| `trajectory_stride` | INTEGER | Stride between trajectory frames stored |
| `box_center_json` | TEXT | Vina search box center `[x, y, z]` (JSON) |
| `box_size_json` | TEXT | Vina search box size `[w, h, d]` (JSON) |
| `summary_json` | TEXT | `SummaryResponse` payload (JSON) |
| `error_message` | TEXT | Top-level error if job failed |
| `created_at` | INTEGER | Unix timestamp |
| `started_at` | INTEGER | Unix timestamp |
| `finished_at` | INTEGER | Unix timestamp |
| `pdb_text` | TEXT | Full PDB file contents |

#### `samples` Table
| Column | Type | Description |
|---|---|---|
| `id` | TEXT PK | UUID for the sample |
| `job_id` | TEXT FK → jobs | Parent job |
| `sample_idx` | INTEGER | 0-based index within the job |
| `status` | TEXT | `completed` or `failed` |
| `error_message` | TEXT | Per-sample error if failed |
| `n_atoms` | INTEGER | Number of ligand atoms |
| `ligand_pos_json` | TEXT | `[[x,y,z], ...]` atom positions (JSON) |
| `ligand_type_json` | TEXT | `[type_idx, ...]` atom type indices (JSON) |
| `ligand_atomic_nums_json` | TEXT | `[atomic_num, ...]` (JSON) |
| `trajectory_json` | TEXT | List of `TrajectoryFrame` objects (JSON) |
| `smiles` | TEXT | Canonical SMILES string |
| `vina_score` | REAL | AutoDock Vina score (kcal/mol; lower = better) |
| `qed_score` | REAL | QED drug-likeness (0–1; higher = better) |
| `sa_score` | REAL | Synthetic accessibility (1–10; lower = harder) |
| `created_at` | INTEGER | Unix timestamp |
| `updated_at` | INTEGER | Unix timestamp |

**Indexes:** `idx_jobs_created` (jobs.created_at DESC), `idx_samples_job_sample_idx` (job_id, sample_idx), `idx_samples_job_vina` (job_id, vina_score).

**Migrations:** Currently one migration (`0001_init.sql`) that creates the full schema from scratch. Apply with:

```bash
sqlite3 bindhard.db < app/db/migrations/0001_init.sql
```

### Software Architecture

**Model architecture detail:**
- `LigandFlowMatching`: learns a conditional vector field from Gaussian noise to real ligand coordinates + atom types, conditioned on protein pocket features.
- `EGNN` denoiser: equivariant GNN with either MLP or graph-attention message passing, Gaussian smearing RBFs for radial distances, coordinate updates preserving SE(3) equivariance.
- Atom types: 7 element classes (C, N, O, F, P, S, Cl) decoded from 0-indexed integers.
- `AtomCountPrior`: empirical distribution learned from training data, stored in the checkpoint `prior` key.


## Testing Information
| Test Type | Coverage | Tool |
|---|---|---|
| Integration (inference endpoint) | Manual / script | `app/backend/tests/inference_test.py` |
| Unit (model, preprocessing) | None formally written | — |
| Frontend (UI flows) | Manual | — |
| System (end-to-end upload → results) | Manual | — |
| Acceptance | Manual demos | — |

### Test Cases
#### `inference_test.py`
Sends 10-sample generation request to the live Modal endpoint with a real pocket PDB file and asserts:
- `response.status_code == 200`
- Response JSON contains `samples` and `summary` keys
- `len(data["samples"]) == 10`

**How to run:**
```bash
export BINDHARD_ENDPOINT=<modal-endpoint-url>
export BINDHARD_TOKEN=<bearer-token>   # optional if no auth
python app/backend/tests/inference_test.py
```

> **Note:** Requires access to `/blue/yanjun.li/ryan.hulke/BindHard/research/data/pocket_examples/protein_pocket10.pdb` (HiPerGator path). Substitute with any valid pocket PDB file.

## User Documentation
### Workflow
```
1. Navigate to the landing page
2. Click "Launch App" or navigate to /dashboard/upload
3. Upload Target
   • Drag-and-drop a .pdb file onto the upload zone
     OR click the zone to open a file picker
   • The interface shows atom count, chain count, and HETATM count
   • Choose number of samples: 4, 8, 16, 32, or a custom value (1–64)
   • Press "Generate Candidate Binders"  (or Cmd+Enter / Ctrl+Enter)
4. Wait for generation
   • Status steps cycle: Preprocessing → Running flow match → Finalizing
   • Typical time: 60–180 seconds depending on sample count and GPU availability
5. Review Results
   • Samples are listed in the left panel, sorted by Vina score (best first)
   • Click a sample to load it in the 3D viewer
6. Explore the 3D Viewer
   • The Molstar viewer renders the protein pocket + selected ligand
   • If trajectory data is available:
     – Use the scrubber to step through frames
     – Press Play to animate the binding trajectory at 20 fps
7. View Metrics
   • Toggle to "Metrics" view to see bar charts for:
     – Vina score (kcal/mol, lower = better)
     – QED drug-likeness (0–1, higher = better)
     – SA score (synthetic accessibility, lower raw = harder to synthesize)
   • Grey bar = CrossDocked dataset baseline
   • Colored bar = selected sample
8. Export
   • Select samples using checkboxes
   • Apply optional filters: valid only, has SMILES, affinity range, atom count range
   • Click "Export SDF" to download a multi-molecule SDF file
```

### FAQs

**What input does BindHard need?**
A PDB file containing your target protein pocket's structure (pocket residues only, not the full protein). The backbone ATOM records are used; HETATM records are ignored during parsing.

**How does the model work?**
BindHard uses a flow matching model on top of an equivariant graph neural network (EGNN). It learns the distribution of binding conformations from the CrossDocked v1.1 dataset (filtered to strong binders) and samples new ligand geometries conditioned on the protein pocket.

**What do the scores mean?**
- **Vina score** (kcal/mol): Estimated binding free energy from AutoDock Vina. More negative = predicted stronger binder. Typical drug-like range: −6 to −12.
- **QED**: Quantitative Estimate of Drug-likeness. 0–1; values above 0.5 are generally considered drug-like.
- **SA score**: Synthetic Accessibility. Raw scale 1–10 (1 = easy, 10 = very hard). Displayed normalized to 0–1 in the metrics panel.

**How are candidates ranked?**
By Vina score, ascending (most negative first). Invalid samples (failed RDKit sanitization) appear at the bottom.

**Can I upload a full protein structure?**
Not yet. You must pre-extract the binding pocket. Tools such as fpocket or PyMOL's `select` command can do this. Automated pocket extraction is on the roadmap.

**Can I use this for drug discovery?**
BindHard is intended for early-stage hit identification. Generated molecules should be treated as computational hypotheses and validated experimentally before drawing any clinical conclusions.

### Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Cmd+Enter` / `Ctrl+Enter` | Submit generation job from Upload page |

