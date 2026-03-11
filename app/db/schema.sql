CREATE TABLE jobs (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  filename TEXT NOT NULL,
  pdb_sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  samples_requested INTEGER NOT NULL,
  samples_completed INTEGER NOT NULL DEFAULT 0,
  samples_valid INTEGER NOT NULL DEFAULT 0,
  samples_invalid INTEGER NOT NULL DEFAULT 0,
  return_trajectory INTEGER NOT NULL DEFAULT 0,
  trajectory_stride INTEGER NOT NULL DEFAULT 1,
  box_center_json TEXT NOT NULL,
  box_size_json TEXT NOT NULL,
  summary_json TEXT,
  error_message TEXT,
  created_at INTEGER NOT NULL,
  started_at INTEGER,
  finished_at INTEGER,
  pdb_text TEXT NOT NULL
);

CREATE TABLE samples (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL,
  sample_idx INTEGER NOT NULL,
  status TEXT NOT NULL,
  error_message TEXT,
  n_atoms INTEGER,
  ligand_pos_json TEXT,
  ligand_type_json TEXT,
  ligand_atomic_nums_json TEXT,
  trajectory_json TEXT,
  smiles TEXT,
  vina_score REAL,
  qed_score REAL,
  sa_score REAL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
  UNIQUE (job_id, sample_idx)
);

CREATE INDEX idx_jobs_created
ON jobs(created_at DESC);

CREATE INDEX idx_samples_job_sample_idx
ON samples(job_id, sample_idx);

CREATE INDEX idx_samples_job_vina
ON samples(job_id, vina_score);