const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '')

function buildApiUrl(pathname) {
  return `${API_BASE}${pathname}`
}

function encodeSegment(value) {
  return encodeURIComponent(String(value ?? ''))
}

function parseJsonField(value, fallback) {
  if (value == null || value === '') return fallback
  if (typeof value !== 'string') return value
  try {
    return JSON.parse(value)
  } catch {
    return fallback
  }
}

function normalizeJob(row) {
  if (!row || typeof row !== 'object') return row
  return {
    ...row,
    summary: parseJsonField(row.summary_json, null),
    box_center: parseJsonField(row.box_center_json, null),
    box_size: parseJsonField(row.box_size_json, null),
  }
}

function normalizeSample(row) {
  if (!row || typeof row !== 'object') return row
  return {
    ...row,
    ligand_pos: parseJsonField(row.ligand_pos_json, null),
    ligand_type: parseJsonField(row.ligand_type_json, null),
    ligand_atomic_nums: parseJsonField(row.ligand_atomic_nums_json, null),
    trajectory: parseJsonField(row.trajectory_json, null),
  }
}

export async function uploadTarget(file, options = {}) {
  const {
    samplesPerTarget = 8,
  } = options

  const formData = new FormData()
  formData.append('file', file)
  formData.append('samples_per_target', String(samplesPerTarget))
  formData.append('return_trajectory', 'true')
  formData.append('trajectory_stride', '1')

  const response = await fetch(buildApiUrl('/api/inference'), {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let details = ''
    try {
      const payload = await response.json()
      if (payload?.error) details = ` (${payload.error})`
      else if (payload?.detail) details = ` (${payload.detail})`
    } catch {
      details = ''
    }
    throw new Error(
      `Failed to upload target '${file?.name ?? 'unknown'}': ` +
      `${response.status} ${response.statusText}${details}`,
    )
  }

  return response.json()
}

export async function fetchJobs() {
  const response = await fetch(buildApiUrl('/api/jobs'))
  if (!response.ok) {
    throw new Error(`Failed to fetch jobs: ${response.status} ${response.statusText}`)
  }
  const payload = await response.json()
  const rows = Array.isArray(payload) ? payload : Array.isArray(payload.jobs) ? payload.jobs : []
  return rows.map(normalizeJob)
}

export async function fetchJob(jobId) {
  const response = await fetch(
    buildApiUrl(`/api/jobs?job_id=${encodeSegment(jobId)}`),
  )

  if (!response.ok) {
    let details = ''
    try {
      const payload = await response.json()
      if (payload?.error) details = ` (${payload.error})`
    } catch {
      details = ''
    }
    throw new Error(
      `Failed to fetch job '${jobId}': ${response.status} ${response.statusText}${details}`,
    )
  }

  const payload = await response.json()
  if (Array.isArray(payload)) {
    return payload.length > 0 ? normalizeJob(payload[0]) : null
  }
  if (Array.isArray(payload.jobs)) {
    return payload.jobs.length > 0 ? normalizeJob(payload.jobs[0]) : null
  }
  return normalizeJob(payload)
}

export async function fetchSamples(jobId) {
  const response = await fetch(
    buildApiUrl(`/api/samples?job_id=${encodeSegment(jobId)}`),
  )

  if (!response.ok) {
    let details = ''
    try {
      const payload = await response.json()
      if (payload?.error) details = ` (${payload.error})`
    } catch {
      details = ''
    }
    throw new Error(
      `Failed to fetch samples for job '${jobId}': ${response.status} ${response.statusText}${details}`,
    )
  }

  const payload = await response.json()
  const rows = Array.isArray(payload) ? payload : Array.isArray(payload.samples) ? payload.samples : []
  return rows.map(normalizeSample)
}

export function sortSamplesByVina(samples) {
  const rows = Array.isArray(samples) ? [...samples] : []
  rows.sort((a, b) => {
    const sampleA = Number(a?.sample_idx ?? 0)
    const sampleB = Number(b?.sample_idx ?? 0)
    const vinaA = Number(a?.vina_score)
    const vinaB = Number(b?.vina_score)
    const hasVinaA = Number.isFinite(vinaA)
    const hasVinaB = Number.isFinite(vinaB)

    if (hasVinaA && hasVinaB) {
      if (vinaA !== vinaB) return vinaA - vinaB
      return sampleA - sampleB
    }
    if (hasVinaA) return -1
    if (hasVinaB) return 1
    return sampleA - sampleB
  })
  return rows
}
