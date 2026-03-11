export interface Env {
  DB: any // D1Database in Cloudflare env
  MODAL_GENERATE_URL: string
  MODAL_BEARER_TOKEN?: string
}

type ModalBond = [number, number, number]

type ModalTrajectoryFrame = {
  t: number
  ligand_pos: number[][]
  ligand_type: number[]
  ligand_atomic_nums: number[]
  bonds: ModalBond[]
}

type ModalSample = {
  sample_idx: number
  status: "completed" | "failed"
  error: string | null
  n_atoms: number | null
  ligand_pos: number[][] | null
  ligand_type: number[] | null
  ligand_atomic_nums: number[] | null
  trajectory: ModalTrajectoryFrame[] | null
  smiles: string | null
  vina_score: number | null
  qed_score: number | null
  sa_score: number | null
}

type ModalResponse = {
  samples: ModalSample[]
  summary: {
    n_samples: number
    n_valid: number
    n_invalid: number
    vina_mean: number | null
    qed_mean: number | null
    sa_mean: number | null
  }
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
  })
}

function parseIntField(value: FormDataEntryValue | null, fallback: number, min: number, max: number): number {
  if (typeof value !== "string" || value.trim() === "") {
    return fallback
  }
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    throw new Error(`value must be an integer in [${min}, ${max}]`)
  }
  return parsed
}

function parseBoolField(value: FormDataEntryValue | null, fallback: boolean): boolean {
  if (typeof value !== "string" || value.trim() === "") {
    return fallback
  }
  const normalized = value.trim().toLowerCase()
  if (["1", "true", "yes", "on"].includes(normalized)) {
    return true
  }
  if (["0", "false", "no", "off"].includes(normalized)) {
    return false
  }
  throw new Error("boolean field must be one of true/false/1/0/yes/no/on/off")
}

function inferElementFromAtomName(atomName: string): string {
  const letters = atomName.replace(/[^A-Za-z]/g, "").toUpperCase()
  if (!letters) {
    return ""
  }
  if (letters.startsWith("SE")) {
    return "SE"
  }
  return letters[0]
}

function computeBoxFromPdbText(pdbText: string): {
  center: [number, number, number]
  size: [number, number, number]
} {
  const mins = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]
  const maxs = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]
  let atomCount = 0

  for (const rawLine of pdbText.split(/\r?\n/)) {
    const line = rawLine.replace(/\r$/, "")
    const record = line.slice(0, 6).trim().toUpperCase()

    if (record !== "ATOM") {
      continue
    }
    if (line.length < 54) {
      continue
    }

    const altLoc = line.slice(16, 17).trim().toUpperCase()
    if (altLoc !== "" && altLoc !== "A") {
      continue
    }

    const atomName = line.slice(12, 16).trim()
    const elementField = line.length >= 78 ? line.slice(76, 78).trim().toUpperCase() : ""
    const element = elementField || inferElementFromAtomName(atomName)
    if (element === "H") {
      continue
    }

    const coords = [
      Number.parseFloat(line.slice(30, 38).trim()),
      Number.parseFloat(line.slice(38, 46).trim()),
      Number.parseFloat(line.slice(46, 54).trim()),
    ]

    if (coords.some((value) => !Number.isFinite(value))) {
      continue
    }

    atomCount += 1
    for (let i = 0; i < 3; i += 1) {
      mins[i] = Math.min(mins[i], coords[i])
      maxs[i] = Math.max(maxs[i], coords[i])
    }
  }

  if (atomCount === 0) {
    throw new Error("uploaded file does not contain any usable ATOM coordinates")
  }

  const center = mins.map((minValue, i) => (minValue + maxs[i]) / 2) as [number, number, number]
  const size = mins.map((minValue, i) => Math.max(22, maxs[i] - minValue + 12)) as [
    number,
    number,
    number,
  ]

  return { center, size }
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes)
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
}

async function markJobFailed(env: Env, jobId: string, message: string, finishedAt: number): Promise<void> {
  await env.DB.prepare(
    `UPDATE jobs
     SET status = ?, error_message = ?, finished_at = ?
     WHERE id = ?`
  )
    .bind("failed", message, finishedAt, jobId)
    .run()
}

export async function onRequestPost(context: { request: Request; env: Env }): Promise<Response> {
  const { request, env } = context
  let jobId = ""

  try {
    const form = await request.formData()
    const filePart = form.get("file")

    if (!(filePart instanceof File)) {
      return jsonResponse({ error: "missing file field" }, 400)
    }

    const samplesPerTarget = parseIntField(form.get("samples_per_target"), 8, 1, 64)
    const returnTrajectory = parseBoolField(form.get("return_trajectory"), false)
    const trajectoryStride = parseIntField(form.get("trajectory_stride"), 1, 1, 100)

    const filename = filePart.name || "uploaded.pdb"
    if (!filename.toLowerCase().endsWith(".pdb")) {
      return jsonResponse({ error: "uploaded file must end with .pdb" }, 400)
    }

    const fileBytes = await filePart.arrayBuffer()
    if (fileBytes.byteLength === 0) {
      return jsonResponse({ error: "uploaded file is empty" }, 400)
    }

    const pdbText = new TextDecoder("utf-8").decode(fileBytes)
    if (!pdbText.includes("ATOM")) {
      return jsonResponse({ error: "uploaded file does not look like a protein PDB" }, 400)
    }

    const { center: boxCenter, size: boxSize } = computeBoxFromPdbText(pdbText)
    const pdbSha256 = await sha256Hex(fileBytes)
    const now = Date.now()
    jobId = crypto.randomUUID()

    await env.DB.prepare(
      `INSERT INTO jobs (
         id,
         filename,
         pdb_text,
         pdb_sha256,
         status,
         samples_requested,
         return_trajectory,
         trajectory_stride,
         box_center_json,
         box_size_json,
         created_at,
         started_at
       )
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
      .bind(
        jobId,
        filename,
        pdbText,
        pdbSha256,
        "running",
        samplesPerTarget,
        returnTrajectory ? 1 : 0,
        trajectoryStride,
        JSON.stringify(boxCenter),
        JSON.stringify(boxSize),
        now,
        now
      )
      .run()

    const headers: Record<string, string> = {
      "content-type": "application/json",
    }
    if (env.MODAL_BEARER_TOKEN && env.MODAL_BEARER_TOKEN.trim() !== "") {
      headers["authorization"] = `Bearer ${env.MODAL_BEARER_TOKEN.trim()}`
    }

    const modalResp = await fetch(env.MODAL_GENERATE_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({
        pdb_text: pdbText,
        samples_per_target: samplesPerTarget,
        return_trajectory: returnTrajectory,
        trajectory_stride: trajectoryStride,
      }),
    })

    if (!modalResp.ok) {
      const body = await modalResp.text()
      await markJobFailed(env, jobId, `Modal ${modalResp.status}: ${body.slice(0, 2000)}`, Date.now())
      return jsonResponse({ error: "modal inference failed", detail: body.slice(0, 2000) }, 502)
    }

    const modalJson = (await modalResp.json()) as ModalResponse
    const nowDone = Date.now()

    for (const sample of modalJson.samples) {
      await env.DB.prepare(
        `INSERT INTO samples (
           id,
           job_id,
           sample_idx,
           status,
           error_message,
           n_atoms,
           ligand_pos_json,
           ligand_type_json,
           ligand_atomic_nums_json,
           trajectory_json,
           smiles,
           vina_score,
           qed_score,
           sa_score,
           created_at,
           updated_at
         )
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
        .bind(
          crypto.randomUUID(),
          jobId,
          sample.sample_idx,
          sample.status,
          sample.error,
          sample.n_atoms,
          sample.ligand_pos ? JSON.stringify(sample.ligand_pos) : null,
          sample.ligand_type ? JSON.stringify(sample.ligand_type) : null,
          sample.ligand_atomic_nums ? JSON.stringify(sample.ligand_atomic_nums) : null,
          sample.trajectory ? JSON.stringify(sample.trajectory) : null,
          sample.smiles,
          sample.vina_score,
          sample.qed_score,
          sample.sa_score,
          nowDone,
          nowDone
        )
        .run()
    }

    await env.DB.prepare(
      `UPDATE jobs
       SET status = ?, samples_completed = ?, samples_valid = ?, samples_invalid = ?, summary_json = ?, finished_at = ?
       WHERE id = ?`
    )
      .bind(
        "completed",
        modalJson.summary.n_samples,
        modalJson.summary.n_valid,
        modalJson.summary.n_invalid,
        JSON.stringify(modalJson.summary),
        nowDone,
        jobId
      )
      .run()

    return jsonResponse({
      job_id: jobId,
      summary: modalJson.summary,
      samples: modalJson.samples,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : "unknown error"
    if (jobId) {
      await markJobFailed(context.env, jobId, message, Date.now())
    }
    return jsonResponse({ error: message }, 500)
  }
}
