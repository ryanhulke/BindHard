export interface Env {
  DB: any
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  })
}

export async function onRequestGet(context: { request: Request; env: Env }): Promise<Response> {
  const url = new URL(context.request.url)
  const jobId = url.searchParams.get("job_id")

  if (!jobId) {
    return jsonResponse({ error: "missing job_id" }, 400)
  }

  const rows = await context.env.DB.prepare(
    `SELECT id, sample_idx, status, error_message, n_atoms, ligand_pos_json, ligand_type_json, ligand_atomic_nums_json, trajectory_json, smiles, vina_score, qed_score, sa_score, created_at, updated_at
     FROM samples
     WHERE job_id = ?
     ORDER BY sample_idx ASC`
  )
    .bind(jobId)
    .all()

  return jsonResponse(rows.results ?? [])
}