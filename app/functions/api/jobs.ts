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

  if (jobId) {
    const row = await context.env.DB.prepare(
      `SELECT id, filename, pdb_text, status, samples_requested, samples_completed, samples_valid, samples_invalid, return_trajectory, trajectory_stride, box_center_json, box_size_json, summary_json, error_message, created_at, started_at, finished_at
       FROM jobs
       WHERE id = ?
       LIMIT 1`
    )
      .bind(jobId)
      .first()

    if (!row) {
      return jsonResponse({ error: "job not found" }, 404)
    }

    return jsonResponse(row)
  }

  const rows = await context.env.DB.prepare(
    `SELECT id, filename, status, samples_requested, samples_completed, samples_valid, samples_invalid, return_trajectory, trajectory_stride, box_center_json, box_size_json, summary_json, error_message, created_at, started_at, finished_at
     FROM jobs
     ORDER BY created_at DESC
     LIMIT 100`
  ).all()

  return jsonResponse(rows.results ?? [])
}