import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const thisFile = fileURLToPath(import.meta.url)
const projectRoot = path.resolve(path.dirname(thisFile), '..', '..', '..')
const inferenceRoot = path.join(projectRoot, 'inference')

function countModelsInPdbText(text) {
  const lines = String(text ?? '').split(/\r?\n/)
  const modelCount = lines.filter((line) => line.startsWith('MODEL')).length
  if (modelCount > 0) return modelCount
  return lines.some((line) => line.startsWith('ATOM') || line.startsWith('HETATM')) ? 1 : 0
}

function getDirectories(dirPath) {
  if (!fs.existsSync(dirPath)) return []
  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
}

function validateTarget(runId, targetId) {
  const trajectoryDir = path.join(inferenceRoot, runId, targetId, 'trajectories')
  const indexPath = path.join(trajectoryDir, 'index.json')

  if (!fs.existsSync(indexPath)) {
    return { ok: false, issues: [`Missing index.json at ${indexPath}`] }
  }

  const payload = JSON.parse(fs.readFileSync(indexPath, 'utf-8'))
  const entries = Array.isArray(payload.entries) ? payload.entries : []
  const issues = []

  for (const entry of entries) {
    const relPath = String(entry?.path ?? '')
    const pdbPath = path.join(inferenceRoot, runId, targetId, relPath)
    if (!fs.existsSync(pdbPath)) {
      issues.push(`Missing file for sample ${entry?.sample_idx}: ${pdbPath}`)
      continue
    }

    const modelCount = countModelsInPdbText(fs.readFileSync(pdbPath, 'utf-8'))
    const expectedFrames = Number(entry?.n_frames ?? -1)
    if (expectedFrames >= 0 && modelCount !== expectedFrames) {
      issues.push(
        `Frame mismatch sample ${entry?.sample_idx}: manifest=${expectedFrames}, pdb=${modelCount} (${relPath})`,
      )
    }
  }

  return { ok: issues.length === 0, issues, entryCount: entries.length }
}

function parseArgs(argv) {
  const args = { run: null, target: null }
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--run') args.run = argv[i + 1] ?? null
    if (argv[i] === '--target') args.target = argv[i + 1] ?? null
  }
  return args
}

function main() {
  const args = parseArgs(process.argv.slice(2))
  const runs = args.run ? [args.run] : getDirectories(inferenceRoot)

  let targetsChecked = 0
  let failedTargets = 0

  if (runs.length === 0) {
    console.log(`No runs found under ${inferenceRoot}`)
    return
  }

  for (const runId of runs) {
    const runDir = path.join(inferenceRoot, runId)
    const targets = args.target ? [args.target] : getDirectories(runDir)

    for (const targetId of targets) {
      const result = validateTarget(runId, targetId)
      targetsChecked += 1
      if (result.ok) {
        console.log(`[ok] ${runId}/${targetId} (${result.entryCount ?? 0} entries)`)
      } else {
        failedTargets += 1
        console.log(`[fail] ${runId}/${targetId}`)
        for (const issue of result.issues) console.log(`  - ${issue}`)
      }
    }
  }

  if (failedTargets > 0) {
    console.error(`Validation failed: ${failedTargets}/${targetsChecked} target(s) have issues.`)
    process.exit(1)
  } else {
    console.log(`Validation passed for ${targetsChecked} target(s).`)
  }
}

main()
