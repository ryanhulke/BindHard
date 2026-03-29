const ATOMIC_SYMBOLS = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 15: 'P', 16: 'S', 17: 'Cl', 19: 'K', 20: 'Ca', 25: 'Mn', 26: 'Fe', 27: 'Co',28: 'Ni', 29: 'Cu', 30: 'Zn', 34: 'Se', 35: 'Br', 53: 'I',}

function buildMolBlock(sample, name) {
  const pos = Array.isArray(sample?.ligand_pos) ? sample.ligand_pos : []
  const nums = Array.isArray(sample?.ligand_atomic_nums) ? sample.ligand_atomic_nums : []
  const n = pos.length
  if (n === 0 || nums.length !== n) return null

  const traj = Array.isArray(sample?.trajectory) ? sample.trajectory : []
  const lastFrame = traj.length > 0 ? traj[traj.length - 1] : null
  const rawBonds = Array.isArray(lastFrame?.bonds) ? lastFrame.bonds : []
  const bonds = rawBonds
    .map((b) => [Number(b[0]), Number(b[1]), Number(b[2] || 1)])
    .filter(([a, b, o]) => Number.isFinite(a) && Number.isFinite(b) && o >= 1 && o <= 3)

  const p = (v, w) => String(v).padStart(w, ' ')
  const coord = (v) => Number(v).toFixed(4).padStart(10, ' ')

  const lines = [
    name || 'Ligand',
    '  BindHard           3D',
    '',
    `${p(n, 3)}${p(bonds.length, 3)}  0  0  0  0  0  0  0  0999 V2000`,
  ]

  for (let i = 0; i < n; i++) {
    const [x, y, z] = pos[i]
    const sym = ATOMIC_SYMBOLS[Number(nums[i])] || 'C'
    lines.push(`${coord(x)}${coord(y)}${coord(z)} ${sym.padEnd(3, ' ')} 0  0  0  0  0  0  0  0  0  0  0  0`)
  }

  for (const [a1, a2, bt] of bonds) {lines.push(`${p(a1 + 1, 3)}${p(a2 + 1, 3)}${p(Math.min(bt, 3), 3)}  0  0  0  0`)}

  lines.push('M  END')
  return lines.join('\n')
}

export function sampleToSdfBlock(sample, { proteinId = '' } = {}) {
  const idx = Number(sample?.sample_idx ?? 0)
  const mol = buildMolBlock(sample, `sample_${idx}`)
  if (!mol) return null

  const props = []
  if (proteinId) props.push(`>  <ProteinID>\n${proteinId}\n`)
  if (Number.isFinite(sample?.vina_score)) props.push(`>  <BindingAffinity>\n${sample.vina_score.toFixed(4)}\n`)
  if (sample?.smiles) props.push(`>  <SMILES>\n${sample.smiles}\n`)
  if (Number.isFinite(sample?.qed_score)) props.push(`>  <QED>\n${sample.qed_score.toFixed(4)}\n`)
  if (Number.isFinite(sample?.sa_score)) props.push(`>  <SA>\n${sample.sa_score.toFixed(4)}\n`)
  return mol + '\n' + props.join('') + '$$$$'
}

export function samplesToSdf(samples, opts = {}) {
  return samples
    .map((s) => sampleToSdfBlock(s, opts))
    .filter(Boolean)
    .join('\n')
}

function triggerDownload(content, filename) {
  const blob = new Blob([content], { type: 'chemical/x-mdl-sdfile' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function downloadSampleSdf(sample, { proteinId = '' } = {}) {
  const sdf = sampleToSdfBlock(sample, { proteinId })
  if (!sdf) return false
  triggerDownload(sdf, `ligand_sample${Number(sample?.sample_idx ?? 0)}.sdf`)
  return true
}

export function downloadBatchSdf(samples, { proteinId = '', filename = 'ligands_batch.sdf' } = {}) {
  const sdf = samplesToSdf(samples, { proteinId })
  if (!sdf) return false
  triggerDownload(sdf, filename)
  return true
}