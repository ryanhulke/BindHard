function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function parseInteger(value, fieldName, { min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER } = {}) {
  if (!Number.isInteger(value)) {
    throw new Error(`Invalid sidecar: '${fieldName}' must be an integer`)
  }
  if (value < min || value > max) {
    throw new Error(`Invalid sidecar: '${fieldName}' must be in [${min}, ${max}]`)
  }
  return Number(value)
}

function parseBondTuple(value, frameIdx, bondIdx, nLigandAtoms) {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`Invalid sidecar: frame ${frameIdx} bond ${bondIdx} must be [i, j, order]`)
  }

  const i = parseInteger(value[0], `frames[${frameIdx}].bonds[${bondIdx}][0]`, { min: 0, max: nLigandAtoms - 1 })
  const j = parseInteger(value[1], `frames[${frameIdx}].bonds[${bondIdx}][1]`, { min: 0, max: nLigandAtoms - 1 })
  if (i === j) {
    throw new Error(`Invalid sidecar: frame ${frameIdx} bond ${bondIdx} has identical atom indices`)
  }

  const order = parseInteger(value[2], `frames[${frameIdx}].bonds[${bondIdx}][2]`, { min: 1, max: 4 })
  return [i, j, order]
}

export function validateDynamicBondSidecar(payload) {
  if (!isPlainObject(payload)) {
    throw new Error('Invalid sidecar: payload must be an object')
  }

  const requiredKeys = ['n_frames', 'n_ligand_atoms', 'index_base', 'frames']
  for (const key of requiredKeys) {
    if (!(key in payload)) {
      throw new Error(`Invalid sidecar: missing required key '${key}'`)
    }
  }

  const nFrames = parseInteger(payload.n_frames, 'n_frames', { min: 1 })
  const nLigandAtoms = parseInteger(payload.n_ligand_atoms, 'n_ligand_atoms', { min: 1 })
  const indexBase = parseInteger(payload.index_base, 'index_base', { min: 0, max: 0 })

  if (!Array.isArray(payload.frames)) {
    throw new Error("Invalid sidecar: 'frames' must be an array")
  }
  if (payload.frames.length !== nFrames) {
    throw new Error(
      `Invalid sidecar: frames length (${payload.frames.length}) does not match n_frames (${nFrames})`,
    )
  }

  const normalizedFrames = payload.frames.map((frame, frameArrayIdx) => {
    if (!isPlainObject(frame)) {
      throw new Error(`Invalid sidecar: frames[${frameArrayIdx}] must be an object`)
    }
    if (!('frame_idx' in frame) || !('bonds' in frame)) {
      throw new Error(`Invalid sidecar: frames[${frameArrayIdx}] must contain 'frame_idx' and 'bonds'`)
    }

    const frameIdx = parseInteger(frame.frame_idx, `frames[${frameArrayIdx}].frame_idx`, { min: 0, max: nFrames - 1 })
    if (!Array.isArray(frame.bonds)) {
      throw new Error(`Invalid sidecar: frames[${frameArrayIdx}].bonds must be an array`)
    }

    const bonds = frame.bonds.map((bond, bondIdx) => parseBondTuple(bond, frameIdx, bondIdx, nLigandAtoms))
    return { frame_idx: frameIdx, bonds }
  })

  return {
    ...payload,
    n_frames: nFrames,
    n_ligand_atoms: nLigandAtoms,
    index_base: indexBase,
    frames: normalizedFrames,
  }
}
