import { useCallback, useEffect, useRef, useState } from 'react'
import { Viewer } from 'molstar/lib/apps/viewer/app.js'
import { CustomInteractions, InteractionsShape } from 'molstar/lib/extensions/interactions/transforms.js'
import { Asset } from 'molstar/lib/mol-util/assets.js'
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder.js'
import { PluginCommands } from 'molstar/lib/mol-plugin/commands.js'
import { StateSelection } from 'molstar/lib/mol-state/index.js'
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms.js'
import { StructureElement, StructureProperties } from 'molstar/lib/mol-model/structure.js'

import 'molstar/build/viewer/molstar.css'
import 'molstar/build/viewer/theme/dark.css'

const SIDECHAIN_BOND_SOURCE_TAG = 'sidecar-bonds-source'
const SIDECHAIN_BOND_SHAPE_TAG = 'sidecar-bonds-shape'
const SIDECHAIN_BOND_REPR_TAG = 'sidecar-bonds-repr'

const ELEMENT_SYMBOLS = {
  1: 'H',
  5: 'B',
  6: 'C',
  7: 'N',
  8: 'O',
  9: 'F',
  11: 'Na',
  12: 'Mg',
  15: 'P',
  16: 'S',
  17: 'Cl',
  19: 'K',
  20: 'Ca',
  25: 'Mn',
  26: 'Fe',
  27: 'Co',
  28: 'Ni',
  29: 'Cu',
  30: 'Zn',
  34: 'Se',
  35: 'Br',
  53: 'I',
}

function residueCompIdQuery(compId) {
  return MS.struct.generator.atomGroups({
    'residue-test': MS.core.rel.eq([MS.struct.atomProperty.macromolecular.label_comp_id(), compId]),
  })
}

function bondOrderToDegree(order) {
  if (order === 4) return 'aromatic'
  return order
}

function collectLigandSourceIndices(ligandStructure) {
  if (!ligandStructure?.units?.length) return []

  const sourceIndices = []
  const location = StructureElement.Location.create(ligandStructure)

  for (const unit of ligandStructure.units) {
    location.unit = unit
    for (const element of unit.elements) {
      location.element = element
      sourceIndices.push(Number(StructureProperties.atom.sourceIndex(location)))
    }
  }

  return Array.from(new Set(sourceIndices)).sort((a, b) => a - b)
}

function inferElementFromAtomName(atomName) {
  const letters = String(atomName ?? '')
    .replace(/[^A-Za-z]/g, '')
    .trim()

  if (!letters) return 'C'

  if (letters.length >= 2) {
    const two = `${letters[0].toUpperCase()}${letters[1].toLowerCase()}`
    if (Object.values(ELEMENT_SYMBOLS).includes(two)) return two
  }

  return letters[0].toUpperCase()
}

function atomicNumToElement(atomicNum) {
  return ELEMENT_SYMBOLS[Number(atomicNum)] || 'C'
}

function parseProteinAtoms(pdbText) {
  const atoms = []
  const lines = String(pdbText ?? '').split(/\r?\n/)
  let sawModel = false

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]
    const record = line.slice(0, 6).trim().toUpperCase()

    if (record === 'MODEL') {
      sawModel = true
      continue
    }
    if (record === 'ENDMDL' && sawModel) {
      break
    }
    if (record !== 'ATOM' && record !== 'HETATM') {
      continue
    }
    if (line.length < 54) {
      continue
    }

    const atomName = line.slice(12, 16).trim() || 'X'
    const resName = line.slice(17, 20).trim() || 'UNK'
    const chainId = line.slice(21, 22).trim() || 'A'
    const residueId = Number.parseInt(line.slice(22, 26).trim(), 10)
    const x = Number.parseFloat(line.slice(30, 38).trim())
    const y = Number.parseFloat(line.slice(38, 46).trim())
    const z = Number.parseFloat(line.slice(46, 54).trim())
    const elementField = line.length >= 78 ? line.slice(76, 78).trim() : ''
    const element = elementField || inferElementFromAtomName(atomName)

    if (![x, y, z].every(Number.isFinite)) {
      continue
    }

    atoms.push({
      atomName,
      resName,
      chainId,
      residueId: Number.isFinite(residueId) ? residueId : 1,
      x,
      y,
      z,
      element,
    })
  }

  return atoms
}

function formatPdbAtomLine({
  recordName,
  serial,
  atomName,
  resName,
  chainId,
  residueId,
  x,
  y,
  z,
  element,
}) {
  const normalizedRecord = String(recordName || 'ATOM').toUpperCase().padEnd(6, ' ')
  const normalizedAtom = String(atomName || 'X').slice(0, 4).padStart(4, ' ')
  const normalizedRes = String(resName || 'UNK').slice(0, 3).padStart(3, ' ')
  const normalizedChain = String(chainId || 'A').slice(0, 1) || 'A'
  const normalizedResidue = Number.isFinite(Number(residueId)) ? Number(residueId) : 1
  const normalizedElement = String(element || 'C').slice(0, 2).padStart(2, ' ')

  return (
    `${normalizedRecord}` +
    `${String(serial).padStart(5, ' ')} ` +
    `${normalizedAtom}` +
    ` ` +
    `${normalizedRes} ` +
    `${normalizedChain}` +
    `${String(normalizedResidue).padStart(4, ' ')}` +
    `    ` +
    `${Number(x).toFixed(3).padStart(8, ' ')}` +
    `${Number(y).toFixed(3).padStart(8, ' ')}` +
    `${Number(z).toFixed(3).padStart(8, ' ')}` +
    `${'1.00'.padStart(6, ' ')}` +
    `${'0.00'.padStart(6, ' ')}` +
    `          ` +
    `${normalizedElement}\n`
  )
}

function ligandAtomNameFromElement(element, atomIndex) {
  const symbol = String(element || 'C').replace(/[^A-Za-z]/g, '').slice(0, 2) || 'C'
  const suffix = String((atomIndex + 1) % 100).padStart(2, '0')
  return `${symbol}${suffix}`.slice(0, 4)
}

function buildRenderableFrames(sample) {
  const trajectory = Array.isArray(sample?.trajectory) ? sample.trajectory : []
  if (trajectory.length > 0) return trajectory

  if (!Array.isArray(sample?.ligand_pos) || !Array.isArray(sample?.ligand_type)) {
    return []
  }

  return [
    {
      t: 0,
      ligand_pos: sample.ligand_pos,
      ligand_type: sample.ligand_type,
      ligand_atomic_nums: Array.isArray(sample?.ligand_atomic_nums) ? sample.ligand_atomic_nums : [],
      bonds: [],
    },
  ]
}

function buildTrajectoryPdbText(proteinPdbText, sample) {
  const proteinAtoms = parseProteinAtoms(proteinPdbText)
  const frames = buildRenderableFrames(sample)

  if (proteinAtoms.length === 0) {
    throw new Error('No protein atoms were found in the stored PDB text.')
  }
  if (frames.length === 0) {
    throw new Error('No ligand frames are available for this sample.')
  }

  const ligandAtomicNums =
    Array.isArray(sample?.ligand_atomic_nums) && sample.ligand_atomic_nums.length > 0
      ? sample.ligand_atomic_nums
      : Array.isArray(frames[0]?.ligand_atomic_nums)
        ? frames[0].ligand_atomic_nums
        : null

  if (!Array.isArray(ligandAtomicNums) || ligandAtomicNums.length !== frames[0].ligand_pos.length) {
    throw new Error('Ligand atomic numbers are missing or do not match the ligand atom count.')
  }

  let output = ''

  for (let frameIdx = 0; frameIdx < frames.length; frameIdx += 1) {
    const frame = frames[frameIdx]
    const ligandPos = Array.isArray(frame?.ligand_pos) ? frame.ligand_pos : []

    if (ligandPos.length !== ligandAtomicNums.length) {
      throw new Error(`Ligand atom count changed at frame ${frameIdx}.`)
    }

    output += `MODEL     ${String(frameIdx + 1).padStart(4, ' ')}\n`

    let serial = 1

    for (const atom of proteinAtoms) {
      output += formatPdbAtomLine({
        recordName: 'ATOM',
        serial,
        atomName: atom.atomName,
        resName: atom.resName,
        chainId: atom.chainId,
        residueId: atom.residueId,
        x: atom.x,
        y: atom.y,
        z: atom.z,
        element: atom.element,
      })
      serial += 1
    }

    for (let atomIdx = 0; atomIdx < ligandPos.length; atomIdx += 1) {
      const xyz = ligandPos[atomIdx]
      const element = atomicNumToElement(ligandAtomicNums[atomIdx])

      output += formatPdbAtomLine({
        recordName: 'HETATM',
        serial,
        atomName: ligandAtomNameFromElement(element, atomIdx),
        resName: 'LIG',
        chainId: 'Z',
        residueId: 1,
        x: Number(xyz[0]),
        y: Number(xyz[1]),
        z: Number(xyz[2]),
        element,
      })
      serial += 1
    }

    output += 'ENDMDL\n'
  }

  output += 'END\n'
  return output
}

export default function TrajectoryViewer({
  proteinPdbText,
  sample,
  frameIndex = 0,
  onFrameApplied,
  onLoadComplete,
  onLoadError,
}) {
  const hostRef = useRef(null)
  const viewerRef = useRef(null)
  const viewerInitPromiseRef = useRef(null)
  const modelRefsRef = useRef([])
  const frameCountRef = useRef(0)
  const pendingFrameRef = useRef(null)
  const isApplyingFrameRef = useRef(false)
  const lastFrameRef = useRef(-1)
  const lastBondFrameRef = useRef(-1)
  const loadTokenRef = useRef(0)
  const ligandSourceIndexByLocalRef = useRef([])
  const ligandStructureRefForBondsRef = useRef('')
  const bondFramesByIndexRef = useRef(new Map())
  const activeBlobUrlRef = useRef('')
  const sampleRef = useRef(sample)
  const onFrameAppliedRef = useRef(onFrameApplied)
  const onLoadCompleteRef = useRef(onLoadComplete)
  const onLoadErrorRef = useRef(onLoadError)

  const [isViewerReady, setIsViewerReady] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')

  useEffect(() => {
    sampleRef.current = sample
  }, [sample])

  useEffect(() => {
    onFrameAppliedRef.current = onFrameApplied
  }, [onFrameApplied])

  useEffect(() => {
    onLoadCompleteRef.current = onLoadComplete
  }, [onLoadComplete])

  useEffect(() => {
    onLoadErrorRef.current = onLoadError
  }, [onLoadError])

  const setBondFrame = useCallback(async (requestedFrameIndex) => {
    const viewer = viewerRef.current
    if (!viewer) return

    const currentSample = sampleRef.current
    if (!currentSample) {
      throw new Error('Sample data is required for bond rendering.')
    }

    if (requestedFrameIndex === lastBondFrameRef.current) return

    const frame = bondFramesByIndexRef.current.get(requestedFrameIndex)
    if (!frame) {
      throw new Error(`No bond frame found for frame index ${requestedFrameIndex}.`)
    }

    const sourceIndexByLocal = ligandSourceIndexByLocalRef.current
    const ligandRef = ligandStructureRefForBondsRef.current
    if (!ligandRef) {
      throw new Error('Ligand sidecar reference is not initialized.')
    }

    const interactions = (Array.isArray(frame.bonds) ? frame.bonds : []).map((bondTuple) => {
      const atomA = sourceIndexByLocal[Number(bondTuple[0])]
      const atomB = sourceIndexByLocal[Number(bondTuple[1])]
      const order = Number(bondTuple[2])

      return {
        kind: 'covalent',
        degree: bondOrderToDegree(order),
        aStructureRef: ligandRef,
        a: { atom_index: atomA },
        bStructureRef: ligandRef,
        b: { atom_index: atomB },
      }
    })

    const state = viewer.plugin.state.data
    const sourceNodes = state.select(
      StateSelection.Generators.ofTransformer(CustomInteractions).withTag(SIDECHAIN_BOND_SOURCE_TAG),
    )

    if (sourceNodes.length !== 1) {
      throw new Error('Unable to locate dynamic bond source in viewer state.')
    }

    const tree = state.build()
    tree.to(sourceNodes[0]).update((oldParams) => ({ ...oldParams, interactions }))

    await PluginCommands.State.Update(viewer.plugin, {
      state,
      tree,
      options: { doNotLogTiming: true },
    })

    lastBondFrameRef.current = requestedFrameIndex
  }, [])

  const setFrame = useCallback(
    async (requestedFrameIndex) => {
      const viewer = viewerRef.current
      if (!viewer) return

      const frameCount = Number(frameCountRef.current || 0)
      const clamped = frameCount > 0
        ? Math.max(0, Math.min(frameCount - 1, Number(requestedFrameIndex || 0)))
        : 0

      if (clamped === lastFrameRef.current) return

      const state = viewer.plugin.state.data
      const modelRefs = modelRefsRef.current.length > 0
        ? modelRefsRef.current
        : state.select(StateSelection.Generators.ofTransformer(StateTransforms.Model.ModelFromTrajectory))

      if (modelRefs.length === 0) return

      const tree = state.build()
      for (const modelRef of modelRefs) {
        tree.to(modelRef).update((oldParams) => ({ ...oldParams, modelIndex: clamped }))
      }

      await PluginCommands.State.Update(viewer.plugin, {
        state,
        tree,
        options: { doNotLogTiming: true },
      })

      await setBondFrame(clamped)

      lastFrameRef.current = clamped
      onFrameAppliedRef.current?.(clamped)
    },
    [setBondFrame],
  )

  const applyLatestFrame = useCallback(async () => {
    if (isApplyingFrameRef.current) return
    isApplyingFrameRef.current = true

    try {
      while (pendingFrameRef.current !== null) {
        const nextFrame = pendingFrameRef.current
        pendingFrameRef.current = null
        await setFrame(nextFrame)
      }
    } catch (error) {
      setErrorMessage(String(error?.message || error))
    } finally {
      isApplyingFrameRef.current = false
      if (pendingFrameRef.current !== null) {
        void applyLatestFrame()
      }
    }
  }, [setFrame])

  useEffect(() => {
    let disposed = false
    let preventContextMenuHandler = null
    const hostElement = hostRef.current

    async function initializeViewer() {
      if (!hostElement || viewerRef.current || disposed) return

      if (viewerInitPromiseRef.current) {
        await viewerInitPromiseRef.current
        if (viewerRef.current || disposed) return
      }

      viewerInitPromiseRef.current = (async () => {
        hostElement.replaceChildren()

        const viewer = await Viewer.create(hostElement, {
          layoutIsExpanded: false,
          layoutShowControls: false,
          layoutShowSequence: false,
          layoutShowLog: false,
          layoutShowLeftPanel: false,
          collapseLeftPanel: true,
          collapseRightPanel: true,
          viewportBackgroundColor: '#0b0f18',
          viewportShowControls: true,
          viewportShowAnimation: false,
          viewportShowTrajectoryControls: false,
          viewportShowSelectionMode: false,
          viewportShowExpand: false,
          viewportShowSettings: false,
          viewportShowToggleFullscreen: false,
          viewportShowScreenshotControls: false,
        })

        if (disposed) {
          viewer.dispose()
          return
        }

        viewerRef.current = viewer
        preventContextMenuHandler = (event) => event.preventDefault()
        hostElement.addEventListener('contextmenu', preventContextMenuHandler)
        setIsViewerReady(true)
      })()

      try {
        await viewerInitPromiseRef.current
      } finally {
        viewerInitPromiseRef.current = null
      }
    }

    initializeViewer().catch((error) => {
      setErrorMessage(String(error?.message || error))
      onLoadErrorRef.current?.(error)
    })

    return () => {
      disposed = true
      if (preventContextMenuHandler && hostElement) {
        hostElement.removeEventListener('contextmenu', preventContextMenuHandler)
      }
      if (viewerRef.current) {
        viewerRef.current.dispose()
        viewerRef.current = null
      }
      if (activeBlobUrlRef.current) {
        URL.revokeObjectURL(activeBlobUrlRef.current)
        activeBlobUrlRef.current = ''
      }
      if (hostElement) {
        hostElement.replaceChildren()
      }
      setIsViewerReady(false)
    }
  }, [])

  useEffect(() => {
    if (!proteinPdbText || !sample || !isViewerReady) return

    let cancelled = false
    const loadToken = loadTokenRef.current + 1
    loadTokenRef.current = loadToken

    async function loadStructure() {
      const viewer = viewerRef.current
      if (!viewer) return

      setIsLoading(true)
      setErrorMessage('')
      modelRefsRef.current = []
      frameCountRef.current = 0
      pendingFrameRef.current = null
      lastFrameRef.current = -1
      lastBondFrameRef.current = -1
      ligandSourceIndexByLocalRef.current = []
      ligandStructureRefForBondsRef.current = ''
      bondFramesByIndexRef.current = new Map()

      try {
        const frames = buildRenderableFrames(sample)
        if (frames.length === 0) {
          throw new Error('Sample does not contain any renderable frames.')
        }

        for (let i = 0; i < frames.length; i += 1) {
          if (bondFramesByIndexRef.current.has(i)) {
            throw new Error(`Duplicate frame index detected: ${i}`)
          }
          bondFramesByIndexRef.current.set(i, frames[i])
        }

        if (activeBlobUrlRef.current) {
          URL.revokeObjectURL(activeBlobUrlRef.current)
          activeBlobUrlRef.current = ''
        }

        const pdbText = buildTrajectoryPdbText(proteinPdbText, sample)
        const blob = new Blob([pdbText], { type: 'chemical/x-pdb' })
        const blobUrl = URL.createObjectURL(blob)
        activeBlobUrlRef.current = blobUrl

        await viewer.plugin.managers.animation.stop()
        await viewer.plugin.clear()

        const data = await viewer.plugin.builders.data.download(
          { url: Asset.Url(blobUrl), isBinary: false },
          { state: { isGhost: true } },
        )
        const trajectory = await viewer.plugin.builders.structure.parseTrajectory(data, 'pdb')
        const model = await viewer.plugin.builders.structure.createModel(trajectory, { modelIndex: 0 })
        const structure = await viewer.plugin.builders.structure.createStructure(
          model,
          { name: 'model', params: { dynamicBonds: false } },
          { isCollapsed: true },
        )

        let protein = await viewer.plugin.builders.structure.tryCreateComponentStatic(structure, 'protein', {
          label: 'Protein',
        })
        if (!protein) {
          protein = await viewer.plugin.builders.structure.tryCreateComponentStatic(structure, 'polymer', {
            label: 'Polymer',
          })
        }

        let ligand = await viewer.plugin.builders.structure.tryCreateComponentFromExpression(
          structure,
          residueCompIdQuery('LIG'),
          'ligand-resname-lig',
          { label: 'Ligand (LIG)' },
        )
        if (!ligand) {
          ligand = await viewer.plugin.builders.structure.tryCreateComponentStatic(structure, 'ligand', {
            label: 'Ligand',
          })
        }
        if (!ligand) {
          throw new Error('Unable to resolve ligand component from trajectory structure.')
        }

        if (protein) {
          await viewer.plugin.builders.structure.representation.addRepresentation(protein, {
            type: 'molecular-surface',
            color: 'chain-id',
          })
        }

        await viewer.plugin.builders.structure.representation.addRepresentation(ligand, {
          type: 'ball-and-stick',
          typeParams: {
            visuals: ['element-sphere'],
            ignoreHydrogens: true,
            ignoreHydrogensVariant: 'all',
          },
          color: 'element-symbol',
          colorParams: { carbonColor: { name: 'element-symbol', params: {} } },
        })

        const ligandSourceIndices = collectLigandSourceIndices(ligand.data)
        const expectedLigandAtoms = Array.isArray(frames[0]?.ligand_pos) ? frames[0].ligand_pos.length : 0
        if (ligandSourceIndices.length !== expectedLigandAtoms) {
          throw new Error(
            `Ligand atom mismatch: structure=${ligandSourceIndices.length}, sample=${expectedLigandAtoms}`,
          )
        }
        ligandSourceIndexByLocalRef.current = ligandSourceIndices

        const loadedFrameCount = Number(trajectory.cell?.obj?.data?.frameCount || 0)
        frameCountRef.current = loadedFrameCount
        if (loadedFrameCount !== frames.length) {
          throw new Error(`Trajectory frame mismatch: loaded=${loadedFrameCount}, sample=${frames.length}`)
        }

        const ligandRef = String(ligand?.ref || ligand?.cell?.transform?.ref || '')
        if (!ligandRef) {
          throw new Error('Unable to determine ligand state ref for dynamic bond rendering.')
        }
        ligandStructureRefForBondsRef.current = ligandRef

        const state = viewer.plugin.state.data
        const dynamicBondTree = state.build()
        const bondSource = dynamicBondTree.toRoot().apply(
          CustomInteractions,
          { interactions: [] },
          {
            dependsOn: [ligandRef],
            tags: [SIDECHAIN_BOND_SOURCE_TAG],
          },
        )

        bondSource
          .apply(InteractionsShape, { kinds: ['covalent'] }, { tags: [SIDECHAIN_BOND_SHAPE_TAG] })
          .apply(StateTransforms.Representation.ShapeRepresentation3D, {}, { tags: [SIDECHAIN_BOND_REPR_TAG] })

        await PluginCommands.State.Update(viewer.plugin, {
          state,
          tree: dynamicBondTree,
          options: { doNotLogTiming: true },
        })

        const modelRefs = state.select(StateSelection.Generators.ofTransformer(StateTransforms.Model.ModelFromTrajectory))
        modelRefsRef.current = modelRefs
        await PluginCommands.Camera.Reset(viewer.plugin, {})

        if (cancelled || loadToken !== loadTokenRef.current) return

        await setFrame(0)
        onLoadCompleteRef.current?.({ frameCount: loadedFrameCount })
      } catch (error) {
        if (cancelled) return
        const message = String(error?.message || error)
        setErrorMessage(message)
        onLoadErrorRef.current?.(error)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    loadStructure()

    return () => {
      cancelled = true
    }
  }, [isViewerReady, proteinPdbText, sample, setFrame])

  useEffect(() => {
    if (!isViewerReady) return
    pendingFrameRef.current = Number(frameIndex || 0)
    void applyLatestFrame()
  }, [applyLatestFrame, frameIndex, isViewerReady])

  return (
    <div className="relative h-full w-full overflow-hidden rounded-2xl border border-white/10 bg-[#0b0f18]">
      <div ref={hostRef} className="h-full w-full" />

      {isLoading && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/35">
          <div className="rounded-lg border border-white/15 bg-black/55 px-4 py-2 text-xs text-white/80">
            Loading trajectory...
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="absolute inset-x-4 top-4 rounded-lg border border-red-400/30 bg-red-900/35 px-3 py-2 text-xs text-red-100">
          {errorMessage}
        </div>
      )}
    </div>
  )
}