import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useNavigate, useOutletContext, useSearchParams } from 'react-router-dom'

import ExportPanel from '../components/ExportPanel'
import MetricsPanel from '../components/MetricsPanel'
import TrajectoryViewer from '../components/TrajectoryViewer'
import { downloadSampleSdf } from '../lib/sdfExport'
import { fetchJob, fetchSamples, sortSamplesByVina } from '../lib/trajectoryApi'

function firstSelectableSample(samples) {
  return samples[0] ?? null
}

function getDefaultFrameIndex(sample) {
  const trajectory = Array.isArray(sample?.trajectory) ? sample.trajectory : []
  return trajectory.length > 0 ? trajectory.length - 1 : 0
}

function normalizeSaScoreToUnit(value) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) return null
  if (numericValue >= 0 && numericValue <= 1) return numericValue
  if (numericValue >= 1 && numericValue <= 10) return (numericValue - 1) / 9
  return null
}

function formatMetricValue(value, decimals, transform) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) return '-'
  const displayValue = transform ? transform(numericValue) : numericValue
  return Number.isFinite(displayValue) ? displayValue.toFixed(decimals) : '-'
}

export default function Results() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const outletContext = useOutletContext() || {}
  const viewerMode = outletContext.viewerMode || 'molstar'
  const isMetricsMode = viewerMode === 'metrics'
  const isMolstarMode = viewerMode === 'molstar'

  const initialJobId = searchParams.get('job_id') || ''

  const [jobId] = useState(initialJobId)
  const [job, setJob] = useState(null)
  const [samples, setSamples] = useState([])
  const [loadError, setLoadError] = useState('')
  const [viewerError, setViewerError] = useState('')
  const [selectedSampleIdx, setSelectedSampleIdx] = useState(null)

  const [isSampleMenuOpen, setIsSampleMenuOpen] = useState(false)
  const [sampleMenuStyle, setSampleMenuStyle] = useState(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [viewerFrameCount, setViewerFrameCount] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  const [selectedForExport, setSelectedForExport] = useState(new Set())
  const [exportFilters, setExportFilters] = useState({
    hasSmilesOnly: false,
    affinityMin: null,
    affinityMax: null,
  })

  const toggleExportSelect = useCallback((idx) => {
    setSelectedForExport((prev) => {
      const next = new Set(prev)
      next.has(idx) ? next.delete(idx) : next.add(idx)
      return next
    })
  }, [])

  const proteinId = job?.filename?.replace(/\.pdb$/i, '') || ''

  const playbackFps = 20
  const sampleMenuRef = useRef(null)
  const sampleMenuPopupRef = useRef(null)
  const sampleMenuTriggerRef = useRef(null)

  const loadData = useCallback(async () => {
    if (!jobId) {
      setLoadError('Missing job_id in URL.')
      setJob(null)
      setSamples([])
      setSelectedSampleIdx(null)
      return
    }

    setLoadError('')
    setViewerError('')

    try {
      const [jobRow, sampleRows] = await Promise.all([
        fetchJob(jobId),
        fetchSamples(jobId),
      ])

      if (!jobRow) {
        throw new Error(`Job '${jobId}' was not found.`)
      }

      const sortedSamples = sortSamplesByVina(sampleRows)
      const completedSamples = sortedSamples.filter((sample) => sample?.status === 'completed')
      const initialSample = firstSelectableSample(completedSamples)

      setJob(jobRow)
      setSamples(completedSamples)
      setSelectedSampleIdx(initialSample ? Number(initialSample.sample_idx) : null)
      setFrameIndex(getDefaultFrameIndex(initialSample))
      setViewerFrameCount(0)
      setIsPlaying(false)
    } catch (error) {
      setJob(null)
      setSamples([])
      setSelectedSampleIdx(null)
      setFrameIndex(0)
      setViewerFrameCount(0)
      setIsPlaying(false)
      setLoadError(String(error?.message || error))
    }
  }, [jobId])

  useEffect(() => {
    loadData()
  }, [loadData])

  useEffect(() => {
    if (!isSampleMenuOpen) return undefined

    const updateMenuPosition = () => {
      const trigger = sampleMenuTriggerRef.current
      if (!trigger) return
      const rect = trigger.getBoundingClientRect()
      const top = rect.bottom + 6
      const maxHeight = Math.max(120, Math.floor(window.innerHeight - top - 12))
      setSampleMenuStyle({
        position: 'fixed',
        left: `${Math.round(rect.left)}px`,
        top: `${Math.round(top)}px`,
        width: `${Math.round(rect.width)}px`,
        maxHeight: `${maxHeight}px`,
      })
    }

    updateMenuPosition()

    const handlePointerDown = (event) => {
      if (!sampleMenuRef.current) return
      const clickedInsideTrigger = sampleMenuRef.current.contains(event.target)
      const clickedInsidePopup = sampleMenuPopupRef.current?.contains(event.target)
      if (!clickedInsideTrigger && !clickedInsidePopup) {
        setIsSampleMenuOpen(false)
      }
    }

    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        setIsSampleMenuOpen(false)
      }
    }

    const handleWindowChange = () => {
      updateMenuPosition()
    }

    window.addEventListener('mousedown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('resize', handleWindowChange)
    window.addEventListener('scroll', handleWindowChange, true)

    return () => {
      window.removeEventListener('mousedown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('resize', handleWindowChange)
      window.removeEventListener('scroll', handleWindowChange, true)
    }
  }, [isSampleMenuOpen])

  const selectedSample = useMemo(
    () => samples.find((sample) => Number(sample?.sample_idx) === Number(selectedSampleIdx)) ?? null,
    [samples, selectedSampleIdx],
  )

  const selectedSampleTrajectory = Array.isArray(selectedSample?.trajectory)
    ? selectedSample.trajectory
    : []

  const frameCount = Math.max(selectedSampleTrajectory.length, Number(viewerFrameCount || 0))
  const maxFrame = Math.max(0, frameCount - 1)
  const canPlayback =
    isMolstarMode &&
    selectedSample?.status === 'completed' &&
    frameCount > 1

  useEffect(() => {
    setFrameIndex(getDefaultFrameIndex(selectedSample))
    setViewerFrameCount(0)
    setViewerError('')
    setIsPlaying(false)
    setIsSampleMenuOpen(false)
  }, [selectedSample])

  useEffect(() => {
    setViewerError('')
    setIsPlaying(false)
  }, [viewerMode])

  useEffect(() => {
    if (!canPlayback || !isPlaying) return undefined
    const intervalMs = Math.round(1000 / playbackFps)
    const timer = window.setInterval(() => {
      setFrameIndex((prev) => {
        if (prev >= maxFrame) {
          setIsPlaying(false)
          return maxFrame
        }

        const next = prev + 1
        if (next >= maxFrame) {
          setIsPlaying(false)
          return maxFrame
        }
        return next
      })
    }, intervalMs)

    return () => window.clearInterval(timer)
  }, [canPlayback, isPlaying, maxFrame, playbackFps])

  useEffect(() => {
    if (!canPlayback && isPlaying) {
      setIsPlaying(false)
    }
  }, [canPlayback, isPlaying])

  const handleFrameApplied = useCallback((appliedFrame) => {
    setFrameIndex((prev) => (prev === appliedFrame ? prev : appliedFrame))
  }, [])

  const handleViewerLoadComplete = useCallback(({ frameCount: loadedFrameCount }) => {
    setViewerError('')
    setViewerFrameCount(Number(loadedFrameCount || 0))
  }, [])

  const handleViewerLoadError = useCallback((error) => {
    setViewerError(String(error?.message || error))
  }, [])

  const sampleLabel = selectedSample
    ? `Sample ${Number(selectedSample.sample_idx)}`
    : '-'

  const displayViewerError = selectedSample?.status !== 'completed'
    ? String(selectedSample?.error || 'Sample generation failed.')
    : viewerError

  const summary = job?.summary || null
  const proteinPdbText = String(job?.pdb_text || '')

  return (
    <div className="h-[calc(100vh-3.5rem)] overflow-hidden px-5 py-6" style={{ background: '#080a0f' }}>
      <div className="mx-auto flex h-full w-full max-w-[1660px] gap-5">
        <aside
          className="w-[400px] max-w-[40vw] shrink-0 overflow-y-auto rounded-2xl border border-white/10 p-4"
          style={{ background: 'rgba(255,255,255,0.025)' }}
        >
          <h1 className="text-xl font-semibold tracking-tight text-white">Results</h1>

          <div className="mt-5 space-y-3">
            <div>
              <span className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
                File
              </span>
              <div className="rounded-lg border border-white/10 bg-[#070b14] px-3 py-2 text-sm text-white">
                {job?.filename || '-'}
              </div>
            </div>

            {summary && (
              <div className="rounded-xl border border-white/10 bg-black/20 p-3 text-xs text-white/75 space-y-1">
                <div>Samples: {summary.n_samples ?? '-'}</div>
                <div>Valid: {summary.n_valid ?? '-'}</div>
                <div>Invalid: {summary.n_invalid ?? '-'}</div>
                <div>Mean Vina: {formatMetricValue(summary.vina_mean, 2)}</div>
                <div>Mean QED: {formatMetricValue(summary.qed_mean, 3)}</div>
                <div>Mean SA: {formatMetricValue(summary.sa_mean, 3, normalizeSaScoreToUnit)}</div>
              </div>
            )}
          </div>

          {loadError && (
            <div className="mt-4 rounded-lg border border-red-400/30 bg-red-900/25 px-3 py-2 text-xs text-red-100">
              {loadError}
            </div>
          )}

          <div className="mt-5">
            <label className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
              Sample
            </label>
            <div className="relative" ref={sampleMenuRef}>
              <button
                ref={sampleMenuTriggerRef}
                type="button"
                onClick={() => {
                  if (samples.length === 0) return
                  setIsSampleMenuOpen((prev) => !prev)
                }}
                className="flex w-full items-center justify-between rounded-lg border border-white/10 bg-[#070b14] px-3 py-2 text-left text-sm text-white outline-none transition-colors hover:border-white/25 hover:bg-[#0d1320] focus:border-blue-400/60"
                disabled={samples.length === 0}
                aria-haspopup="listbox"
                aria-expanded={isSampleMenuOpen}
              >
                <span>{sampleLabel}</span>
                <span className="text-xs text-white/60">{isSampleMenuOpen ? '^' : 'v'}</span>
              </button>

              {isSampleMenuOpen && samples.length > 0 && sampleMenuStyle && createPortal(
                <div
                  ref={sampleMenuPopupRef}
                  className="z-[90] overflow-y-auto rounded-lg border border-white/15 bg-[#0b111d] shadow-[0_18px_38px_rgba(0,0,0,0.45)]"
                  style={sampleMenuStyle}
                >
                  <ul className="py-1" role="listbox" aria-label="Sample">
                    {samples.map((sample) => {
                      const sampleIdx = Number(sample.sample_idx)
                      const isSelected = sampleIdx === Number(selectedSampleIdx)
                      const vinaText =
                        typeof sample.vina_score === 'number'
                          ? `vina ${sample.vina_score.toFixed(2)}`
                          : null

                      return (
                        <li key={sample.sample_idx} className="flex items-center">
                          <label className="flex items-center pl-3 cursor-pointer" onClick={(e) => e.stopPropagation()}>
                            <input type="checkbox" checked={selectedForExport.has(sampleIdx)}
                              onChange={() => toggleExportSelect(sampleIdx)} className="accent-blue-500" />
                          </label>
                          <button
                            type="button"
                            onClick={() => {
                              setSelectedSampleIdx(sampleIdx)
                              setIsSampleMenuOpen(false)
                            }}
                            className={`flex-1 px-3 py-2 text-left text-sm transition-colors ${
                              isSelected
                                ? 'bg-blue-500/25 text-blue-100'
                                : 'text-white/85 hover:bg-white/10 hover:text-white'
                            }`}
                            role="option"
                            aria-selected={isSelected}
                          >
                            <div className="flex items-center justify-between">
                              <span>Sample {sampleIdx}</span>
                              <button
                                type="button"
                                title="Download .sdf"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  downloadSampleSdf(sample, { proteinId })
                                }}
                                className="ml-2 rounded px-1.5 py-0.5 text-[10px] font-semibold text-white/40 hover:bg-white/10 hover:text-white/80 transition-colors"
                              >
                                .sdf
                              </button>
                            </div>
                            {vinaText && <div className="text-[11px] opacity-70">{vinaText}</div>}
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </div>,
                document.body,
              )}
            </div>
          </div>

          {isMolstarMode && (
            <div className="mt-5 rounded-xl border border-white/10 bg-black/20 p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-widest text-white/40">Playback</span>
                <button
                  type="button"
                  onClick={() => {
                    if (!canPlayback) return
                    if (isPlaying) {
                      setIsPlaying(false)
                      return
                    }
                    if (frameIndex >= maxFrame && maxFrame > 0) {
                      setFrameIndex(0)
                    }
                    setIsPlaying(true)
                  }}
                  className="rounded-md border border-white/20 px-2 py-1 text-xs font-semibold text-white/80 hover:border-white/40 hover:text-white"
                  disabled={!canPlayback}
                >
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
              </div>

              <div className="mt-3">
                <input
                  type="range"
                  min={0}
                  max={maxFrame}
                  value={Math.min(frameIndex, maxFrame)}
                  onChange={(event) => {
                    setIsPlaying(false)
                    setFrameIndex(Number(event.target.value))
                  }}
                  className="w-full accent-blue-500"
                  disabled={!canPlayback}
                />
              </div>
            </div>
          )}

          {selectedSample?.status === 'completed' && (
            <div className="mt-5">
              <button type="button" onClick={() => downloadSampleSdf(selectedSample, { proteinId })}
                className="w-full rounded-lg border border-white/10 px-3 py-2 text-xs font-semibold text-white/60 hover:border-white/25 hover:text-white transition-colors">
                Save This Molecule (.sdf)
              </button>
            </div>
          )}

          {samples.length > 0 && (
            <div className="mt-5">
              <ExportPanel
                samples={samples}
                selected={selectedForExport}
                onSelectAll={(ids) => setSelectedForExport(new Set(ids))}
                onDeselectAll={() => setSelectedForExport(new Set())}
                proteinId={proteinId}
                filters={exportFilters}
                onFiltersChange={setExportFilters}
              />
            </div>
          )}

          <div className="mt-5 text-[11px] text-white/35">
            {isMolstarMode
              ? 'Controls: left-drag rotate, right-drag pan, wheel zoom.'
              : 'Compare this molecule against average baseline metrics.'}
          </div>

          <button
            onClick={() => navigate('/dashboard')}
            className="mt-6 w-full rounded-lg border border-white/10 px-3 py-2 text-sm font-semibold text-white/70 hover:border-white/25 hover:text-white"
          >
            Back to Upload
          </button>
        </aside>

        <section className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1">
            {selectedSample ? (
              isMetricsMode ? (
                <MetricsPanel entry={selectedSample} />
              ) : selectedSample.status === 'completed' && proteinPdbText ? (
                <TrajectoryViewer
                  proteinPdbText={proteinPdbText}
                  sample={selectedSample}
                  frameIndex={frameIndex}
                  onFrameApplied={handleFrameApplied}
                  onLoadComplete={handleViewerLoadComplete}
                  onLoadError={handleViewerLoadError}
                />
              ) : (
                <div className="flex h-full items-center justify-center rounded-2xl border border-white/10 bg-white/[0.02]">
                  <p className="text-sm text-white/45">
                    {selectedSample.error || 'This sample has no renderable structure.'}
                  </p>
                </div>
              )
            ) : (
              <div className="flex h-full items-center justify-center rounded-2xl border border-white/10 bg-white/[0.02]">
                <p className="text-sm text-white/45">
                  Choose a sample to view results.
                </p>
              </div>
            )}
          </div>

          {displayViewerError && (
            <div className="mt-2 rounded-lg border border-red-400/30 bg-red-900/25 px-3 py-2 text-xs text-red-100">
              {displayViewerError}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
