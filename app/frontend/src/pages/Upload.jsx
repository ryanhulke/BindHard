import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { uploadTarget } from '../lib/trajectoryApi'

function parseVec3Input(value, label) {
  const parts = String(value)
    .split(',')
    .map((part) => Number(part.trim()))

  if (parts.length !== 3 || parts.some((part) => !Number.isFinite(part))) {
    throw new Error(`${label} must be 3 comma-separated numbers`)
  }

  return parts
}

export default function Upload() {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [stage, setStage] = useState('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const [samplesPerTarget, setSamplesPerTarget] = useState('8')
  const [returnTrajectory, setReturnTrajectory] = useState(true)
  const [trajectoryStride, setTrajectoryStride] = useState('10')
  const [boxCenterInput, setBoxCenterInput] = useState('0, 0, 0')
  const [boxSizeInput, setBoxSizeInput] = useState('24, 24, 24')

  const inputRef = useRef(null)
  const navigate = useNavigate()

  const isBusy = stage === 'uploading' || stage === 'done'

  const handleFile = (nextFile) => {
    if (!nextFile) return
    setFile(nextFile)
    setErrorMessage('')
    setStatusMessage('')
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (isBusy) return
    handleFile(event.dataTransfer.files[0])
  }

  const runModel = async () => {
    if (!file || isBusy) return

    setErrorMessage('')
    setStage('uploading')
    setStatusMessage('Uploading target and running generation...')

    try {
      const boxCenter = parseVec3Input(boxCenterInput, 'Box center')
      const boxSize = parseVec3Input(boxSizeInput, 'Box size')

      const payload = await uploadTarget(file, {
        samplesPerTarget: Number(samplesPerTarget),
        returnTrajectory,
        trajectoryStride: Number(trajectoryStride),
        boxCenter,
        boxSize,
      })

      const jobId = String(payload?.job_id || '')
      if (!jobId) {
        throw new Error('API did not return a job_id.')
      }

      setStage('done')
      setStatusMessage(`Generation complete. Redirecting to results for job ${jobId}...`)

      const search = new URLSearchParams({ job_id: jobId })
      window.setTimeout(() => navigate(`/dashboard/results?${search.toString()}`), 500)
    } catch (error) {
      setStage('idle')
      setStatusMessage('')
      setErrorMessage(String(error?.message || error))
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: '#080a0f' }}>
      <div className="w-full max-w-lg px-6">
        <div className="text-center mb-10">
          <p className="text-white/40 text-xs font-bold tracking-[0.3em] uppercase mb-3">GenBind</p>
          <h1 className="text-white text-4xl font-bold tracking-tight">Upload Target</h1>
          <p className="text-white/40 text-sm mt-2">PDB file + inference settings</p>
        </div>

        <div
          onClick={() => !isBusy && inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault()
            if (!isBusy) setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className="relative rounded-2xl border-2 border-dashed transition-all duration-200 p-12 flex flex-col items-center gap-4 text-center"
          style={{
            cursor: isBusy ? 'not-allowed' : 'pointer',
            borderColor: dragging ? '#196eff' : file ? '#22c55e' : 'rgba(255,255,255,0.12)',
            background: dragging ? 'rgba(25,110,255,0.05)' : 'rgba(255,255,255,0.02)',
            opacity: isBusy ? 0.8 : 1,
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdb"
            className="hidden"
            onChange={(event) => handleFile(event.target.files[0])}
            disabled={isBusy}
          />
          {file ? (
            <>
              <p className="text-white font-semibold">{file.name}</p>
              <p className="text-white/40 text-xs">
                {(file.size / 1024).toFixed(1)} KB | click to change
              </p>
            </>
          ) : (
            <>
              <p className="text-white/70 font-medium">Drop your file here</p>
              <p className="text-white/30 text-xs">or click to browse</p>
            </>
          )}
        </div>

        <div className="mt-6 grid grid-cols-1 gap-4">
          <label className="block">
            <span className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
              Samples
            </span>
            <input
              type="number"
              min="1"
              max="64"
              value={samplesPerTarget}
              onChange={(event) => setSamplesPerTarget(event.target.value)}
              disabled={isBusy}
              className="w-full rounded-xl border border-white/10 bg-[#070b14] px-3 py-3 text-sm text-white outline-none focus:border-blue-400/60"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
              Trajectory stride
            </span>
            <input
              type="number"
              min="1"
              max="100"
              value={trajectoryStride}
              onChange={(event) => setTrajectoryStride(event.target.value)}
              disabled={isBusy || !returnTrajectory}
              className="w-full rounded-xl border border-white/10 bg-[#070b14] px-3 py-3 text-sm text-white outline-none focus:border-blue-400/60 disabled:opacity-40"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
              Box center
            </span>
            <input
              type="text"
              value={boxCenterInput}
              onChange={(event) => setBoxCenterInput(event.target.value)}
              disabled={isBusy}
              placeholder="x, y, z"
              className="w-full rounded-xl border border-white/10 bg-[#070b14] px-3 py-3 text-sm text-white outline-none focus:border-blue-400/60"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
              Box size
            </span>
            <input
              type="text"
              value={boxSizeInput}
              onChange={(event) => setBoxSizeInput(event.target.value)}
              disabled={isBusy}
              placeholder="sx, sy, sz"
              className="w-full rounded-xl border border-white/10 bg-[#070b14] px-3 py-3 text-sm text-white outline-none focus:border-blue-400/60"
            />
          </label>

          <label className="flex items-center gap-3 rounded-xl border border-white/10 bg-[#070b14] px-3 py-3 text-sm text-white">
            <input
              type="checkbox"
              checked={returnTrajectory}
              onChange={(event) => setReturnTrajectory(event.target.checked)}
              disabled={isBusy}
            />
            <span>Return trajectory</span>
          </label>
        </div>

        <button
          onClick={runModel}
          disabled={!file || isBusy}
          className="mt-6 w-full py-4 rounded-2xl font-bold text-white text-base transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
          style={{
            background: file ? '#196eff' : 'rgba(255,255,255,0.08)',
            boxShadow: file ? '0 8px 24px rgba(25,110,255,0.3)' : 'none',
          }}
        >
          {stage === 'uploading' ? 'Generating...' : 'Generate Candidate Binders'}
        </button>

        {(stage === 'uploading' || stage === 'done') && (
          <div
            className="mt-4 rounded-2xl border border-white/10 p-4"
            style={{ background: 'rgba(255,255,255,0.03)' }}
          >
            <p className="text-white/70 text-sm text-center">{statusMessage}</p>
          </div>
        )}

        {errorMessage && (
          <div className="mt-4 rounded-lg border border-red-400/30 bg-red-900/25 px-3 py-2 text-xs text-red-100">
            {errorMessage}
          </div>
        )}
      </div>
    </div>
  )
}