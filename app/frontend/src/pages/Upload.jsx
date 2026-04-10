import { useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'

import { uploadTarget } from '../lib/trajectoryApi'

const SAMPLE_OPTIONS = [4, 8, 16, 32]

export default function Upload() {
  const [file, setFile] = useState(null)
  const [pdbMetadata, setPdbMetadata] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [stage, setStage] = useState('idle')
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const [samplesPerTarget, setSamplesPerTarget] = useState(8)
  const [customSamples, setCustomSamples] = useState('')

  const inputRef = useRef(null)
  const navigate = useNavigate()

  const isBusy = stage === 'uploading' || stage === 'done'
  const usingCustomSamples = !SAMPLE_OPTIONS.includes(samplesPerTarget)
  const resolvedSamplesPerTarget = usingCustomSamples
    ? Math.max(1, Math.min(64, Number(customSamples) || 8))
    : samplesPerTarget

  const parsePdbMetadata = (pdbText) => {
    const chains = new Set()
    let atomCount = 0
    let hetatmCount = 0

    for (const line of pdbText.split('\n')) {
      if (line.startsWith('ATOM')) {
        atomCount += 1
        const chainId = line[21]?.trim()
        if (chainId) chains.add(chainId)
      }

      if (line.startsWith('HETATM')) {
        hetatmCount += 1
      }
    }

    return {
      residueCount: atomCount,
      chainCount: chains.size,
      ligandCount: hetatmCount,
    }
  }

  const handleFile = async (nextFile) => {
    if (!nextFile) return
    const pdbText = await nextFile.text()
    setFile(nextFile)
    setPdbMetadata(parsePdbMetadata(pdbText))
    setErrorMessage('')
    setStatusMessage('')
  }

  const handleDrop = async (event) => {
    event.preventDefault()
    setDragging(false)
    if (isBusy) return
    await handleFile(event.dataTransfer.files[0])
  }

  const runModel = async () => {
    if (!file || isBusy) return

    setErrorMessage('')
    setStage('uploading')
    setStatusMessage('Uploading target and running generation...')

    try {
      const payload = await uploadTarget(file, {
        samplesPerTarget: resolvedSamplesPerTarget,
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
          <p className="text-white/40 text-xs font-bold tracking-[0.3em] uppercase mb-3">Bind Hard</p>
          <h1 className="text-white text-4xl font-bold tracking-tight">Upload Target</h1>
          <p className="text-white/40 text-sm mt-2">PDB file + sample count</p>
        </div>

        <motion.div
          onClick={() => !isBusy && inputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault()
            if (!isBusy) setDragging(true)
          }}
          onDragOver={(event) => {
            event.preventDefault()
            if (!isBusy) setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className="relative rounded-2xl border-2 border-dashed transition-all duration-200 p-12 flex flex-col items-center gap-4 text-center"
          animate={{
            borderColor: dragging ? 'rgba(45, 212, 191, 1)' : 'rgba(45, 212, 191, 0.32)',
            boxShadow: dragging
              ? '0 0 0 1px rgba(45, 212, 191, 0.95), 0 0 30px rgba(45, 212, 191, 0.28)'
              : '0 0 0 1px rgba(45, 212, 191, 0.18), 0 0 0 rgba(45, 212, 191, 0)',
          }}
          transition={{ duration: 0.2 }}
          style={{
            cursor: isBusy ? 'not-allowed' : 'pointer',
            background: dragging ? 'rgba(45, 212, 191, 0.07)' : 'rgba(255,255,255,0.02)',
            opacity: isBusy ? 0.8 : 1,
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdb"
            className="hidden"
            onChange={async (event) => handleFile(event.target.files[0])}
            disabled={isBusy}
          />
          {file ? (
            <>
              <p className="text-white font-semibold">{file.name}</p>
              {pdbMetadata && (
                <div className="flex flex-wrap justify-center gap-2">
                  <span className="rounded-full border border-teal-300/25 bg-teal-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-teal-100">
                    ATOM {pdbMetadata.residueCount}
                  </span>
                  <span className="rounded-full border border-teal-300/25 bg-teal-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-teal-100">
                    Chains {pdbMetadata.chainCount}
                  </span>
                  <span className="rounded-full border border-teal-300/25 bg-teal-300/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-teal-100">
                    HETATM {pdbMetadata.ligandCount}
                  </span>
                </div>
              )}
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
        </motion.div>

        <div className="mt-6 grid grid-cols-1 gap-4">
          <label className="block">
            <span className="mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40">
              Samples
            </span>
            <div className="grid grid-cols-5 gap-2">
              {SAMPLE_OPTIONS.map((option) => {
                const isActive = samplesPerTarget === option

                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => setSamplesPerTarget(option)}
                    disabled={isBusy}
                    className={`rounded-xl border px-3 py-3 text-sm font-semibold transition-colors duration-200 ${
                      isActive
                        ? 'bg-teal-500/10 border-teal-500/50 text-teal-400'
                        : 'bg-white/5 border-white/10 text-white/40'
                    } disabled:cursor-not-allowed disabled:opacity-50`}
                  >
                    {option}
                  </button>
                )
              })}
              <button
                type="button"
                onClick={() => setSamplesPerTarget(0)}
                disabled={isBusy}
                className={`rounded-xl border px-3 py-3 text-sm font-semibold transition-colors duration-200 ${
                  usingCustomSamples
                    ? 'bg-teal-500/10 border-teal-500/50 text-teal-400'
                    : 'bg-white/5 border-white/10 text-white/40'
                } disabled:cursor-not-allowed disabled:opacity-50`}
              >
                Other
              </button>
            </div>
            {usingCustomSamples && (
              <input
                type="number"
                min="1"
                max="64"
                value={customSamples}
                onChange={(event) => setCustomSamples(event.target.value)}
                disabled={isBusy}
                placeholder="Enter a value from 1 to 64"
                className="mt-3 w-full rounded-xl border border-teal-500/30 bg-white/5 px-3 py-3 text-sm text-white outline-none placeholder:text-white/25 focus:border-teal-500/50"
              />
            )}
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
