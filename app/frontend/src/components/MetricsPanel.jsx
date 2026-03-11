const BASELINE_AVG = Object.freeze({
  vina_score: -6.64,
  qed_score: 0.48,
  sa_score: 0.58,
})

const METRIC_CONFIG = [
  {
    key: 'vina_score',
    label: 'Vina',
    betterDirection: 'lower',
    domain: [-12, -2],
    decimals: 2,
  },
  {
    key: 'qed_score',
    label: 'QED',
    betterDirection: 'higher',
    domain: [0, 1],
    decimals: 3,
  },
  {
    key: 'sa_score',
    label: 'SA',
    betterDirection: 'higher',
    domain: [0, 1],
    decimals: 3,
    normalizeValue: normalizeSaScoreToUnit,
  },
]

function normalizeSaScoreToUnit(value) {
  if (!Number.isFinite(value)) return null
  if (value >= 0 && value <= 1) return value
  if (value >= 1 && value <= 10) return (value - 1) / 9
  return null
}

function parseFinite(value) {
  const numberValue = Number(value)
  return Number.isFinite(numberValue) ? numberValue : null
}

function clamp(value, minValue, maxValue) {
  return Math.min(maxValue, Math.max(minValue, value))
}

function toPercent(value, domain) {
  const [minValue, maxValue] = domain
  if (!Number.isFinite(value) || minValue === maxValue) return 50
  return clamp(((value - minValue) / (maxValue - minValue)) * 100, 0, 100)
}

function formatValue(value, decimals) {
  if (!Number.isFinite(value)) return 'N/A'
  return Number(value).toFixed(decimals)
}

function formatDelta(delta, decimals) {
  const sign = delta > 0 ? '+' : ''
  return `${sign}${delta.toFixed(decimals)}`
}

function getComparison(delta, betterDirection) {
  if (delta === 0) return 'neutral'
  if (betterDirection === 'higher') return delta > 0 ? 'better' : 'worse'
  return delta < 0 ? 'better' : 'worse'
}

export default function MetricsPanel({ entry }) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto rounded-2xl border border-white/10 bg-[#0b0f18] p-4 sm:p-5">
      <div className="flex items-start justify-between gap-3 border-b border-white/10 pb-3">
        <div>
          <h2 className="text-base font-semibold tracking-tight text-white sm:text-lg">Model vs Baseline</h2>
          <p className="mt-1 text-[11px] text-white/55 sm:text-xs">
            Baseline values are from (2023) TargetDiff's diffusion model's best-scoring molecules, averaged across the test set.
          </p>
        </div>
      </div>

      <div className="mt-2 grid min-h-0 flex-1 grid-rows-3 gap-2.5">
        {METRIC_CONFIG.map((metric) => {
          const rawBaselineValue = parseFinite(BASELINE_AVG[metric.key])
          const rawSampleValue = parseFinite(entry?.[metric.key])
          const baselineValue = metric.normalizeValue
            ? metric.normalizeValue(rawBaselineValue)
            : rawBaselineValue
          const sampleValue = metric.normalizeValue
            ? metric.normalizeValue(rawSampleValue)
            : rawSampleValue
          const hasSampleValue = sampleValue !== null
          const baselinePercent = toPercent(baselineValue, metric.domain)
          const samplePercent = hasSampleValue ? toPercent(sampleValue, metric.domain) : null

          const delta = hasSampleValue ? sampleValue - baselineValue : null
          const comparison = delta === null ? 'neutral' : getComparison(delta, metric.betterDirection)
          const directionLabel = metric.betterDirection === 'higher' ? 'higher is better' : 'lower is better'
          const scaleLabel = metric.key === 'sa_score' ? 'normalized to 0-1' : ''
          const rowMutedClass = hasSampleValue ? '' : 'opacity-60'
          const baselineLabelPercent = clamp(baselinePercent, 8, 92)
          const sampleLabelPercent = hasSampleValue ? clamp(samplePercent, 10, 90) : null
          const comparisonLabel =
            delta === null
              ? 'N/A'
              : delta === 0
                ? 'matches baseline'
                : `${comparison} vs baseline`
          const comparisonClass =
            comparison === 'better'
              ? 'border-emerald-300/35 bg-emerald-500/15 text-emerald-100'
              : comparison === 'worse'
                ? 'border-rose-300/35 bg-rose-500/15 text-rose-100'
                : 'border-white/20 bg-white/5 text-white/75'

          return (
            <div key={metric.key} className={`flex min-h-0 flex-col justify-between rounded-xl border border-white/10 bg-black/20 p-2.5 sm:p-3 ${rowMutedClass}`}>
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold text-white">{metric.label}</p>
                  <p className="text-[10px] uppercase tracking-wider text-white/45 sm:text-[11px]">
                    {directionLabel}{scaleLabel ? ` • ${scaleLabel}` : ''}
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-1.5 text-[11px] sm:text-xs">
                  <span className="rounded-md border border-white/15 bg-white/5 px-2 py-0.5 font-medium text-white/85">
                    this molecule: {formatValue(sampleValue, metric.decimals)}
                  </span>
                  <span className="rounded-md border border-white/15 bg-white/5 px-2 py-0.5 font-medium text-white/75">
                    TargetDiff: {formatValue(baselineValue, metric.decimals)}
                  </span>
                  <span className={`rounded-md border px-2 py-0.5 font-semibold ${comparisonClass}`}>
                    {comparisonLabel}
                    {delta !== null && delta !== 0 ? ` (${formatDelta(delta, metric.decimals)})` : ''}
                  </span>
                </div>
              </div>

              <div className="mt-1.5">
                <div className="relative h-12">
                  <div className="absolute left-0 right-0 top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-white/10" />

                  <div
                    className="absolute top-1/2 h-5 w-px -translate-y-1/2 bg-cyan-200/90"
                    style={{ left: `${baselinePercent}%` }}
                    aria-hidden="true"
                  />
                  <span
                    className="absolute bottom-0.5 -translate-x-1/2 rounded-sm border border-cyan-200/30 bg-cyan-400/15 px-1.5 py-0 text-[9px] font-semibold uppercase tracking-wide text-cyan-100"
                    style={{ left: `${baselineLabelPercent}%` }}
                  >
                    TargetDiff
                  </span>

                  {hasSampleValue && (
                    <>
                      <div
                        className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-blue-100 bg-blue-500 shadow-[0_0_0_2px_rgba(59,130,246,0.28)]"
                        style={{ left: `${samplePercent}%` }}
                        aria-hidden="true"
                      />
                      <span
                        className="absolute top-0.5 -translate-x-1/2 rounded-sm border border-blue-300/30 bg-blue-500/20 px-1.5 py-0 text-[9px] font-semibold uppercase tracking-wide text-blue-100"
                        style={{ left: `${sampleLabelPercent}%` }}
                      >
                        this molecule
                      </span>
                    </>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
