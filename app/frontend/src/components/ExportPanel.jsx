import { useMemo, useState } from 'react'
import { downloadBatchSdf } from '../lib/sdfExport'

const INPUT_CLS = 'w-full rounded-lg border border-white/10 bg-[#070b14] px-2 py-1.5 text-xs text-white outline-none focus:border-blue-400/60'
const SECTION_CLS = 'rounded-xl border border-white/10 p-3'
const SECTION_BG = { background: 'rgba(255,255,255,0.02)' }
const LABEL_CLS = 'mb-1 block text-[11px] font-semibold uppercase tracking-widest text-white/40'
const BTN_GHOST = 'flex-1 rounded-lg border border-white/10 px-2 py-1.5 text-[10px] font-semibold text-white/60 hover:border-white/25 hover:text-white transition-colors'

export default function ExportPanel({samples, selected, onSelectAll, onDeselectAll, proteinId, filters, onFiltersChange,}) {
  const [status, setStatus] = useState('')

  const affinityRange = useMemo(() => {
    const scores = samples.map((s) => s.vina_score).filter(Number.isFinite)
    if (scores.length === 0) return { min: -12, max: 0 }
    return { min: Math.floor(Math.min(...scores)), max: Math.ceil(Math.max(...scores)) }
  }, [samples])

  const filtered = useMemo(() => samples.filter((s) => {
    if (filters.validOnly && s.status !== 'completed') return false
    if (filters.affinityMin != null && Number.isFinite(s.vina_score) && s.vina_score < filters.affinityMin) return false
    if (filters.affinityMax != null && Number.isFinite(s.vina_score) && s.vina_score > filters.affinityMax) return false
    if (filters.minAtoms != null && Number.isFinite(s.n_atoms) && s.n_atoms < filters.minAtoms) return false
    if (filters.hasSmilesOnly && !s.smiles) return false
    return true
  }), [samples, filters])

  const selectedFiltered = filtered.filter((s) => selected.has(Number(s.sample_idx)))
  const allSelected = filtered.length > 0 && filtered.every((s) => selected.has(Number(s.sample_idx)))
  const validFiltered = filtered.filter((s) => s.status === 'completed')

  const doExport = (rows, name) => {
    if (rows.length === 0) {
      setStatus('No molecules to export.')
      return
    }
    const ok = downloadBatchSdf(rows, { proteinId, filename: `${proteinId || 'ligands'}_${name}.sdf` })
    setStatus(ok ? `Exported ${rows.length} molecules.` : 'Export failed.')
    if (ok) setTimeout(() => setStatus(''), 3000)
  }

  const setFilter = (patch) => onFiltersChange({ ...filters, ...patch })
  const numOrNull = (v) => (v === '' ? null : Number(v))

  const statusIsError = status.includes('failed') || status.includes('No')

  return (
    <div className="space-y-3">
      <div className={SECTION_CLS} style={SECTION_BG}>
        <span className={LABEL_CLS}>Filters</span>

        <div className="space-y-2">
          <div>
            <label className="mb-0.5 block text-[10px] text-white/35">
              Vina range ({filters.affinityMin ?? affinityRange.min} to {filters.affinityMax ?? affinityRange.max})
            </label>
            <div className="flex items-center gap-2">
              <input type="number" step="0.5" value={filters.affinityMin ?? affinityRange.min}
                onChange={(e) => setFilter({ affinityMin: numOrNull(e.target.value) })} className={INPUT_CLS} />
              <span className="text-[10px] text-white/30">to</span>
              <input type="number" step="0.5" value={filters.affinityMax ?? affinityRange.max}
                onChange={(e) => setFilter({ affinityMax: numOrNull(e.target.value) })} className={INPUT_CLS} />
            </div>
          </div>

          <div>
            <label className="mb-0.5 block text-[10px] text-white/35">Min atoms</label>
            <input type="number" min="0" value={filters.minAtoms ?? ''} placeholder="Any"
              onChange={(e) => setFilter({ minAtoms: numOrNull(e.target.value) })} className={INPUT_CLS} />
          </div>

          <div className="flex items-center gap-3 pt-1">
            <label className="flex items-center gap-1.5 text-[10px] text-white/50 cursor-pointer">
              <input type="checkbox" checked={filters.validOnly ?? true}
                onChange={(e) => setFilter({ validOnly: e.target.checked })} className="accent-blue-500" />
              Valid only
            </label>
            <label className="flex items-center gap-1.5 text-[10px] text-white/50 cursor-pointer">
              <input type="checkbox" checked={filters.hasSmilesOnly ?? false}
                onChange={(e) => setFilter({ hasSmilesOnly: e.target.checked })} className="accent-blue-500" />
              Has SMILES
            </label>
          </div>
        </div>

        <div className="mt-2 text-[10px] text-white/30">
          {filtered.length} of {samples.length} match
        </div>
      </div>

      <div className={SECTION_CLS} style={SECTION_BG}>
        <div className="flex items-center justify-between mb-2">
          <span className={LABEL_CLS}>Selection</span>
          <span className="text-[10px] text-white/50">{selectedFiltered.length} selected</span>
        </div>
        <div className="flex gap-2">
          <button type="button" className={BTN_GHOST}
            onClick={() => allSelected ? onDeselectAll() : onSelectAll(filtered.map((s) => Number(s.sample_idx)))}>
            {allSelected ? 'Deselect All' : 'Select All'}
          </button>
          <button type="button" className={BTN_GHOST} onClick={onDeselectAll} disabled={selected.size === 0}>
            Clear
          </button>
        </div>
      </div>

      <div className={SECTION_CLS} style={SECTION_BG}>
        <span className={`${LABEL_CLS} mb-2`}>Export SDF</span>
        <div className="space-y-2">
          <button type="button" onClick={() => doExport(validFiltered, 'all')}
            disabled={validFiltered.length === 0}
            className="w-full rounded-lg py-2 text-xs font-bold text-white transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
            style={{
              background: validFiltered.length > 0 ? '#196eff' : 'rgba(255,255,255,0.08)',
              boxShadow: validFiltered.length > 0 ? '0 4px 16px rgba(25,110,255,0.25)' : 'none',
            }}>
            Save All Molecules ({validFiltered.length})
          </button>

          <button type="button" onClick={() => doExport(selectedFiltered, 'selected')}
            disabled={selectedFiltered.length === 0}
            className="w-full rounded-lg border border-blue-400/30 py-2 text-xs font-bold text-blue-200 transition-all duration-200 hover:border-blue-400/50 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed"
            style={{ background: 'rgba(25,110,255,0.12)' }}>
            Save Selected ({selectedFiltered.length})
          </button>
        </div>

        {status && (
          <div className={`mt-2 rounded-lg px-2 py-1.5 text-[10px] border ${
            statusIsError ? 'border-red-400/30 bg-red-900/25 text-red-100' : 'border-emerald-300/30 bg-emerald-500/15 text-emerald-100'
          }`}>
            {status}
          </div>
        )}
      </div>
    </div>
  )
}