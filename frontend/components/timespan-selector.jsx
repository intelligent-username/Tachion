import { useEffect, useRef, useState } from 'react'
import { useAppState } from '../js/state'

// Display order is fixed for UI readability.
// priority: 1 = kept first, higher number = dropped first as space shrinks.
const ALL_OPTIONS = [
    { label: 'max', value: 'max',  priority: 7 },
    { label: '1Y',  value: '365d', priority: 5 },
    { label: 'YTD', value: 'ytd',  priority: 2 },
    { label: '6m',  value: '180d', priority: 9 },
    { label: '60d', value: '60d',  priority: 11 },
    { label: '30d', value: '30d',  priority: 4 },
    { label: '14d', value: '14d',  priority: 8 },
    { label: '7d',  value: '7d',   priority: 1 },
    { label: '5d',  value: '5d',   priority: 12 },
    { label: '3d',  value: '3d',   priority: 13 },
    { label: '2d',  value: '2d',   priority: 14 },
    { label: '1d',  value: '1d',   priority: 3 },
]

const CHIP_PX   = 52   // approx px per chip incl. gap
const CUSTOM_PX = 144  // approx px for the custom input incl. gap

function getAvailableSelectorWidth(selectorNode) {
    if (!selectorNode) return 0
    const header = selectorNode.closest('.chart-header')
    if (!header) return selectorNode.clientWidth

    const title = header.querySelector('h1')
    const titleWidth = title?.offsetWidth ?? 0
    const headerWidth = header.clientWidth
    const gap = parseFloat(getComputedStyle(header).columnGap || getComputedStyle(header).gap || '0') || 0
    const sidePadding = 8

    return Math.max(0, headerWidth - titleWidth - gap - sidePadding)
}

function getVisibleOptions(containerPx, currentValue) {
    const slots = Math.max(1, Math.floor((containerPx - CUSTOM_PX - 24) / CHIP_PX))
    const byPriority = [...ALL_OPTIONS].sort((a, b) => a.priority - b.priority)
    const chosen = byPriority.slice(0, slots)

    // Always keep the currently selected chip visible
    const currentOpt = ALL_OPTIONS.find(o => o.value === currentValue)
    if (currentOpt && !chosen.includes(currentOpt)) {
        chosen[chosen.length - 1] = currentOpt
    }

    const keep = new Set(chosen.map(o => o.value))
    return ALL_OPTIONS.filter(o => keep.has(o.value))
}

export default function TimespanSelector() {
    const { timespan, setTimespan } = useAppState()
    const [customDays, setCustomDays]           = useState('')
    const [visibleOptions, setVisibleOptions]   = useState(ALL_OPTIONS)
    const [availableWidth, setAvailableWidth] = useState(0)
    const selectorRef = useRef(null)

    const applyCustomDays = (value) => {
        const days = Number(value)
        if (!Number.isFinite(days) || days <= 0) return
        setTimespan(`${Math.floor(days)}d`)
    }

    // Recalculate visible chips on container resize
    useEffect(() => {
        const node = selectorRef.current
        if (!node) return

        const header = node.closest('.chart-header')
        const recalc = () => {
            const nextWidth = getAvailableSelectorWidth(node)
            setAvailableWidth(nextWidth)
            setVisibleOptions(getVisibleOptions(nextWidth, timespan))
        }

        recalc()

        const obs = new ResizeObserver(() => {
            recalc()
        })

        if (header) {
            obs.observe(header)
        } else {
            obs.observe(node)
        }

        return () => obs.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    // Recalculate when timespan changes so the selected chip is always shown
    useEffect(() => {
        if (!availableWidth) return
        setVisibleOptions(getVisibleOptions(availableWidth, timespan))
    }, [timespan, availableWidth])

    return (
        <div className="timespan-selector" role="group" aria-label="Select chart timespan" ref={selectorRef}>

            {/* Custom numeric input — always leftmost inside the ring */}
            <input
                type="number"
                min="1"
                className="timespan-custom-input"
                placeholder="custom"
                value={customDays}
                onChange={(e) => { setCustomDays(e.target.value); applyCustomDays(e.target.value) }}
                onKeyDown={(e) => e.key === 'Enter' && applyCustomDays(customDays)}
                onBlur={() => applyCustomDays(customDays)}
            />

            {/* Priority-filtered chips in display order */}
            {visibleOptions.map(({ label, value }) => (
                <button
                    key={value}
                    type="button"
                    className={`timespan-chip${timespan === value ? ' active' : ''}`}
                    onClick={() => setTimespan(value)}
                >
                    {label}
                </button>
            ))}
        </div>
    )
}
