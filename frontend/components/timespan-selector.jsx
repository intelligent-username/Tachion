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
    { label: '12h', value: '12h',  priority: 6 },
]

const BASE_RESERVED_PX = 148 // custom input + selector padding/border + safety margin
const CHIP_SLOT_PX = 72      // conservative per-chip slot to avoid overflow

function getAvailableSelectorWidth(selectorNode) {
    if (!selectorNode) return 0
    const header = selectorNode.closest('.chart-header')
    if (!header) return selectorNode.clientWidth

    const titleBlock = header.querySelector('.chart-title-wrap') || header.querySelector('h1')
    const titleWidth = titleBlock?.offsetWidth ?? 0
    const headerWidth = header.clientWidth
    const headerStyles = getComputedStyle(header)
    const gap = parseFloat(headerStyles.columnGap || headerStyles.gap || '0') || 0
    const paddingLeft = parseFloat(headerStyles.paddingLeft || '0') || 0
    const paddingRight = parseFloat(headerStyles.paddingRight || '0') || 0
    const sidePadding = paddingLeft + paddingRight + 2

    // When selector wraps to the next line, it should use the full row width.
    if (titleBlock && selectorNode.offsetTop > titleBlock.offsetTop) {
        return Math.max(0, headerWidth - sidePadding)
    }

    return Math.max(0, headerWidth - titleWidth - gap - sidePadding)
}

function getVisibleOptions(containerPx, currentValue) {
    const byPriority = [...ALL_OPTIONS].sort((a, b) => a.priority - b.priority)
    const currentOpt = ALL_OPTIONS.find(o => o.value === currentValue)

    const slots = Math.max(
        1,
        Math.min(
            byPriority.length,
            Math.floor((containerPx - BASE_RESERVED_PX) / CHIP_SLOT_PX)
        )
    )

    const chosen = byPriority.slice(0, slots)

    // Always keep the currently selected chip visible.
    if (currentOpt && !chosen.includes(currentOpt)) {
        if (chosen.length > 0) {
            chosen[chosen.length - 1] = currentOpt
        } else {
            chosen.push(currentOpt)
        }
    }

    const keep = new Set(chosen.map(o => o.value))
    return ALL_OPTIONS.filter(o => keep.has(o.value))
}

function trimOneByPriority(options, currentValue) {
    if (!options.length) return options

    const protectedValue = currentValue
    const removable = options.filter(o => o.value !== protectedValue)
    if (!removable.length) return options

    const drop = removable.reduce((worst, opt) => {
        if (!worst) return opt
        return opt.priority > worst.priority ? opt : worst
    }, null)

    return options.filter(o => o.value !== drop.value)
}

function sameOptionSet(a, b) {
    if (a === b) return true
    if (!a || !b || a.length !== b.length) return false
    for (let i = 0; i < a.length; i += 1) {
        if (a[i].value !== b[i].value) return false
    }
    return true
}

export default function TimespanSelector() {
    const { timespan, setTimespan } = useAppState()
    const [customDays, setCustomDays]           = useState('')
    const [visibleOptions, setVisibleOptions]   = useState(ALL_OPTIONS)
    const [availableWidth, setAvailableWidth] = useState(0)
    const selectorRef = useRef(null)
    const rafRef = useRef(null)

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
        const titleBlock = header?.querySelector('.chart-title-wrap') || header?.querySelector('h1')
        const recalc = () => {
            const nextWidth = getAvailableSelectorWidth(node)
            setAvailableWidth(prev => (prev === nextWidth ? prev : nextWidth))
            setVisibleOptions(prev => {
                const next = getVisibleOptions(nextWidth, timespan)
                return sameOptionSet(prev, next) ? prev : next
            })
        }

        const scheduleRecalc = () => {
            if (rafRef.current != null) return
            rafRef.current = window.requestAnimationFrame(() => {
                rafRef.current = null
                recalc()
            })
        }

        recalc()

        const obs = new ResizeObserver(() => {
            scheduleRecalc()
        })

        if (header) {
            obs.observe(header)
            obs.observe(node)
            if (titleBlock) obs.observe(titleBlock)
        } else {
            obs.observe(node)
        }

        return () => {
            obs.disconnect()
            if (rafRef.current != null) {
                window.cancelAnimationFrame(rafRef.current)
                rafRef.current = null
            }
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    // Recalculate when timespan changes so the selected chip is always shown
    useEffect(() => {
        if (!availableWidth) return
        setVisibleOptions(prev => {
            const next = getVisibleOptions(availableWidth, timespan)
            return sameOptionSet(prev, next) ? prev : next
        })
    }, [timespan, availableWidth])

    // Guard against real overflow after render (single-row edge case).
    useEffect(() => {
        const node = selectorRef.current
        if (!node) return
        if (visibleOptions.length <= 1) return

        const hasOverflow = node.scrollWidth > node.clientWidth + 1
        if (!hasOverflow) return

        setVisibleOptions(prev => {
            const next = trimOneByPriority(prev, timespan)
            return sameOptionSet(prev, next) ? prev : next
        })
    }, [visibleOptions, timespan])

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

            {/* Priority decides visibility; display keeps chronological ring order */}
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
