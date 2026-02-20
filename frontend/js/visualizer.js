// D3.js Chart Visualization (30m OHLCV candlesticks)
import * as d3 from 'd3'

function parseDateLike(input) {
    if (input == null) return null
    const d = input instanceof Date ? input : new Date(input)
    return Number.isNaN(d.getTime()) ? null : d
}

function toNumber(value) {
    const n = Number(value)
    return Number.isFinite(n) ? n : null
}

function normalizeCandle(row, previousClose = null) {
    const date = parseDateLike(
        row.timestamp ?? row.datetime ?? row.date ?? row.time ?? row.t
    )
    if (!date) return null

    const open = toNumber(row.open ?? row.o)
    const high = toNumber(row.high ?? row.h)
    const low = toNumber(row.low ?? row.l)
    const close = toNumber(row.close ?? row.c ?? row.value)
    const volume = toNumber(row.volume ?? row.v ?? 0) ?? 0

    const fallback = close ?? previousClose
    if (fallback == null) return null

    const safeOpen = open ?? previousClose ?? fallback
    const safeClose = close ?? safeOpen
    const safeHigh = high ?? Math.max(safeOpen, safeClose)
    const safeLow = low ?? Math.min(safeOpen, safeClose)

    return {
        date,
        open: safeOpen,
        high: Math.max(safeHigh, safeOpen, safeClose),
        low: Math.min(safeLow, safeOpen, safeClose),
        close: safeClose,
        volume
    }
}

function clampCandlesByTimespan(candles, timespan) {
    if (!candles.length || !timespan || timespan === 'max') return candles

    const sorted = [...candles].sort((a, b) => a.date - b.date)
    const end = sorted[sorted.length - 1].date
    let start = null

    if (timespan === 'ytd') {
        start = new Date(end.getFullYear(), 0, 1)
    } else {
        const hourMatch = String(timespan).match(/^(\d+)h$/i)
        const dayMatch = String(timespan).match(/^(\d+)d$/)
        const monthMatch = String(timespan).match(/^(\d+)m$/i)
        const yearMatch = String(timespan).match(/^(\d+)y$/i)

        if (hourMatch) {
            start = new Date(end)
            start.setHours(start.getHours() - Number(hourMatch[1]))
        } else if (dayMatch) {
            start = new Date(end)
            start.setDate(start.getDate() - Number(dayMatch[1]))
        } else if (monthMatch) {
            start = new Date(end)
            start.setMonth(start.getMonth() - Number(monthMatch[1]))
        } else if (yearMatch) {
            start = new Date(end)
            start.setFullYear(start.getFullYear() - Number(yearMatch[1]))
        }
    }

    if (!start) return sorted

    const clamped = sorted.filter(d => d.date >= start && d.date <= end)
    return clamped.length > 1 ? clamped : sorted.slice(-2)
}

function computeAdaptiveMovingAverage(values, windowSize) {
    if (!values.length) return []
    const out = new Array(values.length)
    let runningSum = 0

    for (let i = 0; i < values.length; i++) {
        runningSum += values[i]
        const start = Math.max(0, i - windowSize + 1)
        if (start > 0) {
            runningSum -= values[start - 1]
        }
        const span = i - start + 1
        out[i] = span > 0 ? (runningSum / span) : values[i]
    }

    return out
}

export class TachionChart {
    constructor(container) {
        this.container = container
        this.margin = { top: 18, right: 18, bottom: 26, left: 54 }
        this.currentData = []
        this.lastTimespan = 'max'
        this.showMovingAverage = false
        this.upColor = '#27ae60'
        this.downColor = '#e74c3c'

        // Clear any existing content
        d3.select(container).selectAll('*').remove()

        // Create responsive SVG that fills the chart container
        this.rootSvg = d3.select(container)
            .append('svg')
            .attr('class', 'tick-chart-svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('preserveAspectRatio', 'none')

        this.svg = this.rootSvg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`)

        // Initialize scales
        // xScale is index-based (not wall-clock continuous) to avoid weekend/closed-market gaps.
        this.xScale = d3.scaleLinear()
        this.yScale = d3.scaleLinear()
        this.xLabelFormat = d3.timeFormat('%b %d')
        this.baseDates = []
        this.currentMaxX = 1
        this.indexedCandles = []

        this.gridLayer = this.svg.append('g').attr('class', 'grid-layer')
        this.candleLayer = this.svg.append('g').attr('class', 'candle-layer')
        this.overlayLayer = this.svg.append('g').attr('class', 'overlay-layer')

        this.xAxis = this.overlayLayer.append('g')
            .attr('class', 'x-axis')

        this.yAxis = this.overlayLayer.append('g')
            .attr('class', 'y-axis')

        this.predictionLine = this.overlayLayer.append('path')
            .attr('class', 'prediction-line')
            .attr('fill', 'none')
            .attr('stroke', 'rgba(114, 137, 218, 0.95)')
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '3,3')
            .attr('opacity', 0)

        this.movingAverageLine = this.overlayLayer.append('path')
            .attr('class', 'moving-average-line')
            .attr('fill', 'none')
            .attr('stroke', 'rgba(241, 196, 15, 0.95)')
            .attr('stroke-width', 1.8)
            .attr('opacity', 0)

        this.hoverLine = this.overlayLayer.append('line')
            .attr('class', 'hover-line')
            .attr('stroke', 'rgba(255,255,255,0.3)')
            .attr('stroke-width', 1)
            .attr('y1', 0)
            .attr('opacity', 0)

        this.hoverRect = this.overlayLayer.append('rect')
            .attr('class', 'hover-capture')
            .attr('fill', 'transparent')
            .style('pointer-events', 'all')

        this.tooltip = d3.select(container)
            .append('div')
            .attr('class', 'chart-tooltip')
            .style('opacity', 0)

        this.resizeObserver = new ResizeObserver(() => {
            this.resize()
            if (this.currentData.length) {
                this.drawCandles(this.currentData)
            }
        })
        this.resizeObserver.observe(container)

        this.resize()
    }

    resize() {
        const rect = this.container.getBoundingClientRect()
        const fullWidth = Math.max(180, rect.width)
        const fullHeight = Math.max(200, rect.height)

        this.width = Math.max(80, fullWidth - this.margin.left - this.margin.right)
        this.height = Math.max(80, fullHeight - this.margin.top - this.margin.bottom)

        this.rootSvg.attr('viewBox', `0 0 ${fullWidth} ${fullHeight}`)

        this.svg.attr('transform', `translate(${this.margin.left},${this.margin.top})`)
        this.xScale.range([0, this.width])
        this.yScale.range([this.height, 0])
        this.xAxis.attr('transform', `translate(0,${this.height})`)
        this.hoverRect
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', this.width)
            .attr('height', this.height)
        this.hoverLine.attr('y2', this.height)
    }

    // Render historical candles from backend OHLCV (30m expected)
    renderHistory(data, timespan = 'max', showMovingAverage = false) {
        let previousClose = null
        const normalized = (data ?? [])
            .map(row => {
                const candle = normalizeCandle(row, previousClose)
                if (candle) previousClose = candle.close
                return candle
            })
            .filter(Boolean)
            .sort((a, b) => a.date - b.date)

        this.lastTimespan = timespan
        this.showMovingAverage = !!showMovingAverage
        this.currentData = clampCandlesByTimespan(normalized, timespan)
        this.drawCandles(this.currentData)
    }

    drawCandles(candles) {
        if (!candles.length) {
            this.candleLayer.selectAll('*').remove()
            this.gridLayer.selectAll('*').remove()
            this.xAxis.selectAll('*').remove()
            this.yAxis.selectAll('*').remove()
            this.predictionLine.attr('opacity', 0)
            return
        }

        const indexedCandles = candles.map((d, i) => ({ ...d, _i: i }))
        this.indexedCandles = indexedCandles
        const xMax = Math.max(1, indexedCandles.length - 1)
        this.currentMaxX = xMax
        this.baseDates = indexedCandles.map(d => d.date)

        const spanMs = this.baseDates.length > 1
            ? this.baseDates[this.baseDates.length - 1] - this.baseDates[0]
            : 0
        this.xLabelFormat = spanMs <= (2 * 24 * 60 * 60 * 1000)
            ? d3.timeFormat('%b %d %H:%M')
            : d3.timeFormat('%b %d')

        const lows = candles.map(d => d.low)
        const highs = candles.map(d => d.high)
        const yMin = d3.min(lows)
        const yMax = d3.max(highs)
        const ySpan = Math.max(1e-9, yMax - yMin)
        const yPad = ySpan * 0.08

        this.xScale.domain([0, xMax])
        this.yScale.domain([yMin - yPad, yMax + yPad])

        const xAxis = d3.axisBottom(this.xScale)
            .ticks(Math.min(8, indexedCandles.length))
            .tickFormat((idx) => {
                const i = Math.max(0, Math.min(indexedCandles.length - 1, Math.round(idx)))
                const dt = this.baseDates[i]
                return dt ? this.xLabelFormat(dt) : ''
            })
        const yAxis = d3.axisLeft(this.yScale).ticks(6)

        this.xAxis.call(xAxis)
        this.yAxis.call(yAxis)

        this.xAxis.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.2)')
        this.yAxis.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.2)')
        this.xAxis.selectAll('text').attr('fill', 'rgba(238,238,238,0.75)').style('font-size', '10px')
        this.yAxis.selectAll('text').attr('fill', 'rgba(238,238,238,0.75)').style('font-size', '10px')

        const yGrid = d3.axisLeft(this.yScale).ticks(6).tickSize(-this.width).tickFormat('')
        this.gridLayer.call(yGrid)
        this.gridLayer.selectAll('line').attr('stroke', 'rgba(255,255,255,0.08)')
        this.gridLayer.selectAll('path').attr('stroke', 'transparent')

        const minSpacingPx = indexedCandles.length > 1
            ? this.width / (indexedCandles.length - 1)
            : this.width
        const bodyWidth = Math.max(2, Math.min(14, (minSpacingPx || this.width) * 0.7))

        const candle = this.candleLayer
            .selectAll('g.candle')
            .data(indexedCandles, d => `${d.date.toISOString()}:${d._i}`)

        candle.exit().remove()

        const candleEnter = candle.enter()
            .append('g')
            .attr('class', 'candle')

        candleEnter.append('line').attr('class', 'wick')
        candleEnter.append('rect').attr('class', 'body')

        const merged = candleEnter.merge(candle)

        merged.select('line.wick')
            .attr('x1', d => this.xScale(d._i))
            .attr('x2', d => this.xScale(d._i))
            .attr('y1', d => this.yScale(d.high))
            .attr('y2', d => this.yScale(d.low))
            .attr('stroke-width', 1)
            .attr('stroke', d => (d.close >= d.open ? this.upColor : this.downColor))

        merged.select('rect.body')
            .attr('x', d => this.xScale(d._i) - bodyWidth / 2)
            .attr('width', bodyWidth)
            .attr('y', d => this.yScale(Math.max(d.open, d.close)))
            .attr('height', d => Math.max(1, Math.abs(this.yScale(d.open) - this.yScale(d.close))))
            .attr('fill', d => (d.close >= d.open ? this.upColor : this.downColor))
            .attr('opacity', 0.95)

        if (this.showMovingAverage && indexedCandles.length > 0) {
            const maWindow = 30
            const closes = indexedCandles.map(d => d.close)
            const maValues = computeAdaptiveMovingAverage(closes, maWindow)
            const maPoints = indexedCandles.map((d, i) => ({ x: d._i, y: maValues[i] }))

            const maLine = d3.line()
                .x(d => this.xScale(d.x))
                .y(d => this.yScale(d.y))

            this.movingAverageLine
                .datum(maPoints)
                .attr('d', maLine)
                .attr('opacity', 1)
        } else {
            this.movingAverageLine.attr('opacity', 0)
        }

        this.predictionLine.attr('opacity', 0)
        this._bindHover()
    }

    _formatPrice(value) {
        const n = Number(value)
        if (!Number.isFinite(n)) return '—'
        return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })
    }

    _formatVolume(value) {
        const n = Number(value)
        if (!Number.isFinite(n)) return '—'
        return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
    }

    _bindHover() {
        this.hoverRect
            .on('mousemove', (event) => {
                if (!this.indexedCandles.length) return

                const [x] = d3.pointer(event, this.hoverRect.node())
                const idx = Math.max(0, Math.min(this.indexedCandles.length - 1, Math.round(this.xScale.invert(x))))
                const d = this.indexedCandles[idx]
                if (!d) return

                const cx = this.xScale(d._i)

                this.hoverLine
                    .attr('x1', cx)
                    .attr('x2', cx)
                    .attr('opacity', 1)

                const dateLabel = d3.timeFormat('%Y-%m-%d %H:%M')(d.date)
                this.tooltip
                    .html(
                        `<div><strong>Time:</strong> ${dateLabel}</div>` +
                        `<div><strong>Price (close):</strong> ${this._formatPrice(d.close)}</div>` +
                        `<div><strong>Volume:</strong> ${this._formatVolume(d.volume)}</div>`
                    )
                    .style('opacity', 1)

                const rect = this.container.getBoundingClientRect()
                const tipNode = this.tooltip.node()
                const tipWidth = tipNode?.offsetWidth ?? 150
                const tipHeight = tipNode?.offsetHeight ?? 70

                const left = Math.min(
                    Math.max(8, cx + this.margin.left + 12),
                    rect.width - tipWidth - 8
                )
                const top = Math.min(
                    Math.max(8, this.yScale(d.high) + this.margin.top - tipHeight - 10),
                    rect.height - tipHeight - 8
                )

                this.tooltip
                    .style('left', `${left}px`)
                    .style('top', `${top}px`)
            })
            .on('mouseleave', () => {
                this.hoverLine.attr('opacity', 0)
                this.tooltip.style('opacity', 0)
            })
    }

    // Keep prediction support as a lightweight close-price overlay
    animatePrediction(prediction) {
        if (!this.currentData.length) return

        const last = this.currentData[this.currentData.length - 1]
        const predictionData = (prediction?.timestamps ?? []).map((t, i) => ({
            date: parseDateLike(t),
            close: toNumber(prediction?.medians?.[i])
        })).filter(d => d.date && d.close != null)

        if (!predictionData.length) {
            this.predictionLine.attr('opacity', 0)
            return
        }

        const startX = Math.max(0, this.currentData.length - 1)
        const pathData = [
            { x: startX, close: last.close },
            ...predictionData.map((d, i) => ({ x: startX + i + 1, close: d.close }))
        ]

        const extendedMaxX = Math.max(this.currentMaxX, startX + predictionData.length)
        const xOverlayScale = this.xScale.copy().domain([0, extendedMaxX])
        const xAxis = d3.axisBottom(xOverlayScale)
            .ticks(Math.min(8, this.baseDates.length))
            .tickFormat((idx) => {
                const i = Math.round(idx)
                if (i < 0 || i >= this.baseDates.length) return ''
                const dt = this.baseDates[i]
                return dt ? this.xLabelFormat(dt) : ''
            })
        this.xAxis.call(xAxis)
        this.xAxis.selectAll('path,line').attr('stroke', 'rgba(255,255,255,0.2)')
        this.xAxis.selectAll('text').attr('fill', 'rgba(238,238,238,0.75)').style('font-size', '10px')

        const line = d3.line()
            .x(d => xOverlayScale(d.x))
            .y(d => this.yScale(d.close))

        this.predictionLine
            .datum(pathData)
            .attr('d', line)
            .attr('opacity', 1)
    }
}
