// D3.js Chart Visualization with Dynamic Expansion Animation
import * as d3 from 'd3'

export class TachionChart {
    constructor(container) {
        this.container = container
        this.margin = { top: 20, right: 80, bottom: 30, left: 60 }
        this.currentData = []

        // Get dimensions
        const rect = container.getBoundingClientRect()
        this.width = rect.width - this.margin.left - this.margin.right
        this.height = 350 - this.margin.top - this.margin.bottom

        // Clear any existing content
        d3.select(container).selectAll('*').remove()

        // Create SVG
        this.svg = d3.select(container)
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`)

        // Initialize scales
        this.xScale = d3.scaleTime().range([0, this.width])
        this.yScale = d3.scaleLinear().range([this.height, 0])

        // Create axis groups
        this.xAxis = this.svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${this.height})`)

        this.yAxis = this.svg.append('g')
            .attr('class', 'y-axis')

        // Create path elements
        this.confidenceArea = this.svg.append('path')
            .attr('class', 'confidence-area')
            .attr('fill', 'rgba(100, 100, 100, 0.3)')
            .attr('opacity', 0)

        this.historyLine = this.svg.append('path')
            .attr('class', 'history-line')
            .attr('fill', 'none')
            .attr('stroke', '#e74c3c')
            .attr('stroke-width', 2)

        this.predictionLine = this.svg.append('path')
            .attr('class', 'prediction-line')
            .attr('fill', 'none')
            .attr('stroke', '#27ae60')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5')
            .attr('opacity', 0)

        // Prediction area label
        this.predictionLabel = this.svg.append('text')
            .attr('class', 'prediction-label')
            .attr('text-anchor', 'middle')
            .attr('fill', '#666')
            .attr('font-size', '12px')
            .attr('opacity', 0)
            .text('Prediction area')
    }

    // Render historical data
    renderHistory(data) {
        // Parse data
        this.currentData = data.map(d => ({
            date: new Date(d.timestamp),
            value: d.value
        }))

        // Update scales
        const xExtent = d3.extent(this.currentData, d => d.date)
        const yExtent = d3.extent(this.currentData, d => d.value)
        const yPadding = (yExtent[1] - yExtent[0]) * 0.1

        this.xScale.domain(xExtent)
        this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding])

        // Update axes
        this.xAxis.call(d3.axisBottom(this.xScale))
        this.yAxis.call(d3.axisLeft(this.yScale))

        // Line generator
        const lineGenerator = d3.line()
            .x(d => this.xScale(d.date))
            .y(d => this.yScale(d.value))

        // Draw history line
        this.historyLine
            .datum(this.currentData)
            .attr('d', lineGenerator)

        // Reset prediction elements
        this.predictionLine.attr('opacity', 0)
        this.confidenceArea.attr('opacity', 0)
        this.predictionLabel.attr('opacity', 0)
    }

    // Animate prediction - THE DYNAMIC EXPANSION
    animatePrediction(prediction) {
        if (!this.currentData.length) return

        const lastHistoryPoint = this.currentData[this.currentData.length - 1]

        // Parse prediction data
        const predictionDates = prediction.timestamps.map(t => new Date(t))
        const newMaxDate = predictionDates[predictionDates.length - 1]

        // Calculate new Y domain including confidence intervals
        const allValues = [
            ...this.currentData.map(d => d.value),
            ...prediction.upper_95s,
            ...prediction.lower_95s
        ]
        const yExtent = d3.extent(allValues)
        const yPadding = (yExtent[1] - yExtent[0]) * 0.1

        // Step 1: Calculate new X domain
        const newXScale = d3.scaleTime()
            .domain([this.xScale.domain()[0], newMaxDate])
            .range([0, this.width])

        // Step 2: Animate X-axis expansion (750ms)
        this.xAxis.transition()
            .duration(750)
            .call(d3.axisBottom(newXScale))

        // Update Y scale
        this.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        this.yAxis.transition()
            .duration(750)
            .call(d3.axisLeft(this.yScale))

        // Line generator with new scale
        const lineGenerator = d3.line()
            .x(d => newXScale(d.date))
            .y(d => this.yScale(d.value))

        // Step 3: Compress historical line
        this.historyLine.transition()
            .duration(750)
            .attr('d', lineGenerator(this.currentData))

        // Prepare prediction data points
        const predictionData = prediction.timestamps.map((t, i) => ({
            date: new Date(t),
            value: prediction.medians[i]
        }))

        // Full prediction path including connection point
        const fullPredictionPath = [
            { date: lastHistoryPoint.date, value: lastHistoryPoint.value },
            ...predictionData
        ]

        // Confidence interval data
        const ciData = prediction.timestamps.map((t, i) => ({
            date: new Date(t),
            lower: prediction.lower_95s[i],
            upper: prediction.upper_95s[i]
        }))

        // Area generator for confidence interval
        const areaGenerator = d3.area()
            .x(d => newXScale(d.date))
            .y0(d => this.yScale(d.lower))
            .y1(d => this.yScale(d.upper))

        // Step 4: Draw prediction line (fade in after axis animation)
        this.predictionLine
            .datum(fullPredictionPath)
            .attr('d', lineGenerator)
            .transition()
            .delay(750)
            .duration(500)
            .attr('opacity', 1)

        // Step 5: Draw confidence interval area (fade in)
        this.confidenceArea
            .datum(ciData)
            .attr('d', areaGenerator)
            .transition()
            .delay(750)
            .duration(500)
            .attr('opacity', 1)

        // Show prediction label
        const labelX = newXScale(predictionDates[Math.floor(predictionDates.length / 2)])
        this.predictionLabel
            .attr('x', labelX)
            .attr('y', 15)
            .transition()
            .delay(750)
            .duration(500)
            .attr('opacity', 1)

        // Update stored scale
        this.xScale = newXScale
    }
}
