// Graph component - D3 chart wrapper
import { useEffect, useRef } from 'react'
import { useAppState } from '../js/state'
import { TachionChart } from '../js/visualizer'

export default function Graph() {
    const containerRef = useRef(null)
    const chartRef = useRef(null)
    const { historicalData, predictionData } = useAppState()

    // Initialize chart on mount
    useEffect(() => {
        if (containerRef.current && !chartRef.current) {
            chartRef.current = new TachionChart(containerRef.current)
        }
    }, [])

    // Update history when data changes
    useEffect(() => {
        if (chartRef.current && historicalData.length > 0) {
            chartRef.current.renderHistory(historicalData)
        }
    }, [historicalData])

    // Animate prediction when it arrives
    useEffect(() => {
        if (chartRef.current && predictionData) {
            chartRef.current.animatePrediction(predictionData)
        }
    }, [predictionData])

    return (
        <div ref={containerRef} className="chart-container">
            {/* D3 will render the chart here */}
        </div>
    )
}
