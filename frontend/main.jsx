// Main entry point for React application
import React from 'react'
import ReactDOM from 'react-dom/client'
import { StateProvider } from './js/state'
import { useAppState } from './js/state'
import Header from './components/header'
import Sidebar from './components/sidebar'
import Graph from './components/graph'
import Footer from './components/footer'
import TimespanSelector from './components/timespan-selector'
import './styles.css'

const CURRENCY_NAMES = {
	USD: 'USD', EUR: 'EUR', GBP: 'GBP', JPY: 'JPY', CAD: 'CAD',
	AUD: 'AUD', NZD: 'NZD', CHF: 'CHF', SGD: 'SGD', HKD: 'HKD',
	SEK: 'SEK', NOK: 'NOK', DKK: 'DKK', ZAR: 'ZAR', MXN: 'MXN',
	TRY: 'TRY', PLN: 'PLN', CNH: 'CNH', INR: 'INR', THB: 'THB',
}

function formatForexLabel(symbol, value) {
	const clean = (symbol || '').toUpperCase().replace(/[\/_-]/g, '')
	const base = clean.slice(0, 3)
	const quote = clean.slice(3, 6)
	const formatted = value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
	return `1 ${CURRENCY_NAMES[base] || base} equals ${formatted} ${CURRENCY_NAMES[quote] || quote}`
}

function TrendlinesTitle() {
	const { historicalData, drawMovingAverage, setDrawMovingAverage, currentSymbol, assetClass } = useAppState()

	const closes = (historicalData || [])
		.map(d => Number(d?.close ?? d?.value))
		.filter(Number.isFinite)

	const first = closes.length ? closes[0] : null
	const last = closes.length ? closes[closes.length - 1] : null
	const pct = first && last ? ((last - first) / first) * 100 : null

	let priceText = '—'
	if (last != null) {
		if (assetClass === 'forex' && currentSymbol) {
			priceText = formatForexLabel(currentSymbol, last)
		} else {
			priceText = `$${last.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 })}`
		}
	}

	const pctText = pct != null
		? `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`
		: '—'

	const pctClass = pct == null ? 'neutral' : pct >= 0 ? 'up' : 'down'

	return (
		<div className="chart-title-wrap">
			<h1>Trendlines</h1>
			<span className={`chart-title-metrics ${pctClass}`}>{priceText} • {pctText}</span>
			<label className="ma-switch" aria-label="Draw moving averages">
				<input
					type="checkbox"
					checked={!!drawMovingAverage}
					onChange={(e) => setDrawMovingAverage(e.target.checked)}
				/>
				<span>Draw MA</span>
			</label>
		</div>
	)
}

function App() {
	return (
		<StateProvider>
			<Header />
			<main className="main-content">
				<section className="chart-panel">
					<div className="chart-header">
						<TrendlinesTitle />
						<TimespanSelector />
					</div>
					<Graph />
				</section>
				<aside className="control-panel">
					<Sidebar />
				</aside>
			</main>
			<Footer />
		</StateProvider>
	)
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />)
