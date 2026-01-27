// Main entry point for React application
import React from 'react'
import ReactDOM from 'react-dom/client'
import { StateProvider } from './js/state'
import Header from './components/header'
import Sidebar from './components/sidebar'
import Graph from './components/graph'
import Footer from './components/footer'
import './styles.css'

function App() {
	return (
		<StateProvider>
			<Header />
			<main className="main-content">
				<section className="chart-panel">
					<h2>Asset Trendlines</h2>
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
