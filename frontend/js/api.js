// API calls
const API_BASE = '/api'

// Send search request to backend
export async function sendSearch(symbol, assetClass, timespan = 'max') {
    const response = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            symbol: symbol,
            type: assetClass,
            timespan: timespan
        })
    })

    if (!response.ok) {
        throw new Error(`Failed to send search request: ${response.statusText}`)
    }

    return await response.json()
}

// Fetch historical OHLCV data for a symbol (30m interval expected from backend)
export async function fetchHistory(symbol, assetClass, timespan = 'max') {
    const params = new URLSearchParams({
        symbol: symbol,
        asset_class: assetClass,
        timespan: timespan,
        interval: '30m'
    })

    const response = await fetch(`${API_BASE}/history?${params.toString()}`, {
        method: 'GET'
    })

    if (!response.ok) {
        throw new Error(`Failed to fetch history: ${response.statusText}`)
    }

    const json = await response.json()
    return json.data
}

// Get prediction from backend
// Sends the visualizer's current OHLCV candle data and expects back
// ⌊ln(n)⌋ prediction points, each with upper, lower, and median.
export async function fetchPrediction(symbol, assetClass, candles) {
    const params = new URLSearchParams({
        symbol,
        asset_class: assetClass,
        candles: JSON.stringify(candles ?? [])
    })

    const response = await fetch(`${API_BASE}/predict?${params.toString()}`, {
        method: 'GET'
    })

    if (!response.ok) {
        throw new Error(`Failed to fetch prediction: ${response.statusText}`)
    }

    // Expected response shape:
    // { predictions: [ { timestamp, upper, lower, median }, ... ] }
    // where predictions.length === Math.floor(Math.log(candles.length))
    return await response.json()
}
