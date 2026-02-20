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
export async function fetchPrediction(symbol, assetClass, horizon) {
    const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            symbol: symbol,
            asset_class: assetClass,
            horizon: horizon
        })
    })

    if (!response.ok) {
        throw new Error(`Failed to fetch prediction: ${response.statusText}`)
    }

    return await response.json()
}
