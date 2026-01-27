// API calls
const API_BASE = '/api'

// Fetch historical data for a symbol
export async function fetchHistory(symbol, assetClass) {
    const params = new URLSearchParams({
        symbol: symbol,
        asset_class: assetClass
    })

    const response = await fetch(`${API_BASE}/history?${params}`)

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
