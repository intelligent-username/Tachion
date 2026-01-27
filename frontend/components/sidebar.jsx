// Sidebar component - Search and Predict controls
import { useState } from 'react'
import { useAppState } from '../js/state'

// Infer asset class from symbol format
function inferAssetClass(symbol) {
    const upper = symbol.toUpperCase()

    // Crypto pairs typically have USDT/USD suffix
    if (upper.includes('USDT') || upper.includes('BTC') || upper.includes('ETH')) {
        return 'crypto'
    }

    // Forex pairs have underscore (e.g., EUR_USD)
    if (upper.includes('_')) {
        return 'forex'
    }

    // Commodities
    if (['XAU', 'XAG', 'OIL', 'GOLD', 'SILVER'].some(c => upper.includes(c))) {
        return 'comm'
    }

    // Default to equities
    return 'equities'
}

export default function Sidebar() {
    const [searchTerm, setSearchTerm] = useState('')
    const [searchResults, setSearchResults] = useState([])
    const { currentSymbol, isLoading, setAsset, runPrediction } = useAppState()

    const handleSearch = () => {
        if (!searchTerm.trim()) return

        // For now, just use the search term directly
        // In production, this would call an external API for autocomplete
        const assetClass = inferAssetClass(searchTerm)
        setAsset(searchTerm.toUpperCase(), assetClass)
    }

    const handlePredict = () => {
        runPrediction(7) // Default horizon of 7 periods
    }

    return (
        <div className="sidebar">
            <div className="search-section">
                <label htmlFor="symbol-input">Search bar</label>
                <input
                    id="symbol-input"
                    type="text"
                    placeholder="Symbol"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    disabled={isLoading}
                />

                {searchResults.length > 0 && (
                    <div className="search-results">
                        {searchResults.map((result, i) => (
                            <div
                                key={i}
                                className="search-result"
                                onClick={() => {
                                    setSearchTerm(result)
                                    setSearchResults([])
                                }}
                            >
                                {result}
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <button
                className="predict-button"
                onClick={handlePredict}
                disabled={!currentSymbol || isLoading}
            >
                <span className="button-text">Predict!</span>
                <span className="button-subtitle">
                    (Model will return a 95% CI)
                </span>
            </button>
        </div>
    )
}
