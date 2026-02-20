// Sidebar component - Search and Predict controls
import { useState } from 'react'
import { useAppState } from '../js/state'

export default function Sidebar() {
    const [searchTerm, setSearchTerm] = useState('')
    const [searchResults, setSearchResults] = useState([])
    const [assetClass, setAssetClass] = useState('equities')
    const { currentSymbol, isLoading, setAsset, runPrediction } = useAppState()

    const handleSearch = () => {
        if (!searchTerm.trim()) return
        setAsset(searchTerm, assetClass)
    }

    const handlePredict = () => {
        runPrediction(7) // Default horizon of 7 periods
    }

    const placeholders = {
        'equities': 'e.g. NVDA',
        'crypto': 'e.g. ETH',
        'forex': 'e.g. EUR_USD',
        'comm': 'e.g. GOLD',
        'interest': ''
    }

    return (
        <div className="sidebar">
            <div className="search-section">
                <div className="search-bar-wrapper">
                    <input
                        id="symbol-input"
                        type="text"
                        placeholder={placeholders[assetClass]}
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        disabled={isLoading}
                    />
                    <svg
                        className="search-icon"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        onClick={handleSearch}
                    >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                </div>

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

                <select
                    id="asset-select"
                    value={assetClass}
                    onChange={(e) => setAssetClass(e.target.value)}
                    disabled={isLoading}
                    className="asset-select"
                >
                    <option value="equities">Equities</option>
                    <option value="forex">Forex</option>
                    <option value="comm">Commodities</option>
                    <option value="crypto">Crypto</option>
                    <option value="interest">Interest Rates</option>
                </select>
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
