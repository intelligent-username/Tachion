// Global state management using React Context
import React, { createContext, useContext, useState, useCallback } from 'react'
import { fetchHistory, fetchPrediction, sendSearch } from './api'

// Initial state
const initialState = {
    currentSymbol: null,
    assetClass: null,
    timespan: '7d',
    drawMovingAverage: false,
    historicalData: [],
    predictionData: null,
    searchedHistory: {},
    isLoading: false,
    loadingMessage: null,
    error: null
}

// Create context
const AppContext = createContext(null)

// Provider component
export function StateProvider({ children }) {
    const [state, setState] = useState(initialState)

    // Set the current asset and fetch its history
    const setAsset = useCallback(async (symbol, assetClass, timespan = state.timespan) => {
        setState(prev => ({
            ...prev,
            currentSymbol: symbol,
            assetClass: assetClass,
            timespan,
            isLoading: true,
            loadingMessage: 'Collecting Data...',
            error: null,
            predictionData: null
        }))

        try {
            await sendSearch(symbol, assetClass, timespan)
            const data = await fetchHistory(symbol, assetClass, timespan)
            const key = `${assetClass}:${symbol}:${timespan}`
            setState(prev => ({
                ...prev,
                historicalData: data,
                searchedHistory: {
                    ...prev.searchedHistory,
                    [key]: true
                },
                isLoading: false,
                loadingMessage: null
            }))
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err.message,
                isLoading: false,
                loadingMessage: null
            }))
        }
    }, [state.timespan])

    // Run prediction for the current asset
    const runPrediction = useCallback(async (horizon = 7, options = {}) => {
        const targetSymbol = options.symbol ?? state.currentSymbol
        const targetAssetClass = options.assetClass ?? state.assetClass
        const targetTimespan = options.timespan ?? state.timespan
        const ensureSearched = options.ensureSearched ?? true

        if (!targetSymbol || !targetAssetClass) return

        const key = `${targetAssetClass}:${targetSymbol}:${targetTimespan}`

        setState(prev => ({
            ...prev,
            currentSymbol: targetSymbol,
            assetClass: targetAssetClass,
            timespan: targetTimespan,
            isLoading: true,
            loadingMessage: 'Loading...',
            error: null
        }))

        try {
            if (ensureSearched && !state.searchedHistory[key]) {
                await sendSearch(targetSymbol, targetAssetClass, targetTimespan)
                const history = await fetchHistory(targetSymbol, targetAssetClass, targetTimespan)
                setState(prev => ({
                    ...prev,
                    historicalData: history,
                    searchedHistory: {
                        ...prev.searchedHistory,
                        [key]: true
                    }
                }))
            }

            const prediction = await fetchPrediction(
                targetSymbol,
                targetAssetClass,
                horizon
            )
            setState(prev => ({
                ...prev,
                predictionData: prediction,
                isLoading: false,
                loadingMessage: null
            }))
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err.message,
                isLoading: false,
                loadingMessage: null
            }))
        }
    }, [state.currentSymbol, state.assetClass, state.timespan, state.searchedHistory])

    // Update only chart timespan for current symbol/asset without re-running search endpoint
    const setTimespan = useCallback(async (timespan) => {
        if (!timespan) return

        const targetSymbol = state.currentSymbol
        const targetAssetClass = state.assetClass
        const key = `${targetAssetClass}:${targetSymbol}:${timespan}`
        const wasSearched = !!(targetSymbol && targetAssetClass && state.searchedHistory[key])

        setState(prev => ({
            ...prev,
            timespan,
            error: null,
            isLoading: !!(targetSymbol && targetAssetClass),
            loadingMessage: (targetSymbol && targetAssetClass)
                ? (wasSearched ? 'Loading...' : 'Collecting Data...')
                : null
        }))

        if (!targetSymbol || !targetAssetClass) return

        try {
            const data = await fetchHistory(targetSymbol, targetAssetClass, timespan)
            setState(prev => ({
                ...prev,
                historicalData: data,
                searchedHistory: {
                    ...prev.searchedHistory,
                    [key]: true
                },
                isLoading: false,
                loadingMessage: null
            }))
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err.message,
                isLoading: false,
                loadingMessage: null
            }))
        }
    }, [state.currentSymbol, state.assetClass, state.searchedHistory])

    const value = {
        ...state,
        setTimespan,
        setDrawMovingAverage: (drawMovingAverage) => setState(prev => ({ ...prev, drawMovingAverage })),
        setAsset,
        runPrediction
    }

    return (
        <AppContext.Provider value={value}>
            {children}
        </AppContext.Provider>
    )
}

// Custom hook for accessing state
export function useAppState() {
    const context = useContext(AppContext)
    if (!context) {
        throw new Error('useAppState must be used within a StateProvider')
    }
    return context
}
