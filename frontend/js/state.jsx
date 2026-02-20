// Global state management using React Context
import React, { createContext, useContext, useState, useCallback } from 'react'
import { fetchHistory, fetchPrediction, sendSearch } from './api'

// Initial state
const initialState = {
    currentSymbol: null,
    assetClass: null,
    timespan: '7d',
    historicalData: [],
    predictionData: null,
    searchedHistory: {},
    isLoading: false,
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
                isLoading: false
            }))
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err.message,
                isLoading: false
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
                isLoading: false
            }))
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err.message,
                isLoading: false
            }))
        }
    }, [state.currentSymbol, state.assetClass, state.timespan, state.searchedHistory])

    const value = {
        ...state,
        setTimespan: (timespan) => setState(prev => ({ ...prev, timespan })),
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
