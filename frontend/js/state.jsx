// Global state management using React Context
import React, { createContext, useContext, useState, useCallback } from 'react'
import { fetchHistory, fetchPrediction } from './api'

// Initial state
const initialState = {
    currentSymbol: null,
    assetClass: null,
    historicalData: [],
    predictionData: null,
    isLoading: false,
    error: null
}

// Create context
const AppContext = createContext(null)

// Provider component
export function StateProvider({ children }) {
    const [state, setState] = useState(initialState)

    // Set the current asset and fetch its history
    const setAsset = useCallback(async (symbol, assetClass) => {
        setState(prev => ({
            ...prev,
            currentSymbol: symbol,
            assetClass: assetClass,
            isLoading: true,
            error: null,
            predictionData: null
        }))

        try {
            const data = await fetchHistory(symbol, assetClass)
            setState(prev => ({
                ...prev,
                historicalData: data,
                isLoading: false
            }))
        } catch (err) {
            setState(prev => ({
                ...prev,
                error: err.message,
                isLoading: false
            }))
        }
    }, [])

    // Run prediction for the current asset
    const runPrediction = useCallback(async (horizon = 7) => {
        if (!state.currentSymbol || !state.assetClass) return

        setState(prev => ({ ...prev, isLoading: true, error: null }))

        try {
            const prediction = await fetchPrediction(
                state.currentSymbol,
                state.assetClass,
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
    }, [state.currentSymbol, state.assetClass])

    const value = {
        ...state,
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
