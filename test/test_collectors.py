"""
Unit tests for data collectors (crypto, equities, forex)
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open


# ============================================================================
# Crypto Collector Tests
# ============================================================================

class TestCryptoCollector:
    """Tests for data/crypto/collector.py"""

    # --- Basic Cases (4) ---

    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_calls_binance_api(self, mock_binance):
        """Test that write_data calls the Binance API"""
        from data.crypto.collector import write_data
        
        write_data(["BTC", "ETH"])
        
        mock_binance.assert_called_once()

    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_passes_correct_num_calls(self, mock_binance):
        """Test that write_data uses 87 calls for ~5 years of data"""
        from data.crypto.collector import write_data
        
        write_data(["BTC"])
        
        call_args = mock_binance.call_args
        assert call_args[1]["num_calls"] == 87

    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_passes_symbols(self, mock_binance):
        """Test that write_data passes symbols correctly"""
        from data.crypto.collector import write_data
        
        symbols = ["BTC", "ETH", "SOL"]
        write_data(symbols)
        
        call_args = mock_binance.call_args
        assert call_args[1]["symbols"] == symbols

    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_creates_correct_path(self, mock_binance):
        """Test that write_data uses correct output path"""
        from data.crypto.collector import write_data
        
        write_data(["BTC"])
        
        call_args = mock_binance.call_args
        assert "crypto" in call_args[0][0]
        assert "raw" in call_args[0][0]

    # --- Edge Cases (3) ---

    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_empty_list(self, mock_binance):
        """Test write_data handles empty list"""
        from data.crypto.collector import write_data
        
        write_data([])
        
        mock_binance.assert_called_once()
        assert mock_binance.call_args[1]["symbols"] == []

    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_single_symbol(self, mock_binance):
        """Test write_data handles single symbol"""
        from data.crypto.collector import write_data
        
        write_data(["BTC"])
        
        mock_binance.assert_called_once()

    @patch('data.crypto.collector.os.makedirs')
    @patch('data.crypto.collector.call_specific_binance')
    def test_write_data_creates_directory(self, mock_binance, mock_makedirs):
        """Test that write_data creates output directory"""
        from data.crypto.collector import write_data
        
        write_data(["BTC"])
        
        mock_makedirs.assert_called_once()


# ============================================================================
# Equities Collector Tests
# ============================================================================

class TestEquitiesCollector:
    """Tests for data/equities/collector.py"""

    # --- Basic Cases (4) ---

    @patch('data.equities.collector.call_specific_td')
    def test_write_data_calls_td_api(self, mock_td):
        """Test that write_data calls TwelveData API"""
        from data.equities.collector import write_data
        
        write_data(["AAPL"])
        
        # Called twice: once for SPY, once for symbols
        assert mock_td.call_count == 2

    @patch('data.equities.collector.call_specific_td')
    def test_write_data_includes_spy(self, mock_td):
        """Test that write_data always includes SPY"""
        from data.equities.collector import write_data
        
        write_data(["AAPL", "MSFT"])
        
        # First call should be for SPY
        first_call = mock_td.call_args_list[0]
        assert first_call[1]["symbols"] == ["SPY"]

    @patch('data.equities.collector.call_specific_td')
    def test_write_data_passes_correct_num_calls(self, mock_td):
        """Test that write_data uses 3 calls for ~5 years"""
        from data.equities.collector import write_data
        
        write_data(["AAPL"])
        
        for call in mock_td.call_args_list:
            assert call[1]["num_calls"] == 3

    @patch('data.equities.collector.call_specific_td')
    def test_write_data_passes_symbols(self, mock_td):
        """Test that write_data passes symbols correctly"""
        from data.equities.collector import write_data
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        write_data(symbols)
        
        # Second call should have our symbols
        second_call = mock_td.call_args_list[1]
        assert second_call[1]["symbols"] == symbols

    # --- Edge Cases (3) ---

    @patch('data.equities.collector.call_specific_td')
    def test_write_data_empty_list_still_gets_spy(self, mock_td):
        """Test that empty list still fetches SPY"""
        from data.equities.collector import write_data
        
        write_data([])
        
        # Should still call for SPY
        assert mock_td.call_count == 2

    @patch('data.equities.collector.call_specific_td')
    def test_write_data_creates_correct_path(self, mock_td):
        """Test that write_data uses correct output path"""
        from data.equities.collector import write_data
        
        write_data(["AAPL"])
        
        call_args = mock_td.call_args_list[0]
        assert "equities" in call_args[0][0]
        assert "raw" in call_args[0][0]

    @patch('data.equities.collector.os.makedirs')
    @patch('data.equities.collector.call_specific_td')
    def test_write_data_creates_directory(self, mock_td, mock_makedirs):
        """Test that write_data creates output directory"""
        from data.equities.collector import write_data
        
        write_data(["AAPL"])
        
        mock_makedirs.assert_called_once()


# ============================================================================
# Forex Collector Tests
# ============================================================================

class TestForexCollector:
    """Tests for data/forex/collector.py"""

    # --- Basic Cases (4) ---

    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_calls_oanda_api(self, mock_oanda):
        """Test that write_data calls OANDA API"""
        from data.forex.collector import write_data
        
        write_data(["EUR_USD"])
        
        mock_oanda.assert_called_once()

    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_passes_correct_num_calls(self, mock_oanda):
        """Test that write_data uses 35 calls for ~10 years"""
        from data.forex.collector import write_data
        
        write_data(["EUR_USD"])
        
        call_args = mock_oanda.call_args
        assert call_args[1]["num_calls"] == 35

    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_passes_instruments(self, mock_oanda):
        """Test that write_data passes instruments correctly"""
        from data.forex.collector import write_data
        
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
        write_data(instruments)
        
        call_args = mock_oanda.call_args
        assert call_args[1]["instruments"] == instruments

    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_creates_correct_path(self, mock_oanda):
        """Test that write_data uses correct output path"""
        from data.forex.collector import write_data
        
        write_data(["EUR_USD"])
        
        call_args = mock_oanda.call_args
        assert "forex" in call_args[0][0]
        assert "raw" in call_args[0][0]

    # --- Edge Cases (3) ---

    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_empty_list(self, mock_oanda):
        """Test write_data handles empty list"""
        from data.forex.collector import write_data
        
        write_data([])
        
        mock_oanda.assert_called_once()
        assert mock_oanda.call_args[1]["instruments"] == []

    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_single_instrument(self, mock_oanda):
        """Test write_data handles single instrument"""
        from data.forex.collector import write_data
        
        write_data(["EUR_USD"])
        
        mock_oanda.assert_called_once()

    @patch('data.forex.collector.os.makedirs')
    @patch('data.forex.collector.call_specific_oanda')
    def test_write_data_creates_directory(self, mock_oanda, mock_makedirs):
        """Test that write_data creates output directory"""
        from data.forex.collector import write_data
        
        write_data(["EUR_USD"])
        
        mock_makedirs.assert_called_once()
