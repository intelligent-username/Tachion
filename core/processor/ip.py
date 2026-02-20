"""
Inference processor
This Script handles:
- history refresh
- history slicing
- formatting
- in-memory preprocessing.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from core.apis.biapi import BinanceAPI
from core.apis.oaapi import OandaAPI
from core.apis.tdapi import TwelveDataAPI
from core.apis.yfapi import YFinanceAPI
from core.processor.dw import add_crypto_date_features, add_date_features
from core.processor.lr import log_return, volume_change
from core.processor.ma import moving_average
from core.processor.rv import rolling_volatility

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / 'data'
ASSET_TO_DATA_DIR = {
	'equities': 'equities',
	'crypto': 'crypto',
	'forex': 'forex',
	'comm': 'comm',
	'interest': 'interest',
}

# In-memory cache for ticker-specific preprocessed data.
PREPROCESSED_CACHE: dict[str, dict] = {}
PREPROCESSED_LOCK = Lock()

# In-memory cache for raw ticker history to avoid repeated file reads.
RAW_HISTORY_CACHE: dict[str, dict] = {}
RAW_HISTORY_LOCK = Lock()


class InferenceProcessorError(Exception):
	def __init__(self, detail: str, status_code: int = 500):
		super().__init__(detail)
		self.detail = detail
		self.status_code = status_code


def _normalize_symbol(symbol: str) -> str:
	text = (symbol or '').strip()
	if text.startswith('$'):
		text = text[1:]
	return text


def normalize_interval(interval: str) -> str:
	if not interval:
		return '30m'
	normalized = interval.strip().lower()
	return '30m' if normalized in {'30m', '30min', 'm30'} else '30m'


def parse_timespan_to_days(timespan: str | None) -> float:
	if not timespan or timespan == 'max':
		return 60
	if timespan == 'ytd':
		now = datetime.datetime.now()
		return max(1, (now - datetime.datetime(now.year, 1, 1)).days)

	text = str(timespan).strip().lower()
	hour_match = re.match(r'^(\d+)h$', text)
	day_match = re.match(r'^(\d+)d$', text)
	month_match = re.match(r'^(\d+)m$', text)
	year_match = re.match(r'^(\d+)y$', text)

	if hour_match:
		return max(1, int(hour_match.group(1))) / 24
	if day_match:
		return max(1, int(day_match.group(1)))
	if month_match:
		return max(1, int(month_match.group(1)) * 30)
	if year_match:
		return max(1, int(year_match.group(1)) * 365)
	return 60


def _to_float(value, default=0.0):
	try:
		if value is None:
			return float(default)
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def parse_timestamp(ts) -> datetime.datetime | None:
	if ts is None:
		return None

	if isinstance(ts, datetime.datetime):
		return ts

	text = str(ts).strip()
	if not text:
		return None

	if text.endswith('Z'):
		text = text[:-1] + '+00:00'

	try:
		dt = datetime.datetime.fromisoformat(text)
		if dt.tzinfo is not None:
			dt = dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
		return dt
	except ValueError:
		pass

	for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
		try:
			return datetime.datetime.strptime(text, fmt)
		except ValueError:
			continue

	return None


def _normalize_timestamp_str(ts) -> str | None:
	dt = parse_timestamp(ts)
	if dt is None:
		return None
	return dt.strftime('%Y-%m-%d %H:%M:%S')


def _normalize_ohlcv_row(row: dict) -> dict | None:
	timestamp = row.get('datetime') or row.get('timestamp') or row.get('date') or row.get('time')
	timestamp = _normalize_timestamp_str(timestamp)
	if timestamp is None:
		return None

	close_f = _to_float(row.get('close', row.get('value', None)), np.nan)
	if np.isnan(close_f):
		return None

	open_f = _to_float(row.get('open', close_f), close_f)
	high_f = _to_float(row.get('high', max(open_f, close_f)), max(open_f, close_f))
	low_f = _to_float(row.get('low', min(open_f, close_f)), min(open_f, close_f))
	volume_f = _to_float(row.get('volume', 0.0), 0.0)

	return {
		'datetime': timestamp,
		'open': open_f,
		'high': max(high_f, open_f, close_f),
		'low': min(low_f, open_f, close_f),
		'close': close_f,
		'volume': volume_f,
	}


def read_raw_data(path: Path) -> list[dict]:
	if not path.exists():
		return []
	try:
		with path.open('r', encoding='utf-8') as f:
			rows = json.load(f)
		if not isinstance(rows, list):
			return []
		normalized = []
		for row in rows:
			if not isinstance(row, dict):
				continue
			n = _normalize_ohlcv_row(row)
			if n is not None:
				normalized.append(n)
		normalized.sort(key=lambda r: r['datetime'])
		return normalized
	except Exception:
		return []


def write_raw_data(path: Path, values: list[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8') as f:
		json.dump(values, f, indent=2)


def raw_file_path(asset_class: str, symbol: str) -> Path:
	asset_key = (asset_class or '').strip().lower()
	if asset_key not in ASSET_TO_DATA_DIR:
		raise InferenceProcessorError(f'Unknown asset class: {asset_class}', status_code=400)

	ticker_name = _normalize_symbol(symbol).replace('/', '_')
	if not ticker_name:
		raise InferenceProcessorError('Missing symbol/ticker', status_code=400)

	return DATA_ROOT / ASSET_TO_DATA_DIR[asset_key] / 'raw' / f'{ticker_name}.json'


def format_ohlcv(values: list[dict]) -> list[dict]:
	formatted = []
	for v in values:
		timestamp = v.get('datetime') or v.get('timestamp') or v.get('date')
		close = v.get('close', v.get('value'))
		if timestamp is None or close is None:
			continue

		close_f = _to_float(close, 0.0)
		open_f = _to_float(v.get('open', close_f), close_f)
		high_f = _to_float(v.get('high', max(open_f, close_f)), max(open_f, close_f))
		low_f = _to_float(v.get('low', min(open_f, close_f)), min(open_f, close_f))
		volume_f = _to_float(v.get('volume', 0), 0.0)

		formatted.append({
			'timestamp': str(timestamp),
			'open': open_f,
			'high': max(high_f, open_f, close_f),
			'low': min(low_f, open_f, close_f),
			'close': close_f,
			'volume': volume_f,
			'value': close_f,
		})

	return formatted


def _fetch_from_yahoo(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, interval: str) -> dict:
	return YFinanceAPI(symbol=symbol, start_date=start_date, end_date=end_date, interval=interval)


def _fetch_from_twelvedata(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> dict:
	try:
		raw = TwelveDataAPI(
			symbol=symbol,
			interval='30min',
			start_date=start_date,
			end_date=end_date,
			format='JSON'
		)
	except Exception as e:
		return {'status': 'error', 'message': str(e)}

	if isinstance(raw, dict) and isinstance(raw.get('values'), list):
		values = list(reversed(raw.get('values', [])))
		return {'status': 'ok', 'values': values}

	if isinstance(raw, dict):
		message = (raw.get('message') or '').lower()
		# Not an upstream failure: just means no new candles in requested window.
		if 'no data is available' in message or 'no data' in message:
			return {'status': 'ok', 'values': []}
		return {'status': 'error', 'message': raw.get('message') or raw.get('status') or 'Unknown TwelveData error'}

	return {'status': 'error', 'message': 'Unexpected TwelveData response'}


def slice_values_by_timespan(values: list[dict], timespan: str, buffer_days: int = 0) -> list[dict]:
	if not values:
		return []

	parsed = []
	for v in values:
		dt = parse_timestamp(v.get('datetime') or v.get('timestamp') or v.get('date'))
		if dt is None:
			continue
		parsed.append((dt, v))

	if not parsed:
		return []

	parsed.sort(key=lambda x: x[0])
	if not timespan or timespan == 'max':
		if buffer_days <= 0:
			return [v for _, v in parsed]
		end = parsed[-1][0]
		start = end - datetime.timedelta(days=buffer_days)
		return [v for dt, v in parsed if dt >= start]

	end_dt = parsed[-1][0]
	days = parse_timespan_to_days(timespan) + max(0, buffer_days)
	start_dt = end_dt - datetime.timedelta(days=days)
	sliced = [v for dt, v in parsed if start_dt <= dt <= end_dt]
	return sliced if sliced else [v for _, v in parsed]


def _merge_values(existing: list[dict], incoming: list[dict]) -> list[dict]:
	merged: dict[str, dict] = {}
	for row in existing:
		n = _normalize_ohlcv_row(row)
		if n is not None:
			merged[n['datetime']] = n
	for row in incoming:
		n = _normalize_ohlcv_row(row)
		if n is not None:
			merged[n['datetime']] = n
	ordered_keys = sorted(merged.keys())
	return [merged[k] for k in ordered_keys]


def _fetch_incremental_values(
	symbol: str,
	asset_class: str,
	start_date: datetime.datetime,
	end_date: datetime.datetime,
	interval: str,
) -> list[dict]:
	normalized_asset = (asset_class or '').strip().lower()
	normalized_interval = normalize_interval(interval)

	result = None

	if normalized_asset == 'equities':
		result = _fetch_from_twelvedata(symbol, start_date, end_date)
		if result.get('status') == 'error':
			result = _fetch_from_yahoo(symbol, start_date, end_date, normalized_interval)

	elif normalized_asset == 'crypto':
		start_ms = int(start_date.timestamp() * 1000)
		result = BinanceAPI(symbol=symbol, interval='30m', limit=1000, start_time=start_ms)

	elif normalized_asset in {'forex', 'comm'}:
		result = OandaAPI(instrument=symbol, granularity='M30', count=5000, from_time=start_date)
		if result.get('status') == 'error':
			yahoo_symbol = symbol
			if normalized_asset == 'forex' and '_' in symbol:
				base, quote = symbol.split('_', 1)
				yahoo_symbol = f'{base}{quote}=X'
			result = _fetch_from_yahoo(yahoo_symbol, start_date, end_date, normalized_interval)

	elif normalized_asset == 'interest':
		result = _fetch_from_yahoo(symbol, start_date, end_date, normalized_interval)

	else:
		raise InferenceProcessorError(f'Unknown asset class: {asset_class}', status_code=400)

	if not result:
		raise InferenceProcessorError('No response from data provider', status_code=500)

	if result.get('status') == 'error':
		raise InferenceProcessorError(result.get('message', 'Upstream API error'), status_code=502)

	values = result.get('values', [])
	if not isinstance(values, list):
		return []
	return values


def _build_preprocessed_frame(asset_class: str, symbol: str, rows: list[dict]) -> pd.DataFrame:
	if not rows:
		return pd.DataFrame()

	df = pd.DataFrame(rows)
	if df.empty:
		return df

	df['datetime'] = pd.to_datetime(df['datetime'])
	df = df.sort_values('datetime').reset_index(drop=True)
	df['close'] = pd.to_numeric(df['close'], errors='coerce')
	df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce').fillna(0)
	df = df.dropna(subset=['close'])

	if df.empty:
		return df

	normalized_asset = (asset_class or '').strip().lower()

	df['log_return'] = log_return(df['close'])
	df['log_return_lag1'] = df['log_return'].shift(1)

	if normalized_asset == 'equities':
		df['volume_change'] = volume_change(df['volume'].replace(0, np.nan).ffill())
		df['5_day_MA'] = moving_average(df['close'], window=5)
		df['50_day_MA'] = moving_average(df['close'], window=50)
		df['rolling_volatility_5'] = rolling_volatility(df['log_return'], window=5)
		df['rolling_volatility_50'] = rolling_volatility(df['log_return'], window=50)
		df = add_date_features(df, date_col='datetime')
		df['ticker'] = symbol
		columns = [
			'ticker', 'datetime', 'log_return', 'log_return_lag1', 'volume_change',
			'5_day_MA', '50_day_MA', 'rolling_volatility_5', 'rolling_volatility_50',
			'day_of_week', 'day_of_month', 'quarter'
		]

	elif normalized_asset == 'crypto':
		df['volume_change'] = volume_change(df['volume'].replace(0, np.nan).ffill())
		df['5_period_MA'] = moving_average(df['close'], window=5)
		df['20_period_MA'] = moving_average(df['close'], window=20)
		df['rolling_volatility_5'] = rolling_volatility(df['log_return'], window=5)
		df['rolling_volatility_20'] = rolling_volatility(df['log_return'], window=20)
		df = add_crypto_date_features(df, date_col='datetime')
		df['symbol'] = symbol
		columns = [
			'symbol', 'datetime', 'log_return', 'log_return_lag1', 'volume_change',
			'5_period_MA', '20_period_MA', 'rolling_volatility_5', 'rolling_volatility_20',
			'hour_of_day', 'day_of_week', 'day_of_month', 'is_weekend'
		]

	elif normalized_asset in {'forex', 'comm'}:
		df['MA_50'] = moving_average(df['close'], window=50)
		df['MA_200'] = moving_average(df['close'], window=200)
		df['rolling_vol_50'] = rolling_volatility(df['log_return'], window=50)
		df['rolling_vol_200'] = rolling_volatility(df['log_return'], window=200)
		df = add_date_features(df, date_col='datetime')
		df['symbol'] = symbol
		columns = [
			'symbol', 'datetime', 'log_return', 'log_return_lag1',
			'MA_50', 'MA_200', 'rolling_vol_50', 'rolling_vol_200',
			'day_of_week', 'day_of_month', 'quarter'
		]

	else:
		df = add_date_features(df, date_col='datetime')
		df['symbol'] = symbol
		columns = ['symbol', 'datetime', 'log_return', 'log_return_lag1', 'day_of_week', 'day_of_month', 'quarter']

	out = df[columns].copy()
	out.rename(columns={'datetime': 'timestamp'}, inplace=True)
	out = out.replace([np.inf, -np.inf], np.nan)
	return out


def background_preprocess_ticker(asset_class: str, symbol: str, full_rows: list[dict], timespan: str) -> None:
	buffered_rows = slice_values_by_timespan(full_rows, timespan, buffer_days=7)
	frame = _build_preprocessed_frame(asset_class, symbol, buffered_rows)
	records = [] if frame.empty else frame.where(pd.notna(frame), None).to_dict(orient='records')

	cache_key = f"{asset_class.lower()}:{symbol}"
	with PREPROCESSED_LOCK:
		PREPROCESSED_CACHE[cache_key] = {
			'asset_class': asset_class.lower(),
			'symbol': symbol,
			'timespan': timespan,
			'updated_at': datetime.datetime.utcnow().isoformat() + 'Z',
			'rows': records,
		}


def get_historical_data(symbol: str, asset_class: str, timespan: str = 'max', interval: str = '30m') -> list[dict]:
	"""Update raw file from last stored point to now and return full normalized rows."""
	symbol = _normalize_symbol(symbol)
	raw_path = raw_file_path(asset_class, symbol)
	cache_key = f"{(asset_class or '').strip().lower()}:{(symbol or '').strip()}"
	with RAW_HISTORY_LOCK:
		cached = RAW_HISTORY_CACHE.get(cache_key)

	if cached is not None:
		existing_values = cached.get('rows', [])
	else:
		existing_values = read_raw_data(raw_path)
		with RAW_HISTORY_LOCK:
			RAW_HISTORY_CACHE[cache_key] = {
				'rows': existing_values,
				'loaded_at': datetime.datetime.utcnow(),
			}

	now = datetime.datetime.now()
	last_dt = parse_timestamp(existing_values[-1].get('datetime')) if existing_values else None
	should_refresh = (last_dt is None) or ((now - last_dt) >= datetime.timedelta(minutes=25))
	if not should_refresh:
		return existing_values

	if existing_values:
		start_date = (last_dt + datetime.timedelta(minutes=30)) if last_dt else (now - datetime.timedelta(days=30))
	else:
		bootstrap_days = max(1, parse_timespan_to_days(timespan))
		start_date = now - datetime.timedelta(days=bootstrap_days)

	try:
		incoming_values = _fetch_incremental_values(symbol, asset_class, start_date, now, interval)
	except InferenceProcessorError:
		if existing_values:
			return existing_values
		raise
	merged_values = _merge_values(existing_values, incoming_values)

	if merged_values:
		write_raw_data(raw_path, merged_values)
		with RAW_HISTORY_LOCK:
			RAW_HISTORY_CACHE[cache_key] = {
				'rows': merged_values,
				'loaded_at': datetime.datetime.utcnow(),
			}

	return merged_values

