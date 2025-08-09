import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from interfaces.base_components import BaseDataCollector
from utils.plugin_manager import data_source

@data_source("enhanced_binance")
class EnhancedBinanceDataCollector(BaseDataCollector):
    """
    Enhanced Binance data collector with additional features:
    - Multiple timeframe data fetching
    - Order book analysis
    - Liquidation data
    - Funding rate tracking
    """
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Initialize Binance client
        from binance import Client
        self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Base URLs
        self.base_url = "https://fapi.binance.com" if not testnet else "https://testnet.binancefuture.com"
        self.stream_url = "wss://fstream.binance.com" if not testnet else "wss://stream.binancefuture.com"
    
    def get_historical_klines(self, symbol: str, interval: str, lookback_period: int) -> pd.DataFrame:
        """Get historical klines data with enhanced error handling"""
        try:
            # Map interval to milliseconds
            interval_map = {
                '1m': 60000,
                '3m': 180000,
                '5m': 300000,
                '15m': 900000,
                '30m': 1800000,
                '45m': 2700000,
                '1h': 3600000,
                '2h': 7200000,
                '4h': 14400000,
                '6h': 21600000,
                '8h': 28800000,
                '12h': 43200000,
                '1d': 86400000,
                '3d': 259200000,
                '1w': 604800000,
                '1M': 2592000000
            }
            
            interval_ms = interval_map.get(interval)
            if not interval_ms:
                raise ValueError(f"Invalid interval: {interval}")
            
            # Calculate start time
            end_time = int(time.time() * 1000)
            start_time = end_time - (lookback_period * interval_ms)
            
            # Fetch klines
            klines = []
            limit = 1000  # Maximum limit per request
            
            while start_time < end_time:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit
                }
                
                response = self.client._request('get', '/fapi/v1/klines', data=params)
                batch = response.json()
                
                if not batch:
                    break
                
                klines.extend(batch)
                
                # Update start_time to the last timestamp + 1ms
                start_time = batch[-1][6] + 1
                
                # Avoid rate limiting
                time.sleep(0.1)
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 
                        'quote_asset_volume', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume']:
                df[col] = df[col].astype(float)
            
            return df
        except Exception as e:
            print(f"Error fetching historical klines: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, depth: int = 100) -> Dict:
        """Get current order book with enhanced data"""
        try:
            order_book = self.client.futures_order_book(symbol=symbol, limit=depth)
            
            # Calculate order book imbalance
            bid_volume = sum(float(level[1]) for level in order_book['bids'])
            ask_volume = sum(float(level[1]) for level in order_book['asks'])
            total_volume = bid_volume + ask_volume
            
            imbalance = 0
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
            
            # Calculate spread
            if order_book['bids'] and order_book['asks']:
                best_bid = float(order_book['bids'][0][0])
                best_ask = float(order_book['asks'][0][0])
                spread = (best_ask - best_bid) / best_bid
            else:
                spread = 0
            
            # Enhance order book data
            enhanced_order_book = {
                'symbol': symbol,
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'imbalance': imbalance,
                'spread': spread,
                'timestamp': datetime.now()
            }
            
            return enhanced_order_book
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return {}
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List:
        """Get recent trades with enhanced data"""
        try:
            trades = self.client.futures_recent_trades(symbol=symbol, limit=limit)
            
            # Calculate trade statistics
            buy_volume = 0
            sell_volume = 0
            
            for trade in trades:
                if trade['isBuyerMaker']:
                    sell_volume += float(trade['quoteQty'])
                else:
                    buy_volume += float(trade['quoteQty'])
            
            total_volume = buy_volume + sell_volume
            buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
            
            # Enhance trade data
            enhanced_trades = {
                'symbol': symbol,
                'trades': trades,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'total_volume': total_volume,
                'buy_ratio': buy_ratio,
                'timestamp': datetime.now()
            }
            
            return enhanced_trades
        except Exception as e:
            print(f"Error fetching recent trades: {e}")
            return []
    
    def get_funding_rate(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get historical funding rates"""
        try:
            funding_rates = self.client.futures_funding_rate(symbol=symbol, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(funding_rates)
            
            # Convert data types
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = df['fundingRate'].astype(float)
            
            # Calculate moving average of funding rate
            df['fundingRateMA'] = df['fundingRate'].rolling(window=7).mean()
            
            return df
        except Exception as e:
            print(f"Error fetching funding rate: {e}")
            return pd.DataFrame()
    
    def get_liquidation_data(self, symbol: str) -> Dict:
        """Get liquidation data for a symbol"""
        try:
            # Get all liquidations in the last 24 hours
            end_time = int(time.time() * 1000)
            start_time = end_time - 24 * 60 * 60 * 1000
            
            liquidations = self.client.futures_liquidation_orders(
                symbol=symbol,
                startTime=start_time,
                endTime=end_time,
                limit=1000
            )
            
            if not liquidations:
                return {'symbol': symbol, 'liquidations': [], 'timestamp': datetime.now()}
            
            # Calculate liquidation statistics
            long_liquidations = [l for l in liquidations if l['side'] == 'SELL']
            short_liquidations = [l for l in liquidations if l['side'] == 'BUY']
            
            long_volume = sum(float(l['origQty']) * float(l['price']) for l in long_liquidations)
            short_volume = sum(float(l['origQty']) * float(l['price']) for l in short_liquidations)
            
            total_volume = long_volume + short_volume
            long_ratio = long_volume / total_volume if total_volume > 0 else 0.5
            
            # Enhance liquidation data
            enhanced_liquidations = {
                'symbol': symbol,
                'liquidations': liquidations,
                'long_liquidations': len(long_liquidations),
                'short_liquidations': len(short_liquidations),
                'long_volume': long_volume,
                'short_volume': short_volume,
                'total_volume': total_volume,
                'long_ratio': long_ratio,
                'timestamp': datetime.now()
            }
            
            return enhanced_liquidations
        except Exception as e:
            print(f"Error fetching liquidation data: {e}")
            return {'symbol': symbol, 'liquidations': [], 'timestamp': datetime.now()}
    
    def get_open_interest(self, symbol: str) -> pd.DataFrame:
        """Get historical open interest data"""
        try:
            # Get open interest for the last 30 days
            end_time = int(time.time() * 1000)
            start_time = end_time - 30 * 24 * 60 * 60 * 1000
            
            open_interest = self.client.futures_open_interest_hist(
                symbol=symbol,
                period='1d',
                limit=30
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(open_interest)
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['openInterest'] = df['openInterest'].astype(float)
            
            # Calculate percentage change
            df['oi_change_pct'] = df['openInterest'].pct_change() * 100
            
            return df
        except Exception as e:
            print(f"Error fetching open interest: {e}")
            return pd.DataFrame()
    
    def start_realtime_collection(self, symbol: str, interval: str, callback) -> None:
        """Start real-time data collection"""
        try:
            from binance import ThreadedWebsocketManager
            
            # Initialize websocket manager
            twm = ThreadedWebsocketManager(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            # Start kline websocket
            twm.start()
            
            # Define the callback function
            def handle_socket_message(msg):
                if msg['e'] == 'kline':
                    # Process kline message
                    kline = msg['k']
                    
                    # Create a DataFrame row
                    data = {
                        'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'close_time': pd.to_datetime(kline['T'], unit='ms'),
                        'quote_asset_volume': float(kline['q']),
                        'number_of_trades': int(kline['n']),
                        'taker_buy_base_asset_volume': float(kline['V']),
                        'taker_buy_quote_asset_volume': float(kline['Q']),
                        'ignore': 0
                    }
                    
                    # Create a DataFrame with a single row
                    df = pd.DataFrame([data])
                    
                    # Call the callback function
                    callback(df)
            
            # Start the kline socket
            twm.start_kline_socket(
                callback=handle_socket_message,
                symbol=symbol,
                interval=interval
            )
            
            return twm
        except Exception as e:
            print(f"Error starting real-time collection: {e}")
            return None
