import pandas as pd
import numpy as np
from binance import Client, ThreadedWebsocketManager
import websocket
import json
import time
from datetime import datetime, timedelta
import logging
from utils.database import TradingDatabase

logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self, api_key, api_secret, db_path='data/trading_bot.db'):
        self.client = Client(api_key, api_secret)
        self.twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        self.twm.start()
        self.db = TradingDatabase(db_path)
        
    def get_historical_klines(self, symbol, interval, lookback_period, use_db=True):
        """Get historical klines/candlesticks, using database cache if available"""
        if use_db:
            # Check if we have recent data in the database
            latest_timestamp = self.db.get_latest_kline_timestamp(symbol, interval)
            
            # Calculate the start time for new data
            if latest_timestamp:
                # Add a small buffer to avoid duplicates
                start_time = latest_timestamp + 60000  # 1 minute in milliseconds
                start_str = datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Latest data in DB for {symbol} {interval}: {start_str}")
            else:
                start_time = None
                logger.info(f"No data in DB for {symbol} {interval}, fetching all data")
            
            # Fetch new data from Binance
            new_klines = self._fetch_new_klines(symbol, interval, lookback_period, start_time)
            
            if new_klines:
                # Store new data in database
                self.db.store_klines(symbol, interval, new_klines)
                logger.info(f"Stored {len(new_klines)} new klines for {symbol} {interval}")
            
            # Get all data from database
            df = self.db.get_klines(symbol, interval, limit=lookback_period)
            
            if not df.empty:
                return df
        
        # Fallback to direct API call if database is empty or disabled
        logger.info(f"Fetching data directly from API for {symbol} {interval}")
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=lookback_period
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def _fetch_new_klines(self, symbol, interval, limit, start_time=None):
        """Fetch new klines from Binance API"""
        klines = []
        
        if start_time:
            # Calculate end time (now)
            end_time = int(datetime.now().timestamp() * 1000)
            
            # Binance has a limit of 1000 klines per request
            # So we may need to make multiple requests
            while start_time < end_time:
                logger.info(f"Fetching klines from {datetime.fromtimestamp(start_time/1000)}")
                
                batch = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_time,
                    endTime=end_time,
                    limit=1000
                )
                
                if not batch:
                    break
                
                klines.extend(batch)
                
                # Update start_time to the last timestamp + 1ms
                start_time = batch[-1][6] + 1  # close_time + 1ms
                
                # Avoid rate limiting
                time.sleep(0.1)
        else:
            # Fetch all klines
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
        
        # Convert to list of dictionaries
        result = []
        for k in klines:
            result.append({
                'timestamp': k[0],
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'close_time': k[6],
                'quote_asset_volume': float(k[7]),
                'number_of_trades': int(k[8]),
                'taker_buy_base_asset_volume': float(k[9]),
                'taker_buy_quote_asset_volume': float(k[10])
            })
        
        return result
    
    def start_realtime_collection(self, symbol, interval, callback):
        """Start real-time data collection"""
        self.twm.start_kline_socket(
            callback=callback,
            symbol=symbol,
            interval=interval
        )
        
    def get_order_book(self, symbol):
        """Get current order book"""
        return self.client.get_order_book(symbol=symbol)
    
    def get_recent_trades(self, symbol):
        """Get recent trades"""
        return self.client.get_recent_trades(symbol=symbol)
    
    def close(self):
        """Close connections"""
        self.twm.stop()
        self.db.close()
