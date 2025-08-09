import sqlite3
import pandas as pd
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingDatabase:
    def __init__(self, db_path='data/trading_bot.db'):
        """Initialize the database connection"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create klines table for historical price data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS klines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER,
            quote_asset_volume REAL,
            number_of_trades INTEGER,
            taker_buy_base_asset_volume REAL,
            taker_buy_quote_asset_volume REAL,
            UNIQUE(symbol, interval, timestamp)
        )
        ''')
        
        # Create signals table for generated trading signals
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            entry_price REAL,
            stop_loss REAL,
            take_profit_levels TEXT,  -- JSON array
            tp_probabilities TEXT,     -- JSON array
            sl_probability REAL,
            confidence REAL,
            rationale TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            timeframe TEXT,
            is_executed BOOLEAN DEFAULT FALSE,
            is_paper_trade BOOLEAN DEFAULT FALSE
        )
        ''')
        
        # Create trades table for executed trades
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER,
            symbol TEXT NOT NULL,
            position_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            quantity REAL NOT NULL,
            stop_loss REAL,
            take_profit_levels TEXT,  -- JSON array
            entry_timestamp DATETIME,
            exit_price REAL,
            exit_timestamp DATETIME,
            pnl REAL,
            leverage INTEGER,
            risk_amount REAL,
            is_paper_trade BOOLEAN DEFAULT FALSE,
            status TEXT DEFAULT 'OPEN',
            FOREIGN KEY (signal_id) REFERENCES signals (id)
        )
        ''')
        
        # Create sentiment_data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            overall_sentiment REAL,
            twitter_sentiment REAL,
            news_sentiment REAL,
            perplexity_insights TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval ON klines (symbol, interval)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_klines_timestamp ON klines (timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp)')
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def store_klines(self, symbol, interval, klines_data):
        """Store klines data in the database"""
        cursor = self.conn.cursor()
        
        # Prepare data for insertion
        records = []
        for kline in klines_data:
            records.append((
                symbol, interval,
                kline['timestamp'], kline['open'], kline['high'], 
                kline['low'], kline['close'], kline['volume'],
                kline.get('close_time'), kline.get('quote_asset_volume'),
                kline.get('number_of_trades'), kline.get('taker_buy_base_asset_volume'),
                kline.get('taker_buy_quote_asset_volume')
            ))
        
        # Insert or replace if exists
        cursor.executemany('''
        INSERT OR REPLACE INTO klines (
            symbol, interval, timestamp, open, high, low, close, volume,
            close_time, quote_asset_volume, number_of_trades,
            taker_buy_base_asset_volume, taker_buy_quote_asset_volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        self.conn.commit()
        logger.info(f"Stored {len(records)} klines records for {symbol} {interval}")
    
    def get_klines(self, symbol, interval, limit=None, start_time=None, end_time=None):
        """Retrieve klines data from the database"""
        cursor = self.conn.cursor()
        
        query = '''
        SELECT timestamp, open, high, low, close, volume,
               close_time, quote_asset_volume, number_of_trades,
               taker_buy_base_asset_volume, taker_buy_quote_asset_volume
        FROM klines
        WHERE symbol = ? AND interval = ?
        '''
        
        params = [symbol, interval]
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume', 
                    'quote_asset_volume', 'taker_buy_base_asset_volume', 
                    'taker_buy_quote_asset_volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    def get_latest_kline_timestamp(self, symbol, interval):
        """Get the timestamp of the latest kline for a symbol and interval"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT MAX(timestamp) FROM klines
        WHERE symbol = ? AND interval = ?
        ''', (symbol, interval))
        
        result = cursor.fetchone()
        return result[0] if result[0] is not None else 0
    
    def store_signal(self, signal):
        """Store a trading signal"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO signals (
            symbol, signal_type, entry_price, stop_loss, take_profit_levels,
            tp_probabilities, sl_probability, confidence, rationale, timeframe,
            is_executed, is_paper_trade
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'], signal['signal_type'], signal['entry_price'],
            signal['stop_loss'], json.dumps(signal['take_profit_levels']),
            json.dumps(signal['tp_probabilities']), signal['sl_probability'],
            signal['confidence'], signal['rationale'], signal['timeframe'],
            signal.get('is_executed', False), signal.get('is_paper_trade', False)
        ))
        
        signal_id = cursor.lastrowid
        self.conn.commit()
        
        return signal_id
    
    def store_trade(self, trade, signal_id=None):
        """Store a trade record"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO trades (
            signal_id, symbol, position_type, entry_price, quantity,
            stop_loss, take_profit_levels, entry_timestamp, exit_price,
            exit_timestamp, pnl, leverage, risk_amount, is_paper_trade, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_id, trade['symbol'], trade['position_type'],
            trade['entry_price'], trade['quantity'], trade['stop_loss'],
            json.dumps(trade['take_profit_levels']), trade.get('entry_timestamp'),
            trade.get('exit_price'), trade.get('exit_timestamp'), trade.get('pnl'),
            trade.get('leverage'), trade.get('risk_amount'),
            trade.get('is_paper_trade', False), trade.get('status', 'OPEN')
        ))
        
        trade_id = cursor.lastrowid
        self.conn.commit()
        
        return trade_id
    
    def update_trade(self, trade_id, exit_price=None, exit_timestamp=None, pnl=None, status=None):
        """Update a trade record"""
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if exit_price is not None:
            updates.append("exit_price = ?")
            params.append(exit_price)
        
        if exit_timestamp is not None:
            updates.append("exit_timestamp = ?")
            params.append(exit_timestamp)
        
        if pnl is not None:
            updates.append("pnl = ?")
            params.append(pnl)
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        
        if not updates:
            return False
        
        params.append(trade_id)
        
        query = f"UPDATE trades SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def store_sentiment_data(self, symbol, sentiment_data):
        """Store sentiment data"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO sentiment_data (
            symbol, overall_sentiment, twitter_sentiment, news_sentiment, perplexity_insights
        ) VALUES (?, ?, ?, ?, ?)
        ''', (
            symbol, sentiment_data.get('overall_sentiment'),
            sentiment_data.get('twitter_sentiment', {}).get('average_sentiment'),
            sentiment_data.get('news_sentiment_avg'),
            sentiment_data.get('perplexity_insights')
        ))
        
        self.conn.commit()
    
    def get_recent_signals(self, symbol=None, limit=10):
        """Get recent trading signals"""
        cursor = self.conn.cursor()
        
        query = '''
        SELECT * FROM signals
        '''
        
        params = []
        if symbol:
            query += ' WHERE symbol = ?'
            params.append(symbol)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        signals = []
        for row in rows:
            signal = dict(zip(columns, row))
            # Parse JSON fields
            if signal['take_profit_levels']:
                signal['take_profit_levels'] = json.loads(signal['take_profit_levels'])
            if signal['tp_probabilities']:
                signal['tp_probabilities'] = json.loads(signal['tp_probabilities'])
            signals.append(signal)
        
        return signals
    
    def get_trade_history(self, symbol=None, limit=100):
        """Get trade history"""
        cursor = self.conn.cursor()
        
        query = '''
        SELECT * FROM trades
        '''
        
        params = []
        if symbol:
            query += ' WHERE symbol = ?'
            params.append(symbol)
        
        query += ' ORDER BY entry_timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        trades = []
        for row in rows:
            trade = dict(zip(columns, row))
            # Parse JSON fields
            if trade['take_profit_levels']:
                trade['take_profit_levels'] = json.loads(trade['take_profit_levels'])
            trades.append(trade)
        
        return trades
    
    def get_performance_metrics(self, symbol=None):
        """Calculate performance metrics from trade history"""
        trades = self.get_trade_history(symbol=symbol, limit=10000)
        
        if not trades:
            return {}
        
        # Convert to DataFrame for easier calculations
        df = pd.DataFrame(trades)
        
        # Filter only closed trades with PnL
        closed_trades = df[df['status'] == 'CLOSED']
        
        if closed_trades.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = closed_trades['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def close(self):
        """Close the database connection"""
        self.conn.close()
