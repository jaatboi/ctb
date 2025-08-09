import pandas as pd
import numpy as np
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from utils.plugin_manager import PluginManager

class TechnicalAnalyzer:
    def __init__(self, plugin_manager: PluginManager = None):
        self.plugin_manager = plugin_manager or PluginManager()
    
    def add_all_indicators(self, df):
        """Add all technical indicators to the dataframe"""
        # Momentum indicators
        try:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['slowk'], df['slowd'] = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowd_period=3)
        except:
            # Fallback implementations if talib is not available
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['macd'], df['macdsignal'], df['macdhist'] = self._calculate_macd(df['close'])
            df['slowk'], df['slowd'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Trend indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Volatility indicators
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Add custom indicators from plugins
        for indicator_name in self.plugin_manager.list_indicators():
            try:
                indicator_func = self.plugin_manager.get_indicator(indicator_name)
                result = indicator_func(df)
                
                if isinstance(result, pd.Series):
                    df[indicator_name] = result
                elif isinstance(result, tuple):
                    for i, series in enumerate(result):
                        if isinstance(series, pd.Series):
                            df[f"{indicator_name}_{i}"] = series
            except Exception as e:
                print(f"Error adding indicator {indicator_name}: {e}")
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def identify_support_resistance(self, df, window=20, tolerance=0.02):
        """Identify support and resistance levels"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Find resistance levels (local highs)
        resistance_indices = []
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                # Check if this is a significant high (within tolerance of neighbors)
                is_resistance = True
                for j in range(i - window, i + window + 1):
                    if j != i and abs(df['high'].iloc[i] - df['high'].iloc[j]) / df['high'].iloc[i] < tolerance:
                        is_resistance = False
                        break
                if is_resistance:
                    resistance_indices.append(i)
        
        # Find support levels (local lows)
        support_indices = []
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == lows.iloc[i]:
                # Check if this is a significant low (within tolerance of neighbors)
                is_support = True
                for j in range(i - window, i + window + 1):
                    if j != i and abs(df['low'].iloc[i] - df['low'].iloc[j]) / df['low'].iloc[i] < tolerance:
                        is_support = False
                        break
                if is_support:
                    support_indices.append(i)
        
        resistance_levels = df['high'].iloc[resistance_indices].tolist()
        support_levels = df['low'].iloc[support_indices].tolist()
        
        return support_levels, resistance_levels
    
    def detect_candlestick_patterns(self, df):
        """Detect candlestick patterns"""
        patterns = {}
        
        # Single candlestick patterns
        try:
            patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        except:
            # Fallback implementations
            patterns['doji'] = self._detect_doji(df)
            patterns['hammer'] = self._detect_hammer(df)
            patterns['shooting_star'] = self._detect_shooting_star(df)
        
        # Dual candlestick patterns
        try:
            patterns['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            patterns['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        except:
            patterns['engulfing'] = self._detect_engulfing(df)
            patterns['harami'] = self._detect_harami(df)
        
        # Triple candlestick patterns
        try:
            patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        except:
            patterns['morning_star'] = self._detect_morning_star(df)
            patterns['evening_star'] = self._detect_evening_star(df)
        
        # Convert to binary (1 for pattern detected, 0 for no pattern)
        for pattern in patterns:
            patterns[pattern] = patterns[pattern].apply(lambda x: 1 if x > 0 else 0)
        
        return patterns
    
    def _detect_doji(self, df):
        """Detect Doji pattern"""
        body_size = abs(df['close'] - df['open'])
        total_size = df['high'] - df['low']
        doji = (body_size <= 0.1 * total_size).astype(int)
        return doji
    
    def _detect_hammer(self, df):
        """Detect Hammer pattern"""
        body_size = abs(df['close'] - df['open'])
        total_size = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        hammer = ((body_size <= 0.3 * total_size) & 
                  (lower_shadow >= 2 * body_size) & 
                  (upper_shadow <= 0.1 * total_size)).astype(int)
        return hammer
    
    def _detect_shooting_star(self, df):
        """Detect Shooting Star pattern"""
        body_size = abs(df['close'] - df['open'])
        total_size = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        shooting_star = ((body_size <= 0.3 * total_size) & 
                         (upper_shadow >= 2 * body_size) & 
                         (lower_shadow <= 0.1 * total_size)).astype(int)
        return shooting_star
    
    def _detect_engulfing(self, df):
        """Detect Engulfing pattern"""
        prev_body_size = abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body_size = abs(df['close'] - df['open'])
        
        bullish_engulfing = ((df['close'].shift(1) < df['open'].shift(1]) &  # Previous red
                             (df['close'] > df['open']) &  # Current green
                             (df['open'] < df['close'].shift(1)) &  # Current open below previous close
                             (df['close'] > df['open'].shift(1))).astype(int)  # Current close above previous open
        
        bearish_engulfing = ((df['close'].shift(1) > df['open'].shift(1)) &  # Previous green
                             (df['close'] < df['open']) &  # Current red
                             (df['open'] > df['close'].shift(1)) &  # Current open above previous close
                             (df['close'] < df['open'].shift(1])).astype(int)  # Current close below previous open
        
        return bullish_engulfing + bearish_engulfing
    
    def _detect_harami(self, df):
        """Detect Harami pattern"""
        prev_body_size = abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body_size = abs(df['close'] - df['open'])
        
        bullish_harami = ((df['close'].shift(1) < df['open'].shift(1)) &  # Previous red
                          (df['close'] > df['open']) &  # Current green
                          (df['open'] > df['close'].shift(1)) &  # Current open above previous close
                          (df['close'] < df['open'].shift(1]) &  # Current close below previous open
                          (curr_body_size < prev_body_size)).astype(int)  # Current body smaller
        
        bearish_harami = ((df['close'].shift(1) > df['open'].shift(1)) &  # Previous green
                          (df['close'] < df['open']) &  # Current red
                          (df['open'] < df['close'].shift(1)) &  # Current open below previous close
                          (df['close'] > df['open'].shift(1]) &  # Current close above previous open
                          (curr_body_size < prev_body_size)).astype(int)  # Current body smaller
        
        return bullish_harami + bearish_harami
    
    def _detect_morning_star(self, df):
        """Detect Morning Star pattern"""
        # Simplified implementation
        first_red = df['close'].shift(2) < df['open'].shift(2)
        second_small = abs(df['close'].shift(1) - df['open'].shift(1)) < 0.3 * (df['high'].shift(1) - df['low'].shift(1))
        third_green = df['close'] > df['open']
        
        morning_star = (first_red & second_small & third_green).astype(int)
        return morning_star
    
    def _detect_evening_star(self, df):
        """Detect Evening Star pattern"""
        # Simplified implementation
        first_green = df['close'].shift(2) > df['open'].shift(2)
        second_small = abs(df['close'].shift(1) - df['open'].shift(1)) < 0.3 * (df['high'].shift(1) - df['low'].shift(1))
        third_red = df['close'] < df['open']
        
        evening_star = (first_green & second_small & third_red).astype(int)
        return evening_star
