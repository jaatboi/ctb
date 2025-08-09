import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from interfaces.base_components import BaseSignalGenerator
from utils.plugin_manager import strategy

@strategy("intraday_multi_tf")
class IntradayMultiTimeframeStrategy(BaseSignalGenerator):
    """
    Multi-timeframe intraday strategy combining momentum, breakout, and mean-reversion approaches.
    Adapts to market conditions and uses confluence across 15m, 30m, 45m, and 1h timeframes.
    """
    
    def __init__(self, risk_percent=1.5, min_rr_ratio=2.0, atr_period=14, atr_multiplier=2.0):
        self.risk_percent = risk_percent  # 1.5% risk per trade
        self.min_rr_ratio = min_rr_ratio  # 1:2 risk-reward minimum
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
    def generate_signals(self, df: pd.DataFrame, sentiment_data: Optional[Dict] = None) -> List[Dict]:
        """Generate trading signals based on multi-timeframe analysis"""
        signals = []
        
        # Add technical indicators
        df = self._add_indicators(df)
        
        # Identify market regime (trending, range-bound, high volatility)
        df['market_regime'] = self._identify_market_regime(df)
        
        # Generate signals for each candle
        for i in range(45, len(df)):  # Start from 45 to have enough data for indicators
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # Skip if we don't have enough data for higher timeframes
            if i < 60:  # Need at least 60 candles for 1h confirmation
                continue
                
            # Get higher timeframe confirmation
            htf_trend = self._get_htf_trend(df, i)
            
            # Generate signal based on market regime
            signal = None
            
            if current_candle['market_regime'] == 'trending':
                signal = self._momentum_signal(df, i, htf_trend)
            elif current_candle['market_regime'] == 'breakout':
                signal = self._breakout_signal(df, i, htf_trend)
            elif current_candle['market_regime'] == 'range_bound':
                signal = self._mean_reversion_signal(df, i, htf_trend)
            
            if signal:
                # Add sentiment confirmation if available
                if sentiment_data and sentiment_data.get('overall_sentiment'):
                    sentiment_score = sentiment_data['overall_sentiment']
                    # Adjust confidence based on sentiment (15-20% weight)
                    sentiment_weight = 0.175  # 17.5% weight
                    sentiment_adjustment = sentiment_score * sentiment_weight
                    
                    # Adjust confidence based on sentiment
                    signal['confidence'] = min(0.95, max(0.5, signal['confidence'] + sentiment_adjustment))
                    
                    # Add rationale for sentiment
                    signal['rationale'] += f" Sentiment: {sentiment_score:.2f} ({'positive' if sentiment_score > 0 else 'negative'})."
                
                signals.append(signal)
        
        return signals
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required technical indicators"""
        try:
            import talib
        except ImportError:
            # Use fallback implementations if talib is not available
            talib = None
        
        # Standard indicators
        if talib:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        else:
            # Fallback implementations
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            df['macd'], df['macdsignal'], df['macdhist'] = self._calculate_macd(df['close'])
            df['upper_band'], df['middle_band'], df['lower_band'] = self._calculate_bollinger_bands(df['close'])
            df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Custom indicators
        df = self._volumewave(df)
        df = self._market_profile(df)
        df = self._intraday_momentum_index(df)
        
        # Support and resistance
        df['support'], df['resistance'] = self._identify_support_resistance(df)
        
        # Multi-timeframe SMAs
        df['sma_15'] = df['close'].rolling(window=15).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()
        df['sma_45'] = df['close'].rolling(window=45).mean()
        df['sma_60'] = df['close'].rolling(window=60).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _volumewave(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volumewave indicator - identifies buying/selling pressure using colored bars"""
        # Calculate price change and volume
        df['price_change'] = df['close'].diff()
        df['volume_change'] = df['volume'].diff()
        
        # Initialize volumewave column
        df['volumewave'] = 0  # 0: neutral, 1: buying pressure, -1: selling pressure
        
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0 and df['volume'].iloc[i] > df['volume_sma'].iloc[i]:
                df.at[i, 'volumewave'] = 1  # Buying pressure
            elif df['price_change'].iloc[i] < 0 and df['volume'].iloc[i] > df['volume_sma'].iloc[i]:
                df.at[i, 'volumewave'] = -1  # Selling pressure
            # Otherwise remains neutral (0)
        
        return df
    
    def _market_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market Profile indicator - provides order flow analysis"""
        # This is a simplified version of Market Profile
        # In a full implementation, this would use TPO (Time Price Opportunity) data
        
        # Calculate value area and point of control
        window = 20  # Lookback period
        
        df['poc'] = np.nan  # Point of Control
        df['value_area_high'] = np.nan
        df['value_area_low'] = np.nan
        
        for i in range(window, len(df)):
            # Get price range for the window
            high_prices = df['high'].iloc[i-window:i].values
            low_prices = df['low'].iloc[i-window:i].values
            volumes = df['volume'].iloc[i-window:i].values
            
            # Find price levels with highest volume (simplified POC)
            price_levels = []
            for j in range(len(high_prices)):
                price_levels.extend(np.linspace(low_prices[j], high_prices[j], 10).tolist())
            
            # Count volume at each price level (simplified)
            price_counts = {}
            for price in price_levels:
                price_counts[price] = price_counts.get(price, 0) + 1
            
            # Find POC (price with highest count)
            poc = max(price_counts, key=price_counts.get)
            
            # Calculate value area (70% of volume around POC)
            sorted_prices = sorted(price_counts.items(), key=lambda x: x[1], reverse=True)
            total_volume = sum(count for _, count in sorted_prices)
            target_volume = total_volume * 0.7
            
            cumulative_volume = 0
            value_area_high = poc
            value_area_low = poc
            
            for price, count in sorted_prices:
                cumulative_volume += count
                if cumulative_volume <= target_volume:
                    if price > poc:
                        value_area_high = price
                    elif price < poc:
                        value_area_low = price
                else:
                    break
            
            df.at[i, 'poc'] = poc
            df.at[i, 'value_area_high'] = value_area_high
            df.at[i, 'value_area_low'] = value_area_low
        
        return df
    
    def _intraday_momentum_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intraday Momentum Index - modified RSI for shorter timeframes"""
        # Calculate typical price
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price changes
        df['tp_change'] = df['tp'].diff()
        
        # Separate up and down movements
        df['up'] = np.where(df['tp_change'] > 0, df['tp_change'], 0)
        df['down'] = np.where(df['tp_change'] < 0, -df['tp_change'], 0)
        
        # Calculate averages
        period = 14
        df['up_avg'] = df['up'].rolling(window=period).mean()
        df['down_avg'] = df['down'].rolling(window=period).mean()
        
        # Calculate IMI
        df['imi'] = 100 * (df['up_avg'] / (df['up_avg'] + df['down_avg']))
        
        # Clean up temporary columns
        df.drop(['tp', 'tp_change', 'up', 'down', 'up_avg', 'down_avg'], axis=1, inplace=True)
        
        return df
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Identify support and resistance levels"""
        window = 20
        tolerance = 0.02
        
        # Find local highs and lows
        df['local_high'] = df['high'].rolling(window=window, center=True).max()
        df['local_low'] = df['low'].rolling(window=window, center=True).min()
        
        # Initialize support and resistance series
        support = pd.Series(index=df.index, dtype=float)
        resistance = pd.Series(index=df.index, dtype=float)
        
        # Find significant support and resistance levels
        for i in range(window, len(df) - window):
            # Check if current high is a resistance level
            if df['high'].iloc[i] == df['local_high'].iloc[i]:
                # Check if this is a significant high
                is_resistance = True
                for j in range(i - window, i + window + 1):
                    if j != i and abs(df['high'].iloc[i] - df['high'].iloc[j]) / df['high'].iloc[i] < tolerance:
                        is_resistance = False
                        break
                
                if is_resistance:
                    resistance.iloc[i] = df['high'].iloc[i]
            
            # Check if current low is a support level
            if df['low'].iloc[i] == df['local_low'].iloc[i]:
                # Check if this is a significant low
                is_support = True
                for j in range(i - window, i + window + 1):
                    if j != i and abs(df['low'].iloc[i] - df['low'].iloc[j]) / df['low'].iloc[i] < tolerance:
                        is_support = False
                        break
                
                if is_support:
                    support.iloc[i] = df['low'].iloc[i]
        
        # Forward fill support and resistance levels
        support = support.fillna(method='ffill')
        resistance = resistance.fillna(method='ffill')
        
        return support, resistance
    
    def _identify_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Identify market regime: trending, breakout, or range-bound"""
        regime = pd.Series(index=df.index, dtype=object)
        
        # Calculate ADX for trend strength
        try:
            import talib
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        except ImportError:
            df['adx'] = pd.Series(0, index=df.index)
        
        # Calculate Bollinger Bandwidth for volatility
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        # Calculate price range
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Determine regime for each candle
        for i in range(14, len(df)):
            adx = df['adx'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            price_range = df['price_range'].iloc[i]
            
            # High volatility regime
            if bb_width > 0.15 or price_range > 0.03:
                regime.iloc[i] = 'breakout'
            # Strong trend regime
            elif adx > 25:
                regime.iloc[i] = 'trending'
            # Range-bound regime
            else:
                regime.iloc[i] = 'range_bound'
        
        return regime
    
    def _get_htf_trend(self, df: pd.DataFrame, index: int) -> str:
        """Get higher timeframe trend for confirmation"""
        # 1-hour trend (using 60-period SMA on 45m data)
        if index >= 60:
            current_price = df['close'].iloc[index]
            sma_60 = df['sma_60'].iloc[index]
            
            if current_price > sma_60:
                return 'bullish'
            else:
                return 'bearish'
        return 'neutral'
    
    def _momentum_signal(self, df: pd.DataFrame, index: int, htf_trend: str) -> Optional[Dict]:
        """Generate momentum-based trading signal"""
        current = df.iloc[index]
        prev = df.iloc[index-1]
        
        # Entry conditions for long momentum
        long_entry = (
            current['rsi'] > 50 and 
            current['macd'] > current['macdsignal'] and 
            current['volumewave'] > 0 and
            current['volume_ratio'] > 1.2 and
            htf_trend == 'bullish'
        )
        
        # Entry conditions for short momentum
        short_entry = (
            current['rsi'] < 50 and 
            current['macd'] < current['macdsignal'] and 
            current['volumewave'] < 0 and
            current['volume_ratio'] > 1.2 and
            htf_trend == 'bearish'
        )
        
        if long_entry:
            # Calculate entry, stop loss, and take profit levels
            entry = current['close']
            stop_loss = current['low'] - (current['atr'] * self.atr_multiplier)
            take_profit_levels = [
                entry + (current['atr'] * 2.0),  # TP1: 2x ATR
                entry + (current['atr'] * 3.5),  # TP2: 3.5x ATR
                entry + (current['atr'] * 5.0)   # TP3: 5x ATR
            ]
            
            # Calculate probabilities
            tp_probabilities = [0.75, 0.50, 0.25]
            sl_probability = 0.25
            
            return {
                'timestamp': current['timestamp'],
                'symbol': df.get('symbol', 'UNKNOWN'),
                'signal_type': 'BUY',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': tp_probabilities,
                'sl_probability': sl_probability,
                'confidence': 0.80,
                'rationale': f"Momentum long signal: RSI {current['rsi']:.1f}, MACD bullish, volume confirmation",
                'timeframe': '45m',
                'strategy': 'momentum'
            }
        
        elif short_entry:
            # Calculate entry, stop loss, and take profit levels
            entry = current['close']
            stop_loss = current['high'] + (current['atr'] * self.atr_multiplier)
            take_profit_levels = [
                entry - (current['atr'] * 2.0),  # TP1: 2x ATR
                entry - (current['atr'] * 3.5),  # TP2: 3.5x ATR
                entry - (current['atr'] * 5.0)   # TP3: 5x ATR
            ]
            
            # Calculate probabilities
            tp_probabilities = [0.75, 0.50, 0.25]
            sl_probability = 0.25
            
            return {
                'timestamp': current['timestamp'],
                'symbol': df.get('symbol', 'UNKNOWN'),
                'signal_type': 'SELL',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': tp_probabilities,
                'sl_probability': sl_probability,
                'confidence': 0.80,
                'rationale': f"Momentum short signal: RSI {current['rsi']:.1f}, MACD bearish, volume confirmation",
                'timeframe': '45m',
                'strategy': 'momentum'
            }
        
        return None
    
    def _breakout_signal(self, df: pd.DataFrame, index: int, htf_trend: str) -> Optional[Dict]:
        """Generate breakout-based trading signal"""
        current = df.iloc[index]
        prev = df.iloc[index-1]
        
        # Entry conditions for long breakout
        long_entry = (
            current['close'] > current['upper_band'] and
            prev['close'] <= prev['upper_band'] and
            current['volume_ratio'] > 1.5 and
            current['volumewave'] > 0 and
            htf_trend != 'bearish'
        )
        
        # Entry conditions for short breakout
        short_entry = (
            current['close'] < current['lower_band'] and
            prev['close'] >= prev['lower_band'] and
            current['volume_ratio'] > 1.5 and
            current['volumewave'] < 0 and
            htf_trend != 'bullish'
        )
        
        if long_entry:
            # Calculate entry, stop loss, and take profit levels
            entry = current['close']
            stop_loss = current['lower_band']  # Use lower band as stop loss
            take_profit_levels = [
                entry + (current['atr'] * 2.5),  # TP1: 2.5x ATR
                entry + (current['atr'] * 4.0),  # TP2: 4x ATR
                entry + (current['atr'] * 6.0)   # TP3: 6x ATR
            ]
            
            # Calculate probabilities
            tp_probabilities = [0.70, 0.45, 0.20]
            sl_probability = 0.30
            
            return {
                'timestamp': current['timestamp'],
                'symbol': df.get('symbol', 'UNKNOWN'),
                'signal_type': 'BUY',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': tp_probabilities,
                'sl_probability': sl_probability,
                'confidence': 0.75,
                'rationale': f"Bullish breakout from Bollinger Bands with volume confirmation",
                'timeframe': '45m',
                'strategy': 'breakout'
            }
        
        elif short_entry:
            # Calculate entry, stop loss, and take profit levels
            entry = current['close']
            stop_loss = current['upper_band']  # Use upper band as stop loss
            take_profit_levels = [
                entry - (current['atr'] * 2.5),  # TP1: 2.5x ATR
                entry - (current['atr'] * 4.0),  # TP2: 4x ATR
                entry - (current['atr'] * 6.0)   # TP3: 6x ATR
            ]
            
            # Calculate probabilities
            tp_probabilities = [0.70, 0.45, 0.20]
            sl_probability = 0.30
            
            return {
                'timestamp': current['timestamp'],
                'symbol': df.get('symbol', 'UNKNOWN'),
                'signal_type': 'SELL',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': tp_probabilities,
                'sl_probability': sl_probability,
                'confidence': 0.75,
                'rationale': f"Bearish breakout from Bollinger Bands with volume confirmation",
                'timeframe': '45m',
                'strategy': 'breakout'
            }
        
        return None
    
    def _mean_reversion_signal(self, df: pd.DataFrame, index: int, htf_trend: str) -> Optional[Dict]:
        """Generate mean-reversion-based trading signal"""
        current = df.iloc[index]
        prev = df.iloc[index-1]
        
        # Entry conditions for long mean-reversion
        long_entry = (
            current['rsi'] < 30 and 
            current['imi'] < 30 and
            current['close'] <= current['support'] * 1.01 and  # Near support
            current['volumewave'] > 0 and  # Buying pressure emerging
            htf_trend != 'bearish'
        )
        
        # Entry conditions for short mean-reversion
        short_entry = (
            current['rsi'] > 70 and 
            current['imi'] > 70 and
            current['close'] >= current['resistance'] * 0.99 and  # Near resistance
            current['volumewave'] < 0 and  # Selling pressure emerging
            htf_trend != 'bullish'
        )
        
        if long_entry:
            # Calculate entry, stop loss, and take profit levels
            entry = current['close']
            stop_loss = current['low'] - (current['atr'] * 1.0)  # Tighter stop for mean-reversion
            take_profit_levels = [
                current['middle_band'],  # TP1: Middle band (mean reversion target)
                current['middle_band'] + (current['atr'] * 1.0),  # TP2: Above mean
                current['resistance']  # TP3: Resistance level
            ]
            
            # Calculate probabilities
            tp_probabilities = [0.85, 0.60, 0.30]
            sl_probability = 0.15
            
            return {
                'timestamp': current['timestamp'],
                'symbol': df.get('symbol', 'UNKNOWN'),
                'signal_type': 'BUY',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': tp_probabilities,
                'sl_probability': sl_probability,
                'confidence': 0.85,
                'rationale': f"Mean-reversion long: RSI {current['rsi']:.1f}, near support level",
                'timeframe': '45m',
                'strategy': 'mean_reversion'
            }
        
        elif short_entry:
            # Calculate entry, stop loss, and take profit levels
            entry = current['close']
            stop_loss = current['high'] + (current['atr'] * 1.0)  # Tighter stop for mean-reversion
            take_profit_levels = [
                current['middle_band'],  # TP1: Middle band (mean reversion target)
                current['middle_band'] - (current['atr'] * 1.0),  # TP2: Below mean
                current['support']  # TP3: Support level
            ]
            
            # Calculate probabilities
            tp_probabilities = [0.85, 0.60, 0.30]
            sl_probability = 0.15
            
            return {
                'timestamp': current['timestamp'],
                'symbol': df.get('symbol', 'UNKNOWN'),
                'signal_type': 'SELL',
                'entry_price': entry,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': tp_probabilities,
                'sl_probability': sl_probability,
                'confidence': 0.85,
                'rationale': f"Mean-reversion short: RSI {current['rsi']:.1f}, near resistance level",
                'timeframe': '45m',
                'strategy': 'mean_reversion'
            }
        
        return None
    
    def filter_signals(self, signals: List[Dict], min_confidence: float = 0.65, 
                      min_risk_reward_ratio: float = 1.5) -> List[Dict]:
        """Filter signals based on quality criteria"""
        filtered_signals = []
        
        for signal in signals:
            # Skip if confidence is too low
            if signal['confidence'] < min_confidence:
                continue
            
            # Calculate risk/reward ratio
            if signal['entry_price'] and signal['stop_loss'] and signal['take_profit_levels']:
                risk = abs(signal['entry_price'] - signal['stop_loss'])
                reward = abs(signal['take_profit_levels'][0] - signal['entry_price'])
                risk_reward_ratio = reward / risk if risk > 0 else 0
                
                if risk_reward_ratio < min_risk_reward_ratio:
                    continue
                
                signal['risk_reward_ratio'] = risk_reward_ratio
            else:
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def train_model(self, df: pd.DataFrame, sentiment_data: Optional[Dict] = None) -> Dict:
        """This strategy doesn't require model training"""
        return {
            'message': 'Intraday multi-timeframe strategy does not require model training',
            'success': True
        }
