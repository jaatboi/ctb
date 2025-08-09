import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.plugin_manager import indicator

@indicator("volumewave")
def volumewave(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volumewave indicator - identifies buying/selling pressure using colored bars
    Returns: Series with values: 1 (buying pressure), -1 (selling pressure), 0 (neutral)
    """
    # Calculate volume SMA
    volume_sma = df['volume'].rolling(window=period).mean()
    
    # Calculate price change
    price_change = df['close'].diff()
    
    # Initialize volumewave series
    volumewave = pd.Series(0, index=df.index)
    
    # Identify buying and selling pressure
    volumewave[(price_change > 0) & (df['volume'] > volume_sma)] = 1  # Buying pressure
    volumewave[(price_change < 0) & (df['volume'] > volume_sma)] = -1  # Selling pressure
    
    return volumewave

@indicator("market_profile")
def market_profile(df: pd.DataFrame, window: int = 20, bins: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Market Profile indicator - provides order flow analysis
    Returns: Tuple of (Point of Control, Value Area High, Value Area Low)
    """
    poc = pd.Series(np.nan, index=df.index)
    vah = pd.Series(np.nan, index=df.index)  # Value Area High
    val = pd.Series(np.nan, index=df.index)  # Value Area Low
    
    for i in range(window, len(df)):
        # Get price range for the window
        high_prices = df['high'].iloc[i-window:i].values
        low_prices = df['low'].iloc[i-window:i].values
        volumes = df['volume'].iloc[i-window:i].values
        
        # Create price levels
        price_levels = []
        volume_at_price = {}
        
        for j in range(len(high_prices)):
            # Divide price range into bins
            price_bins = np.linspace(low_prices[j], high_prices[j], bins)
            
            # Distribute volume across price bins
            volume_per_bin = volumes[j] / bins
            
            for price in price_bins:
                price_levels.append(price)
                volume_at_price[price] = volume_at_price.get(price, 0) + volume_per_bin
        
        # Find Point of Control (price with highest volume)
        if volume_at_price:
            poc_price = max(volume_at_price.items(), key=lambda x: x[1])[0]
            
            # Calculate Value Area (70% of volume around POC)
            sorted_prices = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
            total_volume = sum(count for _, count in sorted_prices)
            target_volume = total_volume * 0.7
            
            cumulative_volume = 0
            vah_price = poc_price
            val_price = poc_price
            
            for price, count in sorted_prices:
                cumulative_volume += count
                if cumulative_volume <= target_volume:
                    if price > poc_price:
                        vah_price = max(vah_price, price)
                    elif price < poc_price:
                        val_price = min(val_price, price)
                else:
                    break
            
            poc.iloc[i] = poc_price
            vah.iloc[i] = vah_price
            val.iloc[i] = val_price
    
    return poc, vah, val

@indicator("intraday_momentum_index")
def intraday_momentum_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Intraday Momentum Index - modified RSI for shorter timeframes
    Returns: Series with IMI values (0-100)
    """
    # Calculate typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate price changes
    tp_change = tp.diff()
    
    # Separate up and down movements
    up = pd.Series(0, index=df.index)
    down = pd.Series(0, index=df.index)
    
    up[tp_change > 0] = tp_change[tp_change > 0]
    down[tp_change < 0] = -tp_change[tp_change < 0]
    
    # Calculate rolling averages
    up_avg = up.rolling(window=period).mean()
    down_avg = down.rolling(window=period).mean()
    
    # Calculate IMI
    imi = 100 * (up_avg / (up_avg + down_avg))
    
    return imi

@indicator("support_resistance")
def support_resistance(df: pd.DataFrame, window: int = 20, tolerance: float = 0.02) -> Tuple[pd.Series, pd.Series]:
    """
    Identify support and resistance levels
    Returns: Tuple of (support levels, resistance levels)
    """
    # Find local highs and lows
    local_high = df['high'].rolling(window=window, center=True).max()
    local_low = df['low'].rolling(window=window, center=True).min()
    
    # Initialize support and resistance series
    support = pd.Series(np.nan, index=df.index)
    resistance = pd.Series(np.nan, index=df.index)
    
    # Find significant support and resistance levels
    for i in range(window, len(df) - window):
        # Check if current high is a resistance level
        if df['high'].iloc[i] == local_high.iloc[i]:
            # Check if this is a significant high
            is_resistance = True
            for j in range(i - window, i + window + 1):
                if j != i and abs(df['high'].iloc[i] - df['high'].iloc[j]) / df['high'].iloc[i] < tolerance:
                    is_resistance = False
                    break
            
            if is_resistance:
                resistance.iloc[i] = df['high'].iloc[i]
        
        # Check if current low is a support level
        if df['low'].iloc[i] == local_low.iloc[i]:
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

@indicator("market_regime")
def market_regime(df: pd.DataFrame, adx_period: int = 14, bb_period: int = 20, bb_std: float = 2.0) -> pd.Series:
    """
    Identify market regime: trending, breakout, or range-bound
    Returns: Series with regime labels
    """
    # Calculate ADX for trend strength
    try:
        import talib
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=adx_period)
    except ImportError:
        # Fallback if talib is not available
        adx = pd.Series(0, index=df.index)
    
    # Calculate Bollinger Bands
    try:
        upper_band, middle_band, lower_band = talib.BBANDS(
            df['close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std, matype=0)
    except ImportError:
        # Fallback if talib is not available
        middle_band = df['close'].rolling(window=bb_period).mean()
        std = df['close'].rolling(window=bb_period).std()
        upper_band = middle_band + (std * bb_std)
        lower_band = middle_band - (std * bb_std)
    
    # Calculate Bollinger Bandwidth for volatility
    bb_width = (upper_band - lower_band) / middle_band
    
    # Calculate price range
    price_range = (df['high'] - df['low']) / df['close']
    
    # Determine regime for each candle
    regime = pd.Series('unknown', index=df.index)
    
    # High volatility regime
    regime[(bb_width > 0.15) | (price_range > 0.03)] = 'breakout'
    # Strong trend regime
    regime[adx > 25] = 'trending'
    # Range-bound regime
    regime[(adx <= 25) & (bb_width <= 0.15) & (price_range <= 0.03)] = 'range_bound'
    
    return regime
