from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional

class BaseDataCollector(ABC):
    """Abstract base class for data collectors"""
    
    @abstractmethod
    def get_historical_klines(self, symbol: str, interval: str, lookback_period: int) -> pd.DataFrame:
        """Get historical klines data"""
        pass
    
    @abstractmethod
    def start_realtime_collection(self, symbol: str, interval: str, callback) -> None:
        """Start real-time data collection"""
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str) -> Dict:
        """Get current order book"""
        pass
    
    @abstractmethod
    def get_recent_trades(self, symbol: str) -> List:
        """Get recent trades"""
        pass


class BaseTechnicalAnalyzer(ABC):
    """Abstract base class for technical analysis"""
    
    @abstractmethod
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe"""
        pass
    
    @abstractmethod
    def identify_support_resistance(self, df: pd.DataFrame, window: int = 20, tolerance: float = 0.02) -> tuple:
        """Identify support and resistance levels"""
        pass
    
    @abstractmethod
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect candlestick patterns"""
        pass


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analysis"""
    
    @abstractmethod
    def get_market_sentiment_summary(self, symbols: List[str]) -> Dict:
        """Get comprehensive market sentiment summary"""
        pass


class BaseSignalGenerator(ABC):
    """Abstract base class for signal generation"""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, sentiment_data: Optional[Dict] = None) -> List[Dict]:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def filter_signals(self, signals: List[Dict], min_confidence: float = 0.65, 
                      min_risk_reward_ratio: float = 1.5) -> List[Dict]:
        """Filter signals based on quality criteria"""
        pass
    
    @abstractmethod
    def train_model(self, df: pd.DataFrame, sentiment_data: Optional[Dict] = None) -> Dict:
        """Train the signal generation model"""
        pass


class BaseExecutionEngine(ABC):
    """Abstract base class for trade execution"""
    
    @abstractmethod
    def execute_trade(self, signal: Dict, risk_percent: float = 1.0, leverage: int = 5) -> Dict:
        """Execute a trade based on a signal"""
        pass
    
    @abstractmethod
    def close_position(self, symbol: str) -> Dict:
        """Close an open position"""
        pass
    
    @abstractmethod
    def get_open_positions(self) -> Dict:
        """Get all open positions"""
        pass
    
    @abstractmethod
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        pass
    
    @abstractmethod
    def update_positions(self) -> None:
        """Update open positions with current market data"""
        pass
    
    @abstractmethod
    def paper_trade(self, signal: Dict, risk_percent: float = 1.0, leverage: int = 5) -> Dict:
        """Simulate a trade without executing it"""
        pass
