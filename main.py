import os
import sys
import time
import json
import logging
from datetime import datetime
import argparse
import threading

# Import modules
from modules.data_collector import BinanceDataCollector
from modules.market_analysis import TechnicalAnalyzer
from modules.sentiment_engine import SentimentAnalyzer
from modules.signal_generator import SignalGenerator
from modules.execution_engine import ExecutionEngine
from ui.dashboard import TradingBotDashboard
from utils.plugin_manager import PluginManager
from utils.database import TradingDatabase
from api.app import app as api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config_path=None):
        """Initialize the trading bot"""
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager(self.config.get('plugin_dirs', ["plugins"]))
        
        # Database path
        db_path = self.config.get('database_path', 'data/trading_bot.db')
        
        # Initialize components based on configuration
        data_source_name = self.config.get('data_source', 'enhanced_binance')
        if data_source_name in self.plugin_manager.list_data_sources():
            self.data_collector = self.plugin_manager.get_data_source(data_source_name)
        else:
            self.data_collector = BinanceDataCollector(
                self.config['binance']['api_key'],
                self.config['binance']['api_secret'],
                db_path=db_path
            )
        
        self.technical_analyzer = TechnicalAnalyzer(self.plugin_manager)
        
        sentiment_source_name = self.config.get('sentiment_source', 'enhanced_crypto_sentiment')
        if sentiment_source_name in self.plugin_manager.list_sentiment_sources():
            self.sentiment_analyzer = self.plugin_manager.get_sentiment_source(sentiment_source_name)
        else:
            self.sentiment_analyzer = SentimentAnalyzer(
                self.config['gemini']['api_key'],
                self.config['perplexity']['api_key'],
                self.config['twitter']['api_key'],
                self.config['twitter']['api_secret'],
                self.config['twitter']['access_token'],
                self.config['twitter']['access_token_secret']
            )
        
        model_path = self.config.get('model_path', 'models/signal_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        strategy_name = self.config.get('strategy', 'intraday_multi_tf')
        if strategy_name in self.plugin_manager.list_strategies():
            self.signal_generator = self.plugin_manager.get_strategy(strategy_name)
        else:
            self.signal_generator = SignalGenerator(model_path)
        
        self.execution_engine = ExecutionEngine(
            self.config['binance']['api_key'],
            self.config['binance']['api_secret'],
            testnet=self.config['binance'].get('testnet', True),
            db_path=db_path
        )
        
        # Dashboard
        self.dashboard = None
        
        # API server
        self.api_server = None
        
        logger.info("Trading bot initialized successfully")
        logger.info(f"Data source: {data_source_name}")
        logger.info(f"Sentiment source: {sentiment_source_name}")
        logger.info(f"Strategy: {strategy_name}")
    
    def load_config(self, config_path=None):
        """Load configuration from file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "binance": {
                    "api_key": os.environ.get('BINANCE_API_KEY', ''),
                    "api_secret": os.environ.get('BINANCE_API_SECRET', ''),
                    "testnet": True
                },
                "gemini": {
                    "api_key": os.environ.get('GEMINI_API_KEY', '')
                },
                "perplexity": {
                    "api_key": os.environ.get('PERPLEXITY_API_KEY', '')
                },
                "twitter": {
                    "api_key": os.environ.get('TWITTER_API_KEY', ''),
                    "api_secret": os.environ.get('TWITTER_API_SECRET', ''),
                    "access_token": os.environ.get('TWITTER_ACCESS_TOKEN', ''),
                    "access_token_secret": os.environ.get('TWITTER_ACCESS_TOKEN_SECRET', '')
                },
                "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],
                "intervals": ["15m", "30m", "45m", "1h"],
                "default_interval": "45m",
                "risk_percent": 1.5,
                "leverage": 5,
                "model_path": "models/signal_model.pkl",
                "database_path": "data/trading_bot.db",
                "plugin_dirs": ["plugins"],
                "data_source": "enhanced_binance",
                "sentiment_source": "enhanced_crypto_sentiment",
                "strategy": "intraday_multi_tf",
                "api": {
                    "enabled": False,
                    "host": "0.0.0.0",
                    "port": 5000
                },
                "notifications": {
                    "telegram": {
                        "enabled": False,
                        "bot_token": "",
                        "chat_id": ""
                    },
                    "email": {
                        "enabled": False,
                        "smtp_server": "",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "recipients": []
                    }
                }
            }
            
            # Save default configuration
            if not os.path.exists('config'):
                os.makedirs('config')
            
            with open('config/default_config.json', 'w') as f:
                json.dump(config, f, indent=4)
        
        return config
    
    def run_dashboard(self, port=8050):
        """Run the trading bot dashboard"""
        self.dashboard = TradingBotDashboard(
            self.signal_generator,
            self.execution_engine,
            port=port
        )
        logger.info(f"Starting dashboard on port {port}")
        self.dashboard.run()
    
    def run_bot(self, symbols=None, interval=None, risk_percent=None, leverage=None, mode='paper', strategy=None):
        """Run the trading bot in automated mode"""
        symbols = symbols or self.config['symbols']
        interval = interval or self.config['default_interval']
        risk_percent = risk_percent or self.config['risk_percent']
        leverage = leverage or self.config['leverage']
        
        logger.info(f"Starting trading bot in {mode} mode")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Interval: {interval}")
        logger.info(f"Risk percent: {risk_percent}%")
        logger.info(f"Leverage: {leverage}x")
        if strategy:
            logger.info(f"Strategy: {strategy}")
        
        # Get strategy instance if specified
        strategy_instance = None
        if strategy and strategy in self.plugin_manager.list_strategies():
            strategy_instance = self.plugin_manager.get_strategy(strategy)
        
        # Main loop
        while True:
            try:
                for symbol in symbols:
                    logger.info(f"Processing {symbol}")
                    
                    # Get market data (using database cache)
                    df = self.data_collector.get_historical_klines(symbol, interval, 100)
                    
                    # Add technical indicators
                    df = self.technical_analyzer.add_all_indicators(df)
                    
                    # Get sentiment data
                    sentiment_data = self.sentiment_analyzer.get_market_sentiment_summary([symbol.replace('USDT', '')])
                    
                    # Store sentiment data in database
                    self.execution_engine.db.store_sentiment_data(symbol, sentiment_data)
                    
                    # Generate signals
                    if strategy_instance:
                        signals = strategy_instance.generate_signals(df, sentiment_data)
                        filtered_signals = strategy_instance.filter_signals(signals)
                    else:
                        signals = self.signal_generator.generate_signals(df, sentiment_data)
                        filtered_signals = self.signal_generator.filter_signals(signals)
                    
                    if filtered_signals:
                        logger.info(f"Generated {len(filtered_signals)} signals for {symbol}")
                        
                        for signal in filtered_signals:
                            logger.info(f"Signal: {signal['signal_type']} for {symbol} at {signal['entry_price']}")
                            
                            # Execute trade
                            if mode == 'live':
                                result = self.execution_engine.execute_trade(signal, risk_percent, leverage)
                            else:
                                result = self.execution_engine.paper_trade(signal, risk_percent, leverage)
                            
                            if result['success']:
                                logger.info(f"Trade executed successfully: {result['message']}")
                                
                                # Send notification if enabled
                                self.send_notification(signal, result)
                            else:
                                logger.error(f"Failed to execute trade: {result['message']}")
                    else:
                        logger.info(f"No signals generated for {symbol}")
                
                # Update positions
                self.execution_engine.update_positions()
                
                # Sleep until next interval
                interval_minutes = int(interval.replace('m', '')) if 'm' in interval else int(interval.replace('h', '')) * 60
                logger.info(f"Sleeping for {interval_minutes} minutes")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def run_api_server(self, host=None, port=None):
        """Run the API server in a separate thread"""
        host = host or self.config.get('api', {}).get('host', '0.0.0.0')
        port = port or self.config.get('api', {}).get('port', 5000)
        
        logger.info(f"Starting API server on {host}:{port}")
        
        # Run API server in a separate thread
        self.api_server = threading.Thread(
            target=lambda: api_app.run(host=host, port=port, debug=False),
            daemon=True
        )
        self.api_server.start()
    
    def send_notification(self, signal, result):
        """Send notification for a trade"""
        # This would be implemented with Telegram, email, etc.
        pass
    
    def train_model(self, symbols=None, interval=None, lookback_days=30):
        """Train the signal generation model"""
        symbols = symbols or self.config['symbols']
        interval = interval or self.config['default_interval']
        
        logger.info(f"Training model with data from {symbols} at {interval} interval")
        
        # Collect historical data
        all_data = []
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}")
            
            # Calculate the number of klines needed
            klines_per_day = 24 * 60 // int(interval.replace('m', '')) if 'm' in interval else 24 // int(interval.replace('h', ''))
            limit = klines_per_day * lookback_days
            
            # Get historical data (using database cache)
            df = self.data_collector.get_historical_klines(symbol, interval, limit)
            df['symbol'] = symbol
            
            # Add technical indicators
            df = self.technical_analyzer.add_all_indicators(df)
            
            all_data.append(df)
        
        # Combine data from all symbols
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Train the model
        logger.info("Training model")
        model_results = self.signal_generator.train_model(combined_df)
        
        logger.info(f"Model trained with accuracy: {model_results['accuracy']}")
        logger.info(f"Feature importance: {model_results['feature_importance']}")
        
        return model_results
    
    def close(self):
        """Close connections"""
        self.data_collector.close()
        self.execution_engine.close()

def main():
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--mode', choices=['dashboard', 'bot', 'train', 'api'], default='dashboard',
                        help='Run mode: dashboard, bot, train, or api')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--port', type=int, default=8050, help='Port for dashboard')
    parser.add_argument('--api-port', type=int, default=5000, help='Port for API server')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--interval', type=str, help='Time interval')
    parser.add_argument('--risk', type=float, help='Risk percentage')
    parser.add_argument('--leverage', type=int, help='Leverage')
    parser.add_argument('--trade-mode', choices=['live', 'paper'], default='paper',
                        help='Trading mode: live or paper')
    parser.add_argument('--strategy', type=str, help='Strategy to use')
    parser.add_argument('--lookback-days', type=int, default=30, help='Days of historical data for training')
    parser.add_argument('--enable-api', action='store_true', help='Enable API server')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = TradingBot(args.config)
    
    try:
        # Start API server if enabled
        if args.enable_api or args.mode == 'api':
            bot.run_api_server(port=args.api_port)
        
        # Run based on mode
        if args.mode == 'dashboard':
            bot.run_dashboard(args.port)
        elif args.mode == 'bot':
            bot.run_bot(
                symbols=args.symbols,
                interval=args.interval,
                risk_percent=args.risk,
                leverage=args.leverage,
                mode=args.trade_mode,
                strategy=args.strategy
            )
        elif args.mode == 'train':
            bot.train_model(
                symbols=args.symbols,
                interval=args.interval,
                lookback_days=args.lookback_days
            )
        elif args.mode == 'api':
            # Keep the API server running
            while True:
                time.sleep(60)
    finally:
        bot.close()

if __name__ == '__main__':
    main()
