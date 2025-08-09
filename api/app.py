from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

# Import trading bot components
from interfaces.base_components import BaseDataCollector, BaseTechnicalAnalyzer, BaseSentimentAnalyzer, BaseSignalGenerator, BaseExecutionEngine
from utils.plugin_manager import PluginManager
from utils.database import TradingDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store bot components
data_collector: Optional[BaseDataCollector] = None
technical_analyzer: Optional[BaseTechnicalAnalyzer] = None
sentiment_analyzer: Optional[BaseSentimentAnalyzer] = None
signal_generator: Optional[BaseSignalGenerator] = None
execution_engine: Optional[BaseExecutionEngine] = None
plugin_manager: Optional[PluginManager] = None
database: Optional[TradingDatabase] = None

# Initialize components
def initialize_components(config_path: str = None):
    """Initialize trading bot components"""
    global data_collector, technical_analyzer, sentiment_analyzer, signal_generator, execution_engine, plugin_manager, database
    
    # Load configuration
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "binance": {
                "api_key": "",
                "api_secret": "",
                "testnet": True
            },
            "database_path": "data/trading_bot.db"
        }
    
    # Initialize plugin manager
    plugin_manager = PluginManager()
    
    # Initialize database
    database = TradingDatabase(config.get('database_path', 'data/trading_bot.db'))
    
    # Initialize components with default implementations
    # In a real implementation, these would be loaded based on configuration
    from modules.data_collector import BinanceDataCollector
    from modules.market_analysis import TechnicalAnalyzer
    from modules.sentiment_engine import SentimentAnalyzer
    from modules.signal_generator import SignalGenerator
    from modules.execution_engine import ExecutionEngine
    
    data_collector = BinanceDataCollector(
        config['binance']['api_key'],
        config['binance']['api_secret']
    )
    
    technical_analyzer = TechnicalAnalyzer(plugin_manager)
    
    sentiment_analyzer = SentimentAnalyzer(
        "", "", "", "", "", ""  # API keys would come from config
    )
    
    signal_generator = SignalGenerator()
    
    execution_engine = ExecutionEngine(
        config['binance']['api_key'],
        config['binance']['api_secret'],
        testnet=config['binance'].get('testnet', True)
    )
    
    logger.info("Trading bot components initialized")

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/plugins', methods=['GET'])
def list_plugins():
    """List all available plugins"""
    if not plugin_manager:
        initialize_components()
    
    return jsonify({
        'indicators': plugin_manager.list_indicators(),
        'strategies': plugin_manager.list_strategies(),
        'data_sources': plugin_manager.list_data_sources(),
        'sentiment_sources': plugin_manager.list_sentiment_sources()
    })

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    """Get market data for a symbol"""
    if not data_collector:
        initialize_components()
    
    symbol = request.args.get('symbol')
    interval = request.args.get('interval', '45m')
    limit = int(request.args.get('limit', 100))
    
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400
    
    try:
        df = data_collector.get_historical_klines(symbol, interval, limit)
        
        # Convert DataFrame to JSON-serializable format
        data = df.to_dict('records')
        
        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'data': data
        })
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/technical-analysis', methods=['POST'])
def perform_technical_analysis():
    """Perform technical analysis on market data"""
    if not technical_analyzer:
        initialize_components()
    
    data = request.json
    symbol = data.get('symbol')
    interval = data.get('interval', '45m')
    limit = int(data.get('limit', 100))
    
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400
    
    try:
        # Get market data
        df = data_collector.get_historical_klines(symbol, interval, limit)
        
        # Add technical indicators
        df = technical_analyzer.add_all_indicators(df)
        
        # Identify support and resistance
        support, resistance = technical_analyzer.identify_support_resistance(df)
        
        # Detect candlestick patterns
        patterns = technical_analyzer.detect_candlestick_patterns(df)
        
        # Convert to JSON-serializable format
        result = {
            'symbol': symbol,
            'interval': interval,
            'data': df.to_dict('records'),
            'support_levels': support,
            'resistance_levels': resistance,
            'patterns': patterns
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error performing technical analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment-analysis', methods=['POST'])
def perform_sentiment_analysis():
    """Perform sentiment analysis for symbols"""
    if not sentiment_analyzer:
        initialize_components()
    
    data = request.json
    symbols = data.get('symbols', [])
    
    if not symbols:
        return jsonify({'error': 'Symbols parameter is required'}), 400
    
    try:
        sentiment_data = sentiment_analyzer.get_market_sentiment_summary(symbols)
        return jsonify(sentiment_data)
    except Exception as e:
        logger.error(f"Error performing sentiment analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-signals', methods=['POST'])
def generate_signals():
    """Generate trading signals"""
    if not signal_generator:
        initialize_components()
    
    data = request.json
    symbol = data.get('symbol')
    interval = data.get('interval', '45m')
    limit = int(data.get('limit', 100))
    strategy = data.get('strategy')
    
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400
    
    try:
        # Get market data
        df = data_collector.get_historical_klines(symbol, interval, limit)
        
        # Add technical indicators
        df = technical_analyzer.add_all_indicators(df)
        
        # Get sentiment data
        sentiment_data = sentiment_analyzer.get_market_sentiment_summary([symbol.replace('USDT', '')])
        
        # Use specified strategy if provided
        if strategy and plugin_manager and strategy in plugin_manager.list_strategies():
            strategy_instance = plugin_manager.get_strategy(strategy)
            signals = strategy_instance.generate_signals(df, sentiment_data)
            filtered_signals = strategy_instance.filter_signals(signals)
        else:
            # Use default signal generator
            signals = signal_generator.generate_signals(df, sentiment_data)
            filtered_signals = signal_generator.filter_signals(signals)
        
        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'signals': filtered_signals
        })
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/execute-trade', methods=['POST'])
def execute_trade():
    """Execute a trade"""
    if not execution_engine:
        initialize_components()
    
    data = request.json
    signal = data.get('signal')
    risk_percent = float(data.get('risk_percent', 1.0))
    leverage = int(data.get('leverage', 5))
    paper_trade = data.get('paper_trade', False)
    
    if not signal:
        return jsonify({'error': 'Signal parameter is required'}), 400
    
    try:
        if paper_trade:
            result = execution_engine.paper_trade(signal, risk_percent, leverage)
        else:
            result = execution_engine.execute_trade(signal, risk_percent, leverage)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get open positions"""
    if not execution_engine:
        initialize_components()
    
    try:
        positions = execution_engine.get_open_positions()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade-history', methods=['GET'])
def get_trade_history():
    """Get trade history"""
    if not execution_engine:
        initialize_components()
    
    symbol = request.args.get('symbol')
    limit = int(request.args.get('limit', 100))
    
    try:
        history = execution_engine.get_trade_history(symbol=symbol, limit=limit)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics"""
    if not database:
        initialize_components()
    
    symbol = request.args.get('symbol')
    
    try:
        metrics = database.get_performance_metrics(symbol=symbol)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/close-position', methods=['POST'])
def close_position():
    """Close a position"""
    if not execution_engine:
        initialize_components()
    
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'Symbol parameter is required'}), 400
    
    try:
        result = execution_engine.close_position(symbol)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error closing position: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Run the API
    app.run(debug=True, host='0.0.0.0', port=5000)
