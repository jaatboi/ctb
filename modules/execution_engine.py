import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
from binance import Client
from binance.enums import *
import logging
from utils.database import TradingDatabase

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, api_key, api_secret, testnet=True, db_path='data/trading_bot.db'):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        self.db = TradingDatabase(db_path)
        self.open_positions = {}
        self.load_open_positions()
        
    def load_open_positions(self):
        """Load open positions from database"""
        open_trades = self.db.get_trade_history(limit=1000)  # Get recent trades
        for trade in open_trades:
            if trade['status'] == 'OPEN':
                self.open_positions[trade['symbol']] = trade
    
    def get_account_info(self):
        """Get account information"""
        return self.client.get_account()
    
    def get_asset_balance(self, asset):
        """Get balance for a specific asset"""
        balance = self.client.get_asset_balance(asset=asset)
        return float(balance['free'])
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def get_futures_position(self, symbol):
        """Get current futures position for a symbol"""
        try:
            position = self.client.futures_position_information(symbol=symbol)[0]
            return {
                'symbol': position['symbol'],
                'position_amt': float(position['positionAmt']),
                'entry_price': float(position['entryPrice']),
                'mark_price': float(position['markPrice']),
                'unrealized_profit': float(position['unRealizedProfit']),
                'leverage': int(position['leverage']),
                'margin_type': position['marginType'],
                'isolated_margin': float(position['isolatedMargin']),
                'is_auto_add_margin': position['isAutoAddMargin'] == 'true',
                'percentage': float(position['percentage']),
                'liquidation_price': float(position['liquidationPrice']),
                'break_even_price': float(position['breakEvenPrice']),
            }
        except Exception as e:
            logger.error(f"Error getting futures position: {e}")
            return None
    
    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol"""
        try:
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            return response
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return None
    
    def set_margin_type(self, symbol, marginType):
        """Set margin type (ISOLATED or CROSSED) for a symbol"""
        try:
            response = self.client.futures_change_margin_type(
                symbol=symbol,
                marginType=marginType
            )
            return response
        except Exception as e:
            logger.error(f"Error setting margin type: {e}")
            return None
    
    def execute_trade(self, signal, risk_percent=1.0, leverage=5):
        """Execute a trade based on a signal"""
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit_levels = signal['take_profit_levels']
        
        # Get current position
        current_position = self.get_futures_position(symbol)
        position_amt = current_position['position_amt'] if current_position else 0
        
        # Skip if we already have a position in the opposite direction
        if (signal_type in ["BUY", "STRONG_BUY"] and position_amt > 0) or \
           (signal_type == "SELL" and position_amt < 0):
            return {
                'success': False,
                'message': f"Already have a {'long' if position_amt > 0 else 'short'} position for {symbol}"
            }
        
        # Close existing position if it's in the opposite direction
        if position_amt != 0:
            close_result = self.close_position(symbol)
            if not close_result['success']:
                return close_result
        
        # Set leverage
        self.set_leverage(symbol, leverage)
        
        # Calculate position size based on risk percentage
        account_info = self.client.futures_account()
        usdt_balance = float([asset for asset in account_info['assets'] if asset['asset'] == 'USDT'][0]['availableBalance'])
        
        # Calculate quantity
        risk_amount = usdt_balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        quantity = risk_amount / price_diff
        
        # Get symbol info for precision
        symbol_info = self.client.futures_exchange_info()
        symbol_data = next((s for s in symbol_info['symbols'] if s['symbol'] == symbol), None)
        
        if symbol_data:
            # Round quantity according to symbol's precision
            quantity_precision = int(symbol_data['quantityPrecision'])
            quantity = round(quantity, quantity_precision)
            
            # Round price according to symbol's precision
            price_precision = int(symbol_data['pricePrecision'])
            entry_price = round(entry_price, price_precision)
            stop_loss = round(stop_loss, price_precision)
            take_profit_levels = [round(tp, price_precision) for tp in take_profit_levels]
        
        # Execute the trade
        try:
            if signal_type in ["BUY", "STRONG_BUY"]:
                # Long position
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # Set stop loss and take profit orders
                # Stop loss
                self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_STOP_MARKET,
                    quantity=quantity,
                    stopPrice=stop_loss,
                    closePosition=True
                )
                
                # Take profit orders (split into multiple orders)
                for i, tp in enumerate(take_profit_levels):
                    tp_quantity = quantity * (0.4 if i == 0 else 0.3 if i == 1 else 0.3)  # 40%, 30%, 30%
                    tp_quantity = round(tp_quantity, quantity_precision)
                    
                    self.client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_LIMIT,
                        quantity=tp_quantity,
                        price=tp,
                        timeInForce=TIME_IN_FORCE_GTC
                    )
                
                position_type = "LONG"
            else:
                # Short position
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # Set stop loss and take profit orders
                # Stop loss
                self.client.futures_create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_STOP_MARKET,
                    quantity=quantity,
                    stopPrice=stop_loss,
                    closePosition=True
                )
                
                # Take profit orders (split into multiple orders)
                for i, tp in enumerate(take_profit_levels):
                    tp_quantity = quantity * (0.4 if i == 0 else 0.3 if i == 1 else 0.3)  # 40%, 30%, 30%
                    tp_quantity = round(tp_quantity, quantity_precision)
                    
                    self.client.futures_create_order(
                        symbol=symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_LIMIT,
                        quantity=tp_quantity,
                        price=tp,
                        timeInForce=TIME_IN_FORCE_GTC
                    )
                
                position_type = "SHORT"
            
            # Record the signal in the database
            signal_id = self.db.store_signal({
                **signal,
                'is_executed': True,
                'is_paper_trade': False
            })
            
            # Record the trade in the database
            trade_record = {
                'signal_id': signal_id,
                'symbol': symbol,
                'position_type': position_type,
                'entry_price': entry_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'take_profit_levels': take_profit_levels,
                'tp_probabilities': signal['tp_probabilities'],
                'sl_probability': signal['sl_probability'],
                'confidence': signal['confidence'],
                'rationale': signal['rationale'],
                'leverage': leverage,
                'risk_amount': risk_amount,
                'entry_timestamp': datetime.now(),
                'is_paper_trade': False,
                'status': 'OPEN'
            }
            
            trade_id = self.db.store_trade(trade_record)
            trade_record['id'] = trade_id
            
            # Update open positions
            self.open_positions[symbol] = trade_record
            
            return {
                'success': True,
                'message': f"{position_type} position opened for {symbol}",
                'trade': trade_record
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return {
                'success': False,
                'message': f"Error executing trade: {str(e)}"
            }
    
    def close_position(self, symbol):
        """Close an open position"""
        try:
            position = self.get_futures_position(symbol)
            if not position or float(position['positionAmt']) == 0:
                return {
                    'success': False,
                    'message': f"No open position for {symbol}"
                }
            
            position_amt = float(position['positionAmt'])
            side = SIDE_SELL if position_amt > 0 else SIDE_BUY
            quantity = abs(position_amt)
            
            # Close the position
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
                reduceOnly=True
            )
            
            # Cancel all open orders for this symbol
            self.client.futures_cancel_all_open_orders(symbol=symbol)
            
            # Update position record in database
            if symbol in self.open_positions:
                trade_record = self.open_positions[symbol]
                trade_id = trade_record['id']
                
                # Get current price
                current_price = self.get_current_price(symbol)
                
                # Calculate profit/loss
                if trade_record['position_type'] == 'LONG':
                    pnl = (current_price - trade_record['entry_price']) * trade_record['quantity']
                else:
                    pnl = (trade_record['entry_price'] - current_price) * trade_record['quantity']
                
                # Update trade in database
                self.db.update_trade(
                    trade_id,
                    exit_price=current_price,
                    exit_timestamp=datetime.now(),
                    pnl=pnl,
                    status='CLOSED'
                )
                
                # Update local record
                trade_record['close_timestamp'] = datetime.now()
                trade_record['close_price'] = current_price
                trade_record['status'] = 'CLOSED'
                trade_record['pnl'] = pnl
                
                # Move to trade history
                self.open_positions[symbol] = trade_record
            
            return {
                'success': True,
                'message': f"Position closed for {symbol}",
                'order_id': order['orderId']
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {
                'success': False,
                'message': f"Error closing position: {str(e)}"
            }
    
    def get_open_positions(self):
        """Get all open positions"""
        return self.open_positions
    
    def get_trade_history(self, symbol=None, limit=100):
        """Get trade history from database"""
        return self.db.get_trade_history(symbol=symbol, limit=limit)
    
    def get_performance_metrics(self, symbol=None):
        """Get performance metrics from database"""
        return self.db.get_performance_metrics(symbol=symbol)
    
    def update_positions(self):
        """Update open positions with current market data"""
        for symbol, position in self.open_positions.items():
            current_price = self.get_current_price(symbol)
            
            # Calculate unrealized PnL
            if position['position_type'] == 'LONG':
                unrealized_pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['quantity']
            
            # Update position
            position['current_price'] = current_price
            position['unrealized_pnl'] = unrealized_pnl
            
            # Check if stop loss or take profit is hit
            if position['position_type'] == 'LONG':
                if current_price <= position['stop_loss']:
                    self.close_position(symbol)
                elif current_price >= position['take_profit_levels'][-1]:
                    self.close_position(symbol)
            else:
                if current_price >= position['stop_loss']:
                    self.close_position(symbol)
                elif current_price <= position['take_profit_levels'][-1]:
                    self.close_position(symbol)
    
    def paper_trade(self, signal, risk_percent=1.0, leverage=5):
        """Simulate a trade without executing it"""
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit_levels = signal['take_profit_levels']
        
        # Simulate account balance
        account_balance = 10000  # Default paper trading balance
        
        # Calculate position size based on risk percentage
        risk_amount = account_balance * (risk_percent / 100)
        price_diff = abs(entry_price - stop_loss)
        quantity = risk_amount / price_diff
        
        # Record the signal in the database
        signal_id = self.db.store_signal({
            **signal,
            'is_executed': True,
            'is_paper_trade': True
        })
        
        # Record the paper trade in the database
        trade_record = {
            'signal_id': signal_id,
            'symbol': symbol,
            'position_type': 'LONG' if signal_type in ["BUY", "STRONG_BUY"] else 'SHORT',
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit_levels': take_profit_levels,
            'tp_probabilities': signal['tp_probabilities'],
            'sl_probability': signal['sl_probability'],
            'confidence': signal['confidence'],
            'rationale': signal['rationale'],
            'leverage': leverage,
            'risk_amount': risk_amount,
            'entry_timestamp': datetime.now(),
            'is_paper_trade': True,
            'status': 'OPEN'
        }
        
        trade_id = self.db.store_trade(trade_record)
        trade_record['id'] = trade_id
        
        # Update open positions
        self.open_positions[symbol] = trade_record
        
        return {
            'success': True,
            'message': f"Paper trade {'long' if signal_type in [\"BUY\", \"STRONG_BUY\"] else 'short'} position opened for {symbol}",
            'trade': trade_record
        }
    
    def close(self):
        """Close database connection"""
        self.db.close()
