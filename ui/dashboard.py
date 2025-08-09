import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json

class TradingBotDashboard:
    def __init__(self, signal_generator, execution_engine, port=8050):
        self.signal_generator = signal_generator
        self.execution_engine = execution_engine
        self.port = port
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Crypto Trading Bot Dashboard", style={'textAlign': 'center'}),
            
            # Control panel
            html.Div([
                html.H2("Control Panel"),
                html.Div([
                    html.Label("Select Symbol:"),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[
                            {'label': 'BTCUSDT', 'value': 'BTCUSDT'},
                            {'label': 'ETHUSDT', 'value': 'ETHUSDT'},
                            {'label': 'BNBUSDT', 'value': 'BNBUSDT'},
                            {'label': 'ADAUSDT', 'value': 'ADAUSDT'},
                            {'label': 'DOTUSDT', 'value': 'DOTUSDT'},
                            {'label': 'SOLUSDT', 'value': 'SOLUSDT'}
                        ],
                        value='BTCUSDT'
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Time Interval:"),
                    dcc.Dropdown(
                        id='interval-dropdown',
                        options=[
                            {'label': '15m', 'value': '15m'},
                            {'label': '30m', 'value': '30m'},
                            {'label': '45m', 'value': '45m'},
                            {'label': '1h', 'value': '1h'},
                            {'label': '4h', 'value': '4h'}
                        ],
                        value='45m'
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Risk Percentage:"),
                    dcc.Input(
                        id='risk-input',
                        type='number',
                        value=1.0,
                        min=0.1,
                        max=5.0,
                        step=0.1
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Leverage:"),
                    dcc.Input(
                        id='leverage-input',
                        type='number',
                        value=5,
                        min=1,
                        max=125,
                        step=1
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Trade Mode:"),
                    dcc.Dropdown(
                        id='mode-dropdown',
                        options=[
                            {'label': 'Live Trading', 'value': 'live'},
                            {'label': 'Paper Trading', 'value': 'paper'}
                        ],
                        value='paper'
                    ),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Button('Generate Signals', id='generate-button', n_clicks=0, style={'marginTop': '20px'}),
                html.Button('Execute Trade', id='execute-button', n_clicks=0, style={'marginTop': '20px', 'marginLeft': '10px'}),
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Signal display
            html.Div([
                html.H2("Current Signals"),
                html.Div(id='signal-display')
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Open positions
            html.Div([
                html.H2("Open Positions"),
                html.Div(id='positions-display')
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Trade history
            html.Div([
                html.H2("Trade History"),
                html.Div(id='history-display')
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Performance metrics
            html.Div([
                html.H2("Performance Metrics"),
                html.Div(id='metrics-display')
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Chart display
            html.Div([
                html.H2("Price Chart"),
                dcc.Graph(id='price-chart')
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px'}),
            
            # Interval component for auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            ),
            
            # Store for signals and positions
            dcc.Store(id='signals-store'),
            dcc.Store(id='positions-store'),
        ])
    
    def setup_callbacks(self):
        """Set up the dashboard callbacks"""
        
        @self.app.callback(
            Output('signals-store', 'data'),
            Input('generate-button', 'n_clicks'),
            State('symbol-dropdown', 'value'),
            State('interval-dropdown', 'value')
        )
        def generate_signals(n_clicks, symbol, interval):
            if n_clicks == 0:
                return None
            
            # Generate signals (this would be replaced with actual signal generation)
            signals = self.signal_generator.generate_signals(symbol, interval)
            
            return signals
        
        @self.app.callback(
            Output('signal-display', 'children'),
            Input('signals-store', 'data')
        )
        def display_signals(signals):
            if not signals:
                return "No signals generated"
            
            signal_cards = []
            for signal in signals:
                card = html.Div([
                    html.H3(f"Signal: {signal['signal_type']}"),
                    html.P(f"Symbol: {signal['symbol']}"),
                    html.P(f"Entry Price: {signal['entry_price']}"),
                    html.P(f"Stop Loss: {signal['stop_loss']}"),
                    html.P(f"Take Profit Levels: {', '.join(str(tp) for tp in signal['take_profit_levels'])}"),
                    html.P(f"Confidence: {signal['confidence']:.2f}"),
                    html.P(f"Rationale: {signal['rationale']}"),
                ], style={'padding': '10px', 'margin': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                signal_cards.append(card)
            
            return signal_cards
        
        @self.app.callback(
            Output('positions-store', 'data'),
            Input('execute-button', 'n_clicks'),
            Input('signals-store', 'data'),
            State('symbol-dropdown', 'value'),
            State('risk-input', 'value'),
            State('leverage-input', 'value'),
            State('mode-dropdown', 'value')
        )
        def execute_trade(n_clicks, signals, symbol, risk_percent, leverage, mode):
            if n_clicks == 0 or not signals:
                return None
            
            # Find the signal for the selected symbol
            signal = next((s for s in signals if s['symbol'] == symbol), None)
            
            if not signal:
                return None
            
            # Execute trade
            if mode == 'live':
                result = self.execution_engine.execute_trade(signal, risk_percent, leverage)
            else:
                result = self.execution_engine.paper_trade(signal, risk_percent, leverage)
            
            # Update positions
            positions = self.execution_engine.get_open_positions()
            
            return positions
        
        @self.app.callback(
            Output('positions-display', 'children'),
            Input('positions-store', 'data'),
            Input('interval-component', 'n_intervals')
        )
        def display_positions(positions, n_intervals):
            # Update positions
            self.execution_engine.update_positions()
            positions = self.execution_engine.get_open_positions()
            
            if not positions:
                return "No open positions"
            
            position_cards = []
            for symbol, position in positions.items():
                card = html.Div([
                    html.H3(f"Position: {position['position_type']} {symbol}"),
                    html.P(f"Entry Price: {position['entry_price']}"),
                    html.P(f"Current Price: {position.get('current_price', 'N/A')}"),
                    html.P(f"Stop Loss: {position['stop_loss']}"),
                    html.P(f"Take Profit Levels: {', '.join(str(tp) for tp in position['take_profit_levels'])}"),
                    html.P(f"Unrealized PnL: {position.get('unrealized_pnl', 'N/A')}"),
                    html.Button('Close Position', id={'type': 'close-button', 'index': symbol}, n_clicks=0),
                ], style={'padding': '10px', 'margin': '10px', 'border': '1px solid #ddd', 'borderRadius': '5px'})
                position_cards.append(card)
            
            return position_cards
        
        @self.app.callback(
            Output('history-display', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def display_trade_history(n_intervals):
            history = self.execution_engine.get_trade_history()
            
            if not history:
                return "No trade history"
            
            # Convert to DataFrame for display
            df = pd.DataFrame(history)
            
            # Format the data
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'close_timestamp' in df.columns:
                df['close_timestamp'] = pd.to_datetime(df['close_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Create table
            table = dash_table.DataTable(
                id='history-table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                page_size=10,
                sort_action="native",
                filter_action="native"
            )
            
            return table
        
        @self.app.callback(
            Output('metrics-display', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def display_metrics(n_intervals):
            metrics = self.execution_engine.get_performance_metrics()
            
            if not metrics:
                return "No performance metrics available"
            
            metrics = html.Div([
                html.Div([
                    html.H4("Total Trades"),
                    html.H2(metrics['total_trades'])
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("Win Rate"),
                    html.H2(f"{metrics['win_rate']:.2%}")
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("Total PnL"),
                    html.H2(f"{metrics['total_pnl']:.2f}")
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("Avg Win"),
                    html.H2(f"{metrics['avg_win']:.2f}")
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("Profit Factor"),
                    html.H2(f"{metrics['profit_factor']:.2f}")
                ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
            ])
            
            return metrics
        
        @self.app.callback(
            Output('price-chart', 'figure'),
            Input('symbol-dropdown', 'value'),
            Input('interval-dropdown', 'value'),
            Input('interval-component', 'n_intervals')
        )
        def update_price_chart(symbol, interval, n_intervals):
            # Get price data (this would be replaced with actual data fetching)
            # For now, we'll create sample data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='15T')
            prices = np.random.normal(loc=100, scale=5, size=100).cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=symbol
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                hovermode='x unified'
            )
            
            return fig
    
    def run(self, debug=True):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=self.port)
