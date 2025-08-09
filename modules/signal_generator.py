import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from typing import Dict, List, Optional
from interfaces.base_components import BaseSignalGenerator

class SignalGenerator(BaseSignalGenerator):
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.feature_columns = []
        
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def prepare_features(self, df, sentiment_data=None):
        """Prepare features for signal generation"""
        df_features = df.copy()
        
        # Price-based features
        df_features['price_change'] = df_features['close'].pct_change()
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_open_ratio'] = df_features['close'] / df_features['open']
        
        # Volume-based features
        df_features['volume_change'] = df_features['volume'].pct_change()
        df_features['volume_sma_ratio'] = df_features['volume'] / df_features['volume'].rolling(20).mean()
        
        # Technical indicator features
        df_features['rsi_overbought'] = (df_features['rsi'] > 70).astype(int)
        df_features['rsi_oversold'] = (df_features['rsi'] < 30).astype(int)
        df_features['macd_above_signal'] = (df_features['macd'] > df_features['macdsignal']).astype(int)
        df_features['price_above_sma20'] = (df_features['close'] > df_features['sma_20']).astype(int)
        df_features['price_above_sma50'] = (df_features['close'] > df_features['sma_50']).astype(int)
        df_features['price_above_bb_middle'] = (df_features['close'] > df_features['middle_band']).astype(int)
        df_features['price_above_bb_upper'] = (df_features['close'] > df_features['upper_band']).astype(int)
        df_features['price_below_bb_lower'] = (df_features['close'] < df_features['lower_band']).astype(int)
        
        # Pattern features
        if 'patterns' in df_features.columns:
            for pattern in df_features['patterns'].iloc[0].keys():
                df_features[f'pattern_{pattern}'] = df_features['patterns'].apply(lambda x: x.get(pattern, 0))
        
        # Add sentiment features if available
        if sentiment_data:
            df_features['sentiment_score'] = sentiment_data.get('overall_sentiment', 0)
            df_features['twitter_sentiment'] = sentiment_data.get('twitter_sentiment', {}).get('average_sentiment', 0)
            df_features['news_sentiment'] = 0
            
            if sentiment_data.get('news_sentiment'):
                news_scores = [item['sentiment']['sentiment_score'] for item in sentiment_data['news_sentiment']]
                df_features['news_sentiment'] = np.mean(news_scores) if news_scores else 0
        
        # Drop rows with NaN values
        df_features = df_features.dropna()
        
        # Define feature columns (excluding target and non-feature columns)
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                       'taker_buy_quote_asset_volume', 'ignore', 'patterns']
        
        self.feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        
        return df_features[self.feature_columns]
    
    def create_target_variable(self, df, future_periods=3):
        """Create target variable for prediction"""
        # Future price change
        df['future_return'] = df['close'].pct_change(future_periods).shift(-future_periods)
        
        # Classify as:
        # 2: Strong buy (future return > threshold_buy)
        # 1: Buy (0 < future return <= threshold_buy)
        # 0: Hold/Sell (future return <= 0)
        threshold_buy = 0.02  # 2% return
        
        df['target'] = 0
        df.loc[df['future_return'] > threshold_buy, 'target'] = 2
        df.loc[(df['future_return'] > 0) & (df['future_return'] <= threshold_buy), 'target'] = 1
        
        return df
    
    def train_model(self, df, sentiment_data=None):
        """Train the signal generation model"""
        # Prepare features
        df_features = self.prepare_features(df, sentiment_data)
        
        # Create target variable
        df_target = self.create_target_variable(df)
        
        # Combine features and target
        df_model = pd.concat([df_features, df_target['target']], axis=1).dropna()
        
        # Split data
        X = df_model[self.feature_columns]
        y = df_model['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy}")
        print(f"Classification report:\n{report}")
        
        # Save model
        if self.model_path:
            self.save_model()
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def save_model(self):
        """Save the trained model"""
        if self.model and self.model_path:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, self.model_path)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model"""
        if self.model_path and os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            print(f"Model loaded from {self.model_path}")
            return True
        return False
    
    def generate_signals(self, df, sentiment_data=None):
        """Generate trading signals"""
        if not self.model:
            print("No trained model available. Please train the model first.")
            return []
        
        # Prepare features
        df_features = self.prepare_features(df, sentiment_data)
        
        if df_features.empty:
            return []
        
        # Get the most recent data point
        latest_features = df_features.iloc[-1:].values
        
        # Generate prediction
        prediction = self.model.predict(latest_features)[0]
        prediction_proba = self.model.predict_proba(latest_features)[0]
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for stop loss and take profit levels
        atr = df['atr'].iloc[-1]
        
        # Determine signal type
        if prediction == 2:  # Strong buy
            signal_type = "STRONG_BUY"
        elif prediction == 1:  # Buy
            signal_type = "BUY"
        else:  # Hold/Sell
            signal_type = "HOLD"
        
        # Calculate entry, stop loss, and take profit levels
        if signal_type in ["BUY", "STRONG_BUY"]:
            entry = current_price
            stop_loss = entry - (1.5 * atr)  # 1.5 ATR below entry
            take_profit_levels = [
                entry + (1.0 * atr),  # TP1: 1 ATR above entry
                entry + (2.0 * atr),  # TP2: 2 ATR above entry
                entry + (3.0 * atr)   # TP3: 3 ATR above entry
            ]
        else:
            entry = None
            stop_loss = None
            take_profit_levels = []
        
        # Calculate probabilities for each take profit level
        tp_probabilities = []
        if signal_type in ["BUY", "STRONG_BUY"]:
            # Use the model's prediction probabilities as a base
            base_probability = prediction_proba[prediction] / 2  # Normalize
            
            # Decrease probability for further take profit levels
            tp_probabilities = [
                min(0.95, base_probability * 1.2),  # TP1 probability
                min(0.85, base_probability * 0.9),  # TP2 probability
                min(0.75, base_probability * 0.7)   # TP3 probability
            ]
        
        # Calculate stop loss probability
        sl_probability = 1 - prediction_proba[prediction] if signal_type in ["BUY", "STRONG_BUY"] else 0
        
        # Create signal object
        signal = {
            'timestamp': datetime.now(),
            'symbol': df.get('symbol', 'UNKNOWN'),
            'signal_type': signal_type,
            'entry_price': entry,
            'stop_loss': stop_loss,
            'take_profit_levels': take_profit_levels,
            'tp_probabilities': tp_probabilities,
            'sl_probability': sl_probability,
            'confidence': prediction_proba[prediction],
            'timeframe': '45m',
            'atr': atr,
            'rationale': self._generate_signal_rationale(df, sentiment_data, prediction, prediction_proba)
        }
        
        return [signal]
    
    def _generate_signal_rationale(self, df, sentiment_data, prediction, prediction_proba):
        """Generate rationale for the signal"""
        rationale = []
        
        # Add technical analysis rationale
        latest = df.iloc[-1]
        
        # RSI analysis
        if latest['rsi'] < 30:
            rationale.append(f"RSI indicates oversold conditions ({latest['rsi']:.2f})")
        elif latest['rsi'] > 70:
            rationale.append(f"RSI indicates overbought conditions ({latest['rsi']:.2f})")
        
        # MACD analysis
        if latest['macd'] > latest['macdsignal']:
            rationale.append("MACD is above signal line, indicating bullish momentum")
        else:
            rationale.append("MACD is below signal line, indicating bearish momentum")
        
        # Moving average analysis
        if latest['close'] > latest['sma_20'] > latest['sma_50']:
            rationale.append("Price is above both 20 and 50-period SMAs, indicating uptrend")
        elif latest['close'] < latest['sma_20'] < latest['sma_50']:
            rationale.append("Price is below both 20 and 50-period SMAs, indicating downtrend")
        
        # Bollinger Bands analysis
        if latest['close'] > latest['upper_band']:
            rationale.append("Price is above upper Bollinger Band, indicating potential overbought condition")
        elif latest['close'] < latest['lower_band']:
            rationale.append("Price is below lower Bollinger Band, indicating potential oversold condition")
        
        # Add sentiment analysis rationale
        if sentiment_data:
            overall_sentiment = sentiment_data.get('overall_sentiment', 0)
            if overall_sentiment > 0.3:
                rationale.append(f"Positive market sentiment detected (score: {overall_sentiment:.2f})")
            elif overall_sentiment < -0.3:
                rationale.append(f"Negative market sentiment detected (score: {overall_sentiment:.2f})")
            
            # Add key insights from Perplexity
            perplexity_insights = sentiment_data.get('perplexity_insights', '')
            if perplexity_insights:
                rationale.append(f"Market insights: {perplexity_insights[:100]}...")
        
        # Add model confidence
        rationale.append(f"Model confidence: {prediction_proba[prediction]:.2f}")
        
        return "; ".join(rationale)
    
    def filter_signals(self, signals, min_confidence=0.65, min_risk_reward_ratio=1.5):
        """Filter signals based on quality criteria"""
        filtered_signals = []
        
        for signal in signals:
            # Skip HOLD signals
            if signal['signal_type'] == 'HOLD':
                continue
            
            # Check confidence threshold
            if signal['confidence'] < min_confidence:
                continue
            
            # Calculate risk/reward ratio
            if signal['entry_price'] and signal['stop_loss'] and signal['take_profit_levels']:
                risk = abs(signal['entry_price'] - signal['stop_loss'])
                reward = abs(signal['take_profit_levels'][0] - signal['entry_price'])
                risk_reward_ratio = reward / risk if risk > 0 else 0
                
                if risk_reward_ratio < min_risk_reward_ratio:
                    continue
                
                # Add risk/reward ratio to signal
                signal['risk_reward_ratio'] = risk_reward_ratio
            else:
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
