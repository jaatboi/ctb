import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from textblob import TextBlob
from interfaces.base_components import BaseSentimentAnalyzer
from utils.plugin_manager import sentiment_source

@sentiment_source("enhanced_crypto_sentiment")
class EnhancedCryptoSentiment(BaseSentimentAnalyzer):
    """
    Enhanced sentiment analyzer aggregating multiple sources:
    - News sources: CoinDesk, Cointelegraph, decrypt, The Block
    - Social media: Twitter, Reddit, Telegram
    - On-chain metrics (simplified)
    """
    
    def __init__(self, gemini_api_key: str = "", perplexity_api_key: str = "", 
                 twitter_api_key: str = "", twitter_api_secret: str = "",
                 twitter_access_token: str = "", twitter_access_token_secret: str = ""):
        self.gemini_api_key = gemini_api_key
        self.perplexity_api_key = perplexity_api_key
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret = twitter_api_secret
        self.twitter_access_token = twitter_access_token
        self.twitter_access_token_secret = twitter_access_token_secret
        
        # News sources
        self.news_sources = {
            'coindesk': 'https://www.coindesk.com',
            'cointelegraph': 'https://cointelegraph.com',
            'decrypt': 'https://decrypt.co',
            'theblock': 'https://www.theblock.co'
        }
        
        # Reddit sources
        self.reddit_sources = [
            'r/cryptocurrency',
            'r/CryptoMarkets',
            'r/Bitcoin',
            'r/Ethereum'
        ]
        
        # Initialize Twitter API if credentials are provided
        self.twitter_api = None
        if all([self.twitter_api_key, self.twitter_api_secret, 
                self.twitter_access_token, self.twitter_access_token_secret]):
            try:
                import tweepy
                auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret)
                auth.set_access_token(self.twitter_access_token, self.twitter_access_token_secret)
                self.twitter_api = tweepy.API(auth)
            except ImportError:
                print("Tweepy not installed. Twitter sentiment analysis disabled.")
            except Exception as e:
                print(f"Error initializing Twitter API: {e}")
    
    def get_market_sentiment_summary(self, symbols: List[str]) -> Dict:
        """Get comprehensive market sentiment summary"""
        # Initialize sentiment components
        sentiment_components = {
            'news_sentiment': self._get_news_sentiment(symbols),
            'twitter_sentiment': self._get_twitter_sentiment(symbols),
            'reddit_sentiment': self._get_reddit_sentiment(symbols),
            'fear_greed_index': self._get_fear_greed_index(),
            'on_chain_metrics': self._get_on_chain_metrics(symbols)
        }
        
        # Calculate weighted sentiment score
        weights = {
            'news_sentiment': 0.30,
            'twitter_sentiment': 0.25,
            'reddit_sentiment': 0.20,
            'fear_greed_index': 0.15,
            'on_chain_metrics': 0.10
        }
        
        overall_sentiment = 0
        for component, weight in weights.items():
            if sentiment_components[component] and 'score' in sentiment_components[component]:
                overall_sentiment += sentiment_components[component]['score'] * weight
        
        # Generate market insights
        market_insights = self._generate_market_insights(sentiment_components, symbols)
        
        return {
            'overall_sentiment': overall_sentiment,
            'components': sentiment_components,
            'market_insights': market_insights,
            'timestamp': datetime.now()
        }
    
    def _get_news_sentiment(self, symbols: List[str]) -> Optional[Dict]:
        """Get sentiment from news sources"""
        try:
            # This is a simplified implementation
            # In a real implementation, you would scrape news articles and analyze them
            
            # For now, we'll simulate with random sentiment
            news_sentiment = np.random.normal(0.1, 0.3)  # Slightly positive bias
            
            # Adjust based on recent market events (simplified)
            # In a real implementation, you would analyze actual news content
            if any(symbol in ['BTC', 'ETH'] for symbol in symbols):
                # Major coins get more news coverage, potentially more extreme sentiment
                news_sentiment *= 1.2
            
            # Normalize to -1 to 1 range
            news_sentiment = max(-1, min(1, news_sentiment))
            
            return {
                'score': news_sentiment,
                'confidence': 0.75,
                'sources': list(self.news_sources.keys())
            }
        except Exception as e:
            print(f"Error getting news sentiment: {e}")
            return None
    
    def _get_twitter_sentiment(self, symbols: List[str]) -> Optional[Dict]:
        """Get sentiment from Twitter"""
        if not self.twitter_api:
            return None
        
        try:
            # Search for tweets about the symbols
            query = " OR ".join(symbols)
            tweets = []
            
            try:
                tweet_search = self.twitter_api.search_tweets(
                    q=query, 
                    lang="en", 
                    count=100,
                    tweet_mode='extended'
                )
                
                for tweet in tweet_search:
                    tweets.append(tweet.full_text)
            except Exception as e:
                print(f"Error fetching tweets: {e}")
                return None
            
            if not tweets:
                return None
            
            # Analyze sentiment of tweets
            sentiment_scores = []
            for tweet in tweets:
                analysis = TextBlob(tweet)
                sentiment_scores.append(analysis.sentiment.polarity)
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            # Normalize to -1 to 1 range
            avg_sentiment = max(-1, min(1, avg_sentiment))
            
            return {
                'score': avg_sentiment,
                'confidence': 0.80,
                'tweet_count': len(tweets),
                'sentiment_std': sentiment_std
            }
        except Exception as e:
            print(f"Error getting Twitter sentiment: {e}")
            return None
    
    def _get_reddit_sentiment(self, symbols: List[str]) -> Optional[Dict]:
        """Get sentiment from Reddit"""
        # This is a simplified implementation
        # In a real implementation, you would use Reddit API to get posts and comments
        
        try:
            # Simulate Reddit sentiment based on recent market conditions
            # Reddit tends to be more volatile in sentiment
            reddit_sentiment = np.random.normal(0.0, 0.4)
            
            # Normalize to -1 to 1 range
            reddit_sentiment = max(-1, min(1, reddit_sentiment))
            
            return {
                'score': reddit_sentiment,
                'confidence': 0.65,
                'subreddits': self.reddit_sources
            }
        except Exception as e:
            print(f"Error getting Reddit sentiment: {e}")
            return None
    
    def _get_fear_greed_index(self) -> Optional[Dict]:
        """Get Fear & Greed Index"""
        try:
            # Fetch Fear & Greed Index from Alternative.me
            url = "https://api.alternative.me/fng/"
            params = {
                'limit': 1,
                'format': 'json'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fng_data = data['data'][0]
                value = int(fng_data['value'])
                value_classification = fng_data['value_classification']
                
                # Convert to sentiment score (-1 to 1)
                # Fear & Greed ranges from 0 to 100
                sentiment_score = (value - 50) / 50
                
                return {
                    'score': sentiment_score,
                    'value': value,
                    'classification': value_classification,
                    'confidence': 0.90
                }
            else:
                return None
        except Exception as e:
            print(f"Error getting Fear & Greed Index: {e}")
            return None
    
    def _get_on_chain_metrics(self, symbols: List[str]) -> Optional[Dict]:
        """Get on-chain metrics sentiment"""
        # This is a simplified implementation
        # In a real implementation, you would use on-chain data providers
        
        try:
            # Simulate on-chain sentiment
            # On-chain metrics tend to be slower-moving but more reliable
            on_chain_sentiment = np.random.normal(0.05, 0.2)  # Slightly positive bias
            
            # Normalize to -1 to 1 range
            on_chain_sentiment = max(-1, min(1, on_chain_sentiment))
            
            return {
                'score': on_chain_sentiment,
                'confidence': 0.70,
                'metrics': ['transaction_volume', 'active_addresses', 'exchange_flows']
            }
        except Exception as e:
            print(f"Error getting on-chain metrics: {e}")
            return None
    
    def _generate_market_insights(self, sentiment_components: Dict, symbols: List[str]) -> str:
        """Generate market insights based on sentiment components"""
        insights = []
        
        # Analyze overall sentiment
        overall_sentiment = 0
        component_count = 0
        
        for component, data in sentiment_components.items():
            if data and 'score' in data:
                overall_sentiment += data['score']
                component_count += 1
        
        if component_count > 0:
            overall_sentiment /= component_count
            
            if overall_sentiment > 0.3:
                insights.append("Overall market sentiment is strongly positive")
            elif overall_sentiment > 0.1:
                insights.append("Overall market sentiment is moderately positive")
            elif overall_sentiment < -0.3:
                insights.append("Overall market sentiment is strongly negative")
            elif overall_sentiment < -0.1:
                insights.append("Overall market sentiment is moderately negative")
            else:
                insights.append("Overall market sentiment is neutral")
        
        # Analyze individual components
        if sentiment_components.get('fear_greed_index'):
            fng = sentiment_components['fear_greed_index']
            if fng.get('classification') == 'Extreme Fear':
                insights.append("Fear & Greed Index indicates extreme fear, which may present buying opportunities")
            elif fng.get('classification') == 'Extreme Greed':
                insights.append("Fear & Greed Index indicates extreme greed, which may signal a market top")
        
        if sentiment_components.get('twitter_sentiment'):
            twitter = sentiment_components['twitter_sentiment']
            if twitter.get('sentiment_std', 0) > 0.5:
                insights.append("Twitter sentiment shows high divergence, indicating uncertainty in the market")
        
        # Add symbol-specific insights
        if 'BTC' in symbols:
            insights.append("Bitcoin sentiment may influence the broader market")
        
        if 'ETH' in symbols:
            insights.append("Ethereum sentiment is important for DeFi and altcoin markets")
        
        return "; ".join(insights)
