import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tweepy
import newspaper
from textblob import TextBlob
from typing import Dict, List, Optional
from interfaces.base_components import BaseSentimentAnalyzer

class SentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, gemini_api_key, perplexity_api_key, twitter_api_key, twitter_api_secret, 
                 twitter_access_token, twitter_access_token_secret):
        self.gemini_api_key = gemini_api_key
        self.perplexity_api_key = perplexity_api_key
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        # Twitter API setup
        self.twitter_auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
        self.twitter_auth.set_access_token(twitter_access_token, twitter_access_token_secret)
        self.twitter_api = tweepy.API(self.twitter_auth)
        
        # News sources
        self.crypto_news_sources = [
            "https://cointelegraph.com",
            "https://coindesk.com",
            "https://decrypt.co",
            "https://www.theblock.co",
        ]
    
    def analyze_with_gemini(self, text):
        """Analyze text sentiment using Google Gemini Pro"""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key,
        }
        
        prompt = f"""
        Analyze the sentiment of the following text related to cryptocurrency trading.
        Provide a sentiment score from -1 (extremely bearish) to 1 (extremely bullish),
        identify key market-moving factors mentioned, and assess the potential impact
        on cryptocurrency prices in the short term (next 45 minutes).
        
        Text: {text}
        
        Format your response as JSON with the following structure:
        {{
            "sentiment_score": <float between -1 and 1>,
            "confidence": <float between 0 and 1>,
            "key_factors": [<list of key factors>],
            "potential_impact": <description of potential impact>,
            "volatility_forecast": <float between 0 and 1 indicating expected volatility>
        }}
        """
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(self.gemini_url, headers=headers, json=data)
        
        if response.status_code == 200:
            try:
                result = response.json()
                # Extract the text from the response
                response_text = result['candidates'][0]['content']['parts'][0]['text']
                # Parse the JSON from the response text
                return json.loads(response_text)
            except Exception as e:
                print(f"Error parsing Gemini response: {e}")
                return None
        else:
            print(f"Error calling Gemini API: {response.status_code} - {response.text}")
            return None
    
    def query_perplexity(self, query):
        """Query Perplexity Pro for market insights"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.perplexity_api_key}"
        }
        
        data = {
            "model": "pplx-7b-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cryptocurrency market analyst. Provide concise, data-driven insights about market conditions, focusing on factors that could affect prices in the next 45 minutes."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }
        
        response = requests.post("https://api.perplexity.ai/chat/completions", headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"Error calling Perplexity API: {response.status_code} - {response.text}")
            return None
    
    def get_twitter_sentiment(self, query, count=100):
        """Analyze sentiment from Twitter"""
        tweets = []
        
        try:
            # Search for tweets
            tweet_search = self.twitter_api.search_tweets(
                q=query, 
                lang="en", 
                count=count,
                tweet_mode='extended'
            )
            
            for tweet in tweet_search:
                tweets.append({
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                })
        except Exception as e:
            print(f"Error fetching tweets: {e}")
        
        # Analyze sentiment of tweets
        sentiment_scores = []
        for tweet in tweets:
            analysis = TextBlob(tweet['text'])
            sentiment_scores.append(analysis.sentiment.polarity)
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
        else:
            avg_sentiment = 0
            sentiment_std = 0
        
        return {
            'average_sentiment': avg_sentiment,
            'sentiment_std': sentiment_std,
            'tweet_count': len(tweets),
            'tweets': tweets[:10]  # Return top 10 tweets for reference
        }
    
    def get_news_sentiment(self, symbols):
        """Get and analyze news sentiment for given symbols"""
        articles = []
        
        for source in self.crypto_news_sources:
            try:
                paper = newspaper.build(source, memoize_articles=False)
                for article in paper.articles:
                    if any(symbol.lower() in article.url for symbol in symbols):
                        try:
                            article.download()
                            article.parse()
                            articles.append({
                                'title': article.title,
                                'text': article.text,
                                'url': article.url,
                                'published_date': article.publish_date
                            })
                        except Exception as e:
                            print(f"Error processing article: {e}")
            except Exception as e:
                print(f"Error processing news source {source}: {e}")
        
        # Analyze sentiment of articles
        news_sentiments = []
        for article in articles:
            # Use Gemini for more sophisticated sentiment analysis
            gemini_analysis = self.analyze_with_gemini(article['title'] + " " + article['text'][:500])
            if gemini_analysis:
                news_sentiments.append({
                    'article': article,
                    'sentiment': gemini_analysis
                })
        
        return news_sentiments
    
    def get_market_sentiment_summary(self, symbols):
        """Get comprehensive market sentiment summary"""
        # Get Twitter sentiment
        twitter_sentiment = self.get_twitter_sentiment(" OR ".join(symbols))
        
        # Get news sentiment
        news_sentiment = self.get_news_sentiment(symbols)
        
        # Get market insights from Perplexity
        perplexity_query = f"Current market conditions for {', '.join(symbols)} cryptocurrencies. Focus on short-term price drivers for the next 45 minutes."
        perplexity_insights = self.query_perplexity(perplexity_query)
        
        # Calculate overall sentiment score
        sentiment_components = []
        
        if twitter_sentiment and twitter_sentiment['tweet_count'] > 0:
            sentiment_components.append(('twitter', twitter_sentiment['average_sentiment']))
        
        if news_sentiment:
            news_scores = [item['sentiment']['sentiment_score'] for item in news_sentiment]
            avg_news_sentiment = np.mean(news_scores) if news_scores else 0
            sentiment_components.append(('news', avg_news_sentiment))
        
        # Calculate weighted average sentiment
        if sentiment_components:
            weights = {'twitter': 0.4, 'news': 0.6}
            weighted_sentiment = sum(score * weights[source] for source, score in sentiment_components)
        else:
            weighted_sentiment = 0
        
        return {
            'overall_sentiment': weighted_sentiment,
            'twitter_sentiment': twitter_sentiment,
            'news_sentiment': news_sentiment,
            'perplexity_insights': perplexity_insights,
            'timestamp': datetime.now()
        }
