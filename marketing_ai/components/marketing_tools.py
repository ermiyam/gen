import tweepy
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import pandas as pd
from transformers import pipeline

class MarketingTools:
    def __init__(self, twitter_bearer_token: str, google_api_key: str):
        self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
        self.google_api_key = google_api_key
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.topic_classifier = pipeline("zero-shot-classification")

    def analyze_social_media(self, brand_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze brand mentions across social media."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get tweets about the brand
        tweets = self.twitter_client.search_recent_tweets(
            query=brand_name,
            start_time=start_time,
            end_time=end_time,
            max_results=100
        )
        
        if not tweets.data:
            return {"error": "No tweets found"}
            
        # Analyze sentiment and topics
        results = []
        for tweet in tweets.data:
            sentiment = self.sentiment_analyzer(tweet.text)[0]
            topics = self.topic_classifier(
                tweet.text,
                candidate_labels=["product", "service", "price", "quality", "support"]
            )
            
            results.append({
                "text": tweet.text,
                "sentiment": sentiment,
                "topics": topics,
                "engagement": tweet.public_metrics,
                "timestamp": tweet.created_at
            })
            
        return {
            "total_mentions": len(results),
            "sentiment_distribution": self._calculate_sentiment_distribution(results),
            "topic_distribution": self._calculate_topic_distribution(results),
            "engagement_metrics": self._calculate_engagement_metrics(results),
            "detailed_results": results
        }

    def analyze_campaign_performance(self, campaign_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze marketing campaign performance metrics."""
        df = pd.DataFrame(campaign_data)
        
        metrics = {
            "total_reach": df["reach"].sum(),
            "total_engagement": df["engagement"].sum(),
            "conversion_rate": df["conversions"].sum() / df["reach"].sum() if df["reach"].sum() > 0 else 0,
            "cost_per_conversion": df["cost"].sum() / df["conversions"].sum() if df["conversions"].sum() > 0 else 0,
            "roi": (df["revenue"].sum() - df["cost"].sum()) / df["cost"].sum() if df["cost"].sum() > 0 else 0
        }
        
        # Calculate trends
        df["date"] = pd.to_datetime(df["date"])
        daily_metrics = df.groupby("date").agg({
            "reach": "sum",
            "engagement": "sum",
            "conversions": "sum",
            "cost": "sum",
            "revenue": "sum"
        }).reset_index()
        
        return {
            "overall_metrics": metrics,
            "daily_trends": daily_metrics.to_dict("records"),
            "channel_performance": self._analyze_channel_performance(df)
        }

    def generate_content_recommendations(self, 
                                      target_audience: Dict[str, Any],
                                      content_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate content recommendations based on audience and history."""
        # Analyze successful content patterns
        successful_content = [c for c in content_history if c["performance_score"] > 0.7]
        
        recommendations = []
        for content in successful_content:
            # Adapt content for target audience
            adapted_content = self._adapt_content_for_audience(content, target_audience)
            recommendations.append({
                "original_content": content,
                "adapted_content": adapted_content,
                "recommended_channels": self._determine_channels(target_audience),
                "timing_recommendations": self._generate_timing_recommendations(target_audience)
            })
            
        return recommendations

    def _calculate_sentiment_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sentiment distribution from results."""
        sentiments = [r["sentiment"]["label"] for r in results]
        total = len(sentiments)
        return {
            "positive": sentiments.count("POSITIVE") / total,
            "negative": sentiments.count("NEGATIVE") / total,
            "neutral": sentiments.count("NEUTRAL") / total if "NEUTRAL" in sentiments else 0
        }

    def _calculate_topic_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate topic distribution from results."""
        topics = {}
        for r in results:
            for topic, score in zip(r["topics"]["labels"], r["topics"]["scores"]):
                topics[topic] = topics.get(topic, 0) + score
                
        total = len(results)
        return {topic: score/total for topic, score in topics.items()}

    def _calculate_engagement_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate engagement metrics from results."""
        total_likes = sum(r["engagement"]["like_count"] for r in results)
        total_retweets = sum(r["engagement"]["retweet_count"] for r in results)
        total_replies = sum(r["engagement"]["reply_count"] for r in results)
        
        return {
            "average_likes": total_likes / len(results),
            "average_retweets": total_retweets / len(results),
            "average_replies": total_replies / len(results)
        }

    def _analyze_channel_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance metrics by channel."""
        channel_metrics = {}
        for channel in df["channel"].unique():
            channel_data = df[df["channel"] == channel]
            channel_metrics[channel] = {
                "reach": channel_data["reach"].sum(),
                "engagement": channel_data["engagement"].sum(),
                "conversion_rate": channel_data["conversions"].sum() / channel_data["reach"].sum(),
                "cost_per_conversion": channel_data["cost"].sum() / channel_data["conversions"].sum()
            }
        return channel_metrics

    def _adapt_content_for_audience(self, 
                                  content: Dict[str, Any],
                                  target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt content for specific target audience."""
        # Implement content adaptation logic based on audience demographics and preferences
        return {
            "title": content["title"],
            "content": content["content"],
            "tone": self._determine_appropriate_tone(target_audience),
            "keywords": self._generate_keywords(target_audience),
            "call_to_action": self._generate_cta(target_audience)
        }

    def _determine_channels(self, target_audience: Dict[str, Any]) -> List[str]:
        """Determine appropriate channels based on audience."""
        channels = []
        demographics = target_audience.get("demographics", {})
        
        if demographics.get("age", {}).get("18-24", 0) > 0.3:
            channels.extend(["instagram", "tiktok"])
        if demographics.get("age", {}).get("25-34", 0) > 0.3:
            channels.extend(["linkedin", "twitter"])
        if demographics.get("age", {}).get("35-44", 0) > 0.3:
            channels.extend(["facebook", "linkedin"])
            
        return list(set(channels))

    def _generate_timing_recommendations(self, target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Generate timing recommendations based on audience."""
        return {
            "best_days": ["Monday", "Wednesday", "Friday"],
            "best_times": ["9:00 AM", "2:00 PM", "6:00 PM"],
            "frequency": "3-4 times per week"
        }

    def _determine_appropriate_tone(self, target_audience: Dict[str, Any]) -> str:
        """Determine appropriate tone based on audience."""
        if target_audience.get("professional", False):
            return "formal"
        return "casual"

    def _generate_keywords(self, target_audience: Dict[str, Any]) -> List[str]:
        """Generate relevant keywords based on audience."""
        return ["marketing", "brand", "engagement", "growth"]

    def _generate_cta(self, target_audience: Dict[str, Any]) -> str:
        """Generate appropriate call-to-action based on audience."""
        return "Learn More"

# Example usage
if __name__ == "__main__":
    tools = MarketingTools("YOUR_TWITTER_BEARER_TOKEN", "YOUR_GOOGLE_API_KEY")
    
    # Test social media analysis
    brand_analysis = tools.analyze_social_media("ExampleBrand")
    print("Brand Analysis:", json.dumps(brand_analysis, indent=2))
    
    # Test campaign performance analysis
    campaign_data = [
        {
            "date": "2024-03-15",
            "channel": "facebook",
            "reach": 1000,
            "engagement": 100,
            "conversions": 50,
            "cost": 500,
            "revenue": 2000
        }
    ]
    campaign_analysis = tools.analyze_campaign_performance(campaign_data)
    print("Campaign Analysis:", json.dumps(campaign_analysis, indent=2))
    
    # Test content recommendations
    target_audience = {
        "demographics": {
            "age": {"25-34": 0.4, "35-44": 0.3},
            "income": "high"
        },
        "interests": ["technology", "business"],
        "professional": True
    }
    content_history = [
        {
            "title": "Product Launch",
            "content": "New product features...",
            "performance_score": 0.8
        }
    ]
    recommendations = tools.generate_content_recommendations(target_audience, content_history)
    print("Content Recommendations:", json.dumps(recommendations, indent=2)) 