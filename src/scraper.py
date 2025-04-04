"""
Advanced Scraper for Mak: Dynamic content collection with quality filtering and self-optimizing strategies
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import instaloader
from tiktok_api import TikTokApi
import praw  # Reddit API
import tweepy  # Twitter API
import schedule
import time
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)

class QualityFilter:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.known_good_patterns = [
            "hook", "CTA", "conversion", "viral", "engagement",
            "storytelling", "emotional", "urgency", "social proof"
        ]
        
    def score_content(self, content: str) -> float:
        """Score content quality based on various metrics."""
        try:
            # Basic metrics
            length_score = min(len(content.split()) / 100, 1.0)  # Normalize to 0-1
            pattern_score = sum(1 for pattern in self.known_good_patterns if pattern in content.lower()) / len(self.known_good_patterns)
            
            # Engagement metrics (if available)
            engagement_score = 0.5  # Default if no engagement data
            
            # Combine scores
            final_score = (length_score * 0.3 + pattern_score * 0.4 + engagement_score * 0.3)
            return final_score
        except Exception as e:
            logging.error(f"Error scoring content: {e}")
            return 0.0

class DynamicStrategy:
    def __init__(self):
        self.strategy_history = []
        self.performance_metrics = {}
        
    def update_strategy(self, platform: str, performance: float):
        """Update scraping strategy based on performance."""
        self.strategy_history.append({
            "platform": platform,
            "performance": performance,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 entries
        if len(self.strategy_history) > 100:
            self.strategy_history = self.strategy_history[-100:]
            
        # Calculate platform performance
        platform_performance = {}
        for entry in self.strategy_history:
            if entry["platform"] not in platform_performance:
                platform_performance[entry["platform"]] = []
            platform_performance[entry["platform"]].append(entry["performance"])
            
        # Update metrics
        self.performance_metrics = {
            platform: np.mean(scores) for platform, scores in platform_performance.items()
        }
        
    def get_platform_priority(self) -> List[str]:
        """Get platform priority based on historical performance."""
        if not self.performance_metrics:
            return ["YouTube", "Instagram", "TikTok"]  # Temporarily removed Reddit and Twitter
        
        return sorted(
            self.performance_metrics.keys(),
            key=lambda x: self.performance_metrics[x],
            reverse=True
        )

class ContentScraper:
    def __init__(self):
        self.insta_loader = instaloader.Instaloader()
        self.tiktok_api = TikTokApi()
        # Temporarily disable Reddit and Twitter
        # self.reddit = praw.Reddit(
        #     client_id=os.getenv("REDDIT_CLIENT_ID"),
        #     client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        #     user_agent="MakAI/1.0"
        # )
        # self.twitter_api = tweepy.Client(
        #     bearer_token=os.getenv("TWITTER_BEARER_TOKEN")
        # )
        self.quality_filter = QualityFilter()
        self.strategy = DynamicStrategy()
        self.training_data = []
        
    def scrape_youtube(self, video_url: str, min_views: int = 10000) -> Optional[Dict]:
        """Scrape content from YouTube video with quality filtering."""
        try:
            yt = YouTube(video_url)
            if yt.views < min_views:
                return None
                
            transcript = YouTubeTranscriptApi.get_transcript(yt.video_id)
            
            content = {
                "title": yt.title,
                "description": yt.description,
                "transcript": " ".join([t["text"] for t in transcript]),
                "views": yt.views,
                "length": yt.length,
                "likes": yt.rating,
                "comments": yt.comments
            }
            
            # Score content quality
            content["quality_score"] = self.quality_filter.score_content(
                f"{content['title']} {content['description']} {content['transcript']}"
            )
            
            return content
        except Exception as e:
            logging.error(f"Error scraping YouTube video {video_url}: {e}")
            return None

    def scrape_instagram(self, post_url: str, min_likes: int = 1000) -> Optional[Dict]:
        """Scrape content from Instagram post with quality filtering."""
        try:
            post = instaloader.Post.from_shortcode(self.insta_loader.context, post_url.split("/")[-2])
            if post.likes < min_likes:
                return None
                
            content = {
                "caption": post.caption,
                "likes": post.likes,
                "comments": post.comments,
                "hashtags": [tag for tag in post.caption.split() if tag.startswith("#")]
            }
            
            content["quality_score"] = self.quality_filter.score_content(content["caption"])
            return content
        except Exception as e:
            logging.error(f"Error scraping Instagram post {post_url}: {e}")
            return None

    def scrape_tiktok(self, video_url: str, min_views: int = 10000) -> Optional[Dict]:
        """Scrape content from TikTok video with quality filtering."""
        try:
            video = self.tiktok_api.get_video_by_url(video_url)
            if video.stats["views"] < min_views:
                return None
                
            content = {
                "description": video.description,
                "music": video.music,
                "stats": video.stats,
                "hashtags": [tag for tag in video.description.split() if tag.startswith("#")]
            }
            
            content["quality_score"] = self.quality_filter.score_content(content["description"])
            return content
        except Exception as e:
            logging.error(f"Error scraping TikTok video {video_url}: {e}")
            return None

    def scrape_reddit(self, subreddit: str, min_score: int = 100) -> List[Dict]:
        """Scrape top posts from Reddit with quality filtering."""
        # Temporarily disabled
        logging.info("Reddit scraping is temporarily disabled")
        return []

    def scrape_twitter(self, query: str, min_likes: int = 100) -> List[Dict]:
        """Scrape tweets with quality filtering."""
        # Temporarily disabled
        logging.info("Twitter scraping is temporarily disabled")
        return []

    def format_training_data(self, content: Dict, source: str) -> str:
        """Format scraped content into training data with quality score."""
        if not content or content.get("quality_score", 0) < 0.5:
            return None
            
        prompt = f"### Input:\nAnalyze this {source} content (Quality Score: {content['quality_score']:.2f}):\n"
        
        if source == "YouTube":
            prompt += f"Title: {content['title']}\nDescription: {content['description']}\nTranscript: {content['transcript'][:500]}...\n"
        elif source == "Instagram":
            prompt += f"Caption: {content['caption']}\nHashtags: {', '.join(content['hashtags'])}\n"
        elif source == "TikTok":
            prompt += f"Description: {content['description']}\nMusic: {content['music']}\nHashtags: {', '.join(content['hashtags'])}\n"
        elif source == "Reddit":
            prompt += f"Title: {content['title']}\nText: {content['text'][:500]}...\n"
        else:  # Twitter
            prompt += f"Tweet: {content['text']}\n"
            
        prompt += "\n### Response:\nThis content performed well because it used effective hooks, emotional triggers, and clear CTAs. The key elements that made it viral were..."
        
        return prompt

    def save_training_data(self):
        """Save collected training data to train.txt with quality filtering."""
        if not self.training_data:
            logging.warning("No training data collected in this run")
            return
            
        # Sort by quality score
        self.training_data.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        # Take top 50% by quality
        top_data = self.training_data[:len(self.training_data)//2]
            
        os.makedirs("data", exist_ok=True)
        with open("data/train.txt", "a", encoding="utf-8") as f:
            for item in top_data:
                f.write(item["formatted"] + "\n")
                
        logging.info(f"Saved {len(top_data)} high-quality training examples")

    def run_scraper(self):
        """Main scraper function with dynamic strategy and quality filtering."""
        try:
            # Get platform priority based on performance
            platforms = self.strategy.get_platform_priority()
            
            for platform in platforms:
                try:
                    if platform == "YouTube":
                        # Example YouTube URLs - replace with actual discovery logic
                        urls = ["https://www.youtube.com/watch?v=example1"]
                        for url in urls:
                            content = self.scrape_youtube(url)
                            if content:
                                formatted = self.format_training_data(content, "YouTube")
                                if formatted:
                                    self.training_data.append({
                                        "content": content,
                                        "formatted": formatted,
                                        "quality_score": content["quality_score"]
                                    })
                                    
                    elif platform == "Instagram":
                        # Example Instagram URLs
                        urls = ["https://www.instagram.com/p/example1"]
                        for url in urls:
                            content = self.scrape_instagram(url)
                            if content:
                                formatted = self.format_training_data(content, "Instagram")
                                if formatted:
                                    self.training_data.append({
                                        "content": content,
                                        "formatted": formatted,
                                        "quality_score": content["quality_score"]
                                    })
                                    
                    elif platform == "TikTok":
                        # Example TikTok URLs
                        urls = ["https://www.tiktok.com/@user/video/example1"]
                        for url in urls:
                            content = self.scrape_tiktok(url)
                            if content:
                                formatted = self.format_training_data(content, "TikTok")
                                if formatted:
                                    self.training_data.append({
                                        "content": content,
                                        "formatted": formatted,
                                        "quality_score": content["quality_score"]
                                    })
                    
                    # Update strategy based on performance
                    platform_performance = len([d for d in self.training_data if d["content"].get("source") == platform])
                    self.strategy.update_strategy(platform, platform_performance)
                    
                except Exception as e:
                    logging.error(f"Error scraping {platform}: {e}")
                    continue
                    
            # Save collected data
            self.save_training_data()
            
        except Exception as e:
            logging.error(f"Error in scraper run: {e}")

def main():
    scraper = ContentScraper()
    
    # Schedule scraper to run every 6 hours
    schedule.every(6).hours.do(scraper.run_scraper)
    
    # Run immediately on startup
    scraper.run_scraper()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 