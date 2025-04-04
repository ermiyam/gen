import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import subprocess
import requests
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from elevenlabs import generate, save
import pandas as pd
import plotly.express as px
from instagram_private_api import Client, ClientCompatPatch

class Platform(Enum):
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"

class ContentType(Enum):
    POST = "post"
    STORY = "story"
    REEL = "reel"
    CAROUSEL = "carousel"

@dataclass
class PostSchedule:
    platform: Platform
    content_type: ContentType
    date: datetime
    caption: str
    hashtags: List[str]
    media_path: Optional[str] = None

class MarketingTools:
    """Collection of marketing tools that Gen can use."""
    
    def __init__(
        self,
        config_dir: str = "config",
        output_dir: str = "output"
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load config
        self.config = self._load_config()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"marketing_tools_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_file = self.config_dir / "marketing_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default config
        config = {
            "instagram": {
                "username": "",
                "password": "",
                "api_key": ""
            },
            "tiktok": {
                "api_key": ""
            },
            "elevenlabs": {
                "api_key": ""
            }
        }
        
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def post_to_instagram(
        self,
        caption: str,
        media_path: str,
        hashtags: List[str] = None
    ) -> bool:
        """Post content to Instagram."""
        try:
            # Initialize Instagram API
            api = Client(
                self.config["instagram"]["username"],
                self.config["instagram"]["password"]
            )
            
            # Prepare caption with hashtags
            if hashtags:
                caption += "\n\n" + " ".join(f"#{tag}" for tag in hashtags)
            
            # Upload media
            if media_path.endswith(('.jpg', '.jpeg', '.png')):
                api.post_photo(media_path, caption=caption)
            elif media_path.endswith(('.mp4', '.mov')):
                api.post_video(media_path, caption=caption)
            
            self.logger.info(f"Posted to Instagram: {caption[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Error posting to Instagram: {str(e)}")
            return False
    
    def generate_voiceover(
        self,
        script: str,
        voice_id: str = "default",
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate voiceover from script using ElevenLabs."""
        try:
            # Generate audio
            audio = generate(
                text=script,
                voice=voice_id,
                api_key=self.config["elevenlabs"]["api_key"]
            )
            
            # Save audio
            if output_path is None:
                output_path = str(self.output_dir / f"voiceover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
            
            save(audio, output_path)
            
            self.logger.info(f"Generated voiceover: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating voiceover: {str(e)}")
            return None
    
    def edit_video(
        self,
        video_path: str,
        script: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Edit video with captions and voiceover."""
        try:
            # Load video
            video = VideoFileClip(video_path)
            
            # Generate voiceover
            voiceover_path = self.generate_voiceover(script)
            if not voiceover_path:
                return None
            
            # Create text clips for captions
            text_clips = []
            for i, line in enumerate(script.split('\n')):
                text_clip = TextClip(
                    line,
                    fontsize=30,
                    color='white',
                    stroke_color='black',
                    stroke_width=2
                )
                text_clip = text_clip.set_position(('center', 0.8))
                text_clip = text_clip.set_duration(video.duration / len(script.split('\n')))
                text_clip = text_clip.set_start(i * (video.duration / len(script.split('\n'))))
                text_clips.append(text_clip)
            
            # Combine video, voiceover, and captions
            final_video = CompositeVideoClip([video] + text_clips)
            final_video = final_video.set_audio(voiceover_path)
            
            # Save video
            if output_path is None:
                output_path = str(self.output_dir / f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            
            final_video.write_videofile(output_path)
            
            self.logger.info(f"Edited video: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error editing video: {str(e)}")
            return None
    
    def analyze_instagram(
        self,
        username: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze Instagram profile performance."""
        try:
            # Initialize Instagram API
            api = Client(
                self.config["instagram"]["username"],
                self.config["instagram"]["password"]
            )
            
            # Get user info
            user_info = api.username_info(username)
            
            # Get recent posts
            user_feed = api.user_feed(user_info['user']['pk'])
            
            # Analyze posts
            posts = []
            for post in user_feed.get('items', [])[:days]:
                posts.append({
                    'date': datetime.fromtimestamp(post['taken_at']),
                    'likes': post.get('like_count', 0),
                    'comments': post.get('comment_count', 0),
                    'type': 'carousel' if 'carousel_media' in post else 'single'
                })
            
            # Calculate metrics
            df = pd.DataFrame(posts)
            metrics = {
                'total_posts': len(posts),
                'avg_likes': df['likes'].mean(),
                'avg_comments': df['comments'].mean(),
                'engagement_rate': (df['likes'] + df['comments']).mean() / user_info['user']['follower_count'],
                'carousel_rate': (df['type'] == 'carousel').mean()
            }
            
            # Create engagement chart
            fig = px.line(
                df,
                x='date',
                y=['likes', 'comments'],
                title='Engagement Over Time'
            )
            chart_path = str(self.output_dir / f"engagement_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            fig.write_html(chart_path)
            
            return {
                'metrics': metrics,
                'chart_path': chart_path
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing Instagram: {str(e)}")
            return None
    
    def schedule_post(
        self,
        schedule: PostSchedule
    ) -> bool:
        """Schedule a post for later."""
        try:
            # Save schedule
            schedule_file = self.output_dir / "schedules.jsonl"
            
            with open(schedule_file, 'a') as f:
                json.dump(schedule.__dict__, f)
                f.write('\n')
            
            self.logger.info(f"Scheduled post for {schedule.date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error scheduling post: {str(e)}")
            return False 