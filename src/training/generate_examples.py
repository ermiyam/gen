import json
import random
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ExampleType(Enum):
    VIDEO_CONTENT = "video_content"
    SOCIAL_MEDIA = "social_media"
    MARKETING_STRATEGY = "marketing_strategy"
    CONTENT_CREATION = "content_creation"
    INFLUENCER_MARKETING = "influencer_marketing"
    EMAIL_CAMPAIGN = "email_campaign"
    PAID_ADS = "paid_ads"
    SEO_STRATEGY = "seo_strategy"

@dataclass
class Metrics:
    engagement_rate: float
    conversion_rate: float
    roi: float
    cpa: float
    ltv: float
    reach: int
    impressions: int
    clicks: int
    ctr: float
    bounce_rate: float
    time_on_page: float
    social_shares: int
    comments: int
    likes: int
    followers_gained: int
    email_open_rate: float
    email_click_rate: float
    email_unsubscribe_rate: float
    influencer_engagement: float
    brand_sentiment: float
    content_quality_score: float

class MarketingExampleGenerator:
    def __init__(self):
        self.platforms = [
            "TikTok", "Instagram", "Facebook", "LinkedIn", "Twitter",
            "YouTube", "Pinterest", "Snapchat", "Reddit", "TikTok"
        ]
        
        self.content_types = [
            "video", "image", "carousel", "story", "reel",
            "post", "article", "thread", "pin", "short"
        ]
        
        self.marketing_goals = [
            "brand awareness", "lead generation", "sales conversion",
            "customer engagement", "community building", "product launch",
            "brand loyalty", "market research", "crisis management",
            "thought leadership"
        ]
        
        self.industries = [
            "fashion", "technology", "health", "food", "travel",
            "education", "finance", "entertainment", "sports", "beauty"
        ]
        
        self.target_audiences = [
            "Gen Z", "Millennials", "Gen X", "Boomers", "professionals",
            "students", "parents", "entrepreneurs", "creators", "gaming"
        ]
        
        # New specialized categories
        self.influencer_types = [
            "nano", "micro", "macro", "mega", "celebrity",
            "industry expert", "thought leader", "brand ambassador",
            "content creator", "social media personality"
        ]
        
        self.email_types = [
            "newsletter", "promotional", "welcome series", "abandoned cart",
            "product launch", "educational", "re-engagement", "survey",
            "announcement", "personalized recommendation"
        ]
        
        self.ad_platforms = [
            "Google Ads", "Facebook Ads", "Instagram Ads", "LinkedIn Ads",
            "TikTok Ads", "Twitter Ads", "Pinterest Ads", "YouTube Ads",
            "Reddit Ads", "Programmatic Display"
        ]
        
        self.seo_elements = [
            "keyword research", "on-page optimization", "technical SEO",
            "content strategy", "link building", "local SEO", "mobile SEO",
            "voice search", "image optimization", "schema markup"
        ]
    
    def generate_metrics(self) -> Metrics:
        """Generate realistic marketing metrics."""
        return Metrics(
            engagement_rate=random.uniform(0.01, 0.15),
            conversion_rate=random.uniform(0.01, 0.10),
            roi=random.uniform(1.5, 5.0),
            cpa=random.uniform(10, 100),
            ltv=random.uniform(50, 500),
            reach=random.randint(1000, 100000),
            impressions=random.randint(5000, 500000),
            clicks=random.randint(100, 10000),
            ctr=random.uniform(0.01, 0.10),
            bounce_rate=random.uniform(0.2, 0.7),
            time_on_page=random.uniform(30, 300),
            social_shares=random.randint(10, 1000),
            comments=random.randint(5, 500),
            likes=random.randint(50, 5000),
            followers_gained=random.randint(10, 1000),
            email_open_rate=random.uniform(0.15, 0.45),
            email_click_rate=random.uniform(0.02, 0.15),
            email_unsubscribe_rate=random.uniform(0.001, 0.05),
            influencer_engagement=random.uniform(0.02, 0.20),
            brand_sentiment=random.uniform(0.6, 0.95),
            content_quality_score=random.uniform(0.7, 0.98)
        )
    
    def generate_video_content_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(10):
            platform = random.choice(self.platforms)
            goal = random.choice(self.marketing_goals)
            industry = random.choice(self.industries)
            audience = random.choice(self.target_audiences)
            
            examples.append({
                "instruction": f"Create a {platform} video strategy for {industry} industry targeting {audience} to achieve {goal}",
                "input": f"Platform: {platform}\nIndustry: {industry}\nTarget Audience: {audience}\nGoal: {goal}",
                "output": f"""Here's a comprehensive {platform} video strategy for {industry} targeting {audience}:

1. Content Strategy:
   - Create {random.choice(['educational', 'entertaining', 'inspirational'])} videos
   - Focus on {random.choice(['trending topics', 'industry insights', 'behind-the-scenes'])}
   - Length: {random.choice(['15-30 seconds', '30-60 seconds', '60+ seconds'])}
   - Posting frequency: {random.choice(['daily', '3x/week', 'weekly'])}

2. Engagement Tactics:
   - Use {random.choice(['trending sounds', 'original music', 'popular hashtags'])}
   - Implement {random.choice(['duets', 'stitches', 'reactions'])} for interaction
   - Encourage {random.choice(['comments', 'shares', 'follows'])} through calls-to-action

3. Optimization:
   - Track {random.choice(['engagement rate', 'watch time', 'conversion rate'])}
   - A/B test {random.choice(['thumbnails', 'captions', 'posting times'])}
   - Analyze {random.choice(['audience retention', 'demographics', 'peak hours'])}""",
                "metadata": {
                    "type": "video_content",
                    "platform": platform,
                    "industry": industry,
                    "goal": goal,
                    "audience": audience
                }
            })
        return examples
    
    def generate_social_media_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(10):
            platform = random.choice(self.platforms)
            content_type = random.choice(self.content_types)
            goal = random.choice(self.marketing_goals)
            
            examples.append({
                "instruction": f"Design a {platform} {content_type} campaign to achieve {goal}",
                "input": f"Platform: {platform}\nContent Type: {content_type}\nGoal: {goal}",
                "output": f"""Here's a {platform} {content_type} campaign strategy:

1. Campaign Structure:
   - Theme: {random.choice(['seasonal', 'product-focused', 'brand story'])}
   - Duration: {random.choice(['1 week', '2 weeks', '1 month'])}
   - Posting schedule: {random.choice(['daily', 'alternate days', 'weekly'])}
   - Content mix: {random.choice(['70% value, 30% promotional', '60% educational, 40% entertaining'])}

2. Content Elements:
   - Visual style: {random.choice(['minimalist', 'bold', 'professional'])}
   - Color scheme: {random.choice(['brand colors', 'trending palette', 'seasonal'])}
   - Typography: {random.choice(['modern', 'classic', 'playful'])}
   - Hashtag strategy: {random.choice(['branded + trending', 'niche-specific', 'campaign-specific'])}

3. Engagement Plan:
   - Interactive elements: {random.choice(['polls', 'quizzes', 'contests'])}
   - User-generated content: {random.choice(['challenges', 'testimonials', 'reviews'])}
   - Influencer collaboration: {random.choice(['micro-influencers', 'brand ambassadors', 'industry experts'])}""",
                "metadata": {
                    "type": "social_media",
                    "platform": platform,
                    "content_type": content_type,
                    "goal": goal
                }
            })
        return examples
    
    def generate_marketing_strategy_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(10):
            industry = random.choice(self.industries)
            goal = random.choice(self.marketing_goals)
            audience = random.choice(self.target_audiences)
            
            examples.append({
                "instruction": f"Develop a comprehensive marketing strategy for {industry} industry targeting {audience} to achieve {goal}",
                "input": f"Industry: {industry}\nTarget Audience: {audience}\nGoal: {goal}",
                "output": f"""Here's a comprehensive marketing strategy for {industry}:

1. Market Analysis:
   - Target audience: {audience}
   - Key demographics: {random.choice(['age 18-34', 'urban professionals', 'digital natives'])}
   - Pain points: {random.choice(['time management', 'cost efficiency', 'quality assurance'])}
   - Market trends: {random.choice(['sustainability', 'personalization', 'digital transformation'])}

2. Channel Strategy:
   - Primary channels: {random.choice(['social media + email', 'content marketing + SEO', 'paid ads + PR'])}
   - Content mix: {random.choice(['40% educational, 30% entertaining, 30% promotional'])}
   - Budget allocation: {random.choice(['60% digital, 40% traditional', '80% performance, 20% brand'])}
   - KPI framework: {random.choice(['awareness, consideration, conversion', 'reach, engagement, revenue'])}

3. Implementation Plan:
   - Timeline: {random.choice(['3 months', '6 months', '1 year'])}
   - Resource allocation: {random.choice(['in-house team + agencies', 'freelancers + automation'])}
   - Risk mitigation: {random.choice(['crisis management plan', 'competitor monitoring', 'compliance checks'])}
   - Success metrics: {random.choice(['ROI, CAC, LTV', 'brand awareness, engagement rate'])}""",
                "metadata": {
                    "type": "marketing_strategy",
                    "industry": industry,
                    "goal": goal,
                    "audience": audience
                }
            })
        return examples
    
    def generate_content_creation_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(10):
            content_type = random.choice(self.content_types)
            platform = random.choice(self.platforms)
            industry = random.choice(self.industries)
            
            examples.append({
                "instruction": f"Create a content creation guide for {platform} {content_type} in {industry} industry",
                "input": f"Platform: {platform}\nContent Type: {content_type}\nIndustry: {industry}",
                "output": f"""Here's a comprehensive guide for creating {platform} {content_type} content:

1. Content Planning:
   - Research phase: {random.choice(['competitor analysis', 'trend research', 'audience insights'])}
   - Content calendar: {random.choice(['monthly themes', 'weekly topics', 'daily content'])}
   - Resource gathering: {random.choice(['stock assets', 'brand guidelines', 'templates'])}
   - Quality checklist: {random.choice(['brand voice', 'visual consistency', 'grammar check'])}

2. Creation Process:
   - Tools needed: {random.choice(['editing software', 'design tools', 'analytics platform'])}
   - Time allocation: {random.choice(['2 hours per piece', '1 day per week', 'batch creation'])}
   - Team roles: {random.choice(['content writer, designer, strategist', 'creator, editor, manager'])}
   - Review process: {random.choice(['internal review + client approval', 'peer review + stakeholder sign-off'])}

3. Optimization Tips:
   - SEO elements: {random.choice(['keywords, meta descriptions', 'alt text, captions'])}
   - Engagement hooks: {random.choice(['question prompts', 'call-to-action', 'value proposition'])}
   - Performance tracking: {random.choice(['analytics dashboard', 'engagement metrics', 'conversion tracking'])}
   - Iteration process: {random.choice(['A/B testing', 'audience feedback', 'performance analysis'])}""",
                "metadata": {
                    "type": "content_creation",
                    "platform": platform,
                    "content_type": content_type,
                    "industry": industry
                }
            })
        return examples
    
    def generate_influencer_marketing_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(10):
            influencer_type = random.choice(self.influencer_types)
            industry = random.choice(self.industries)
            platform = random.choice(self.platforms)
            metrics = self.generate_metrics()
            
            examples.append({
                "instruction": f"Design an influencer marketing campaign using {influencer_type} influencers for {industry} industry on {platform}",
                "input": f"Influencer Type: {influencer_type}\nIndustry: {industry}\nPlatform: {platform}",
                "output": f"""Here's a comprehensive influencer marketing strategy:

1. Influencer Selection:
   - Type: {influencer_type} influencers
   - Platform focus: {platform}
   - Industry expertise: {industry}
   - Follower range: {random.choice(['1K-10K', '10K-50K', '50K-100K', '100K+'])}
   - Engagement rate target: {metrics.influencer_engagement:.2%}

2. Campaign Structure:
   - Duration: {random.choice(['1 month', '3 months', '6 months'])}
   - Content mix: {random.choice(['60% sponsored, 40% organic', '70% educational, 30% promotional'])}
   - Posting frequency: {random.choice(['weekly', 'bi-weekly', 'monthly'])}
   - Collaboration type: {random.choice(['one-time post', 'series', 'ambassador program'])}

3. Performance Metrics:
   - Expected reach: {metrics.reach:,}
   - Target engagement rate: {metrics.engagement_rate:.2%}
   - Conversion rate goal: {metrics.conversion_rate:.2%}
   - ROI target: {metrics.roi:.1f}x
   - Brand sentiment goal: {metrics.brand_sentiment:.2%}

4. Content Guidelines:
   - Brand voice: {random.choice(['professional', 'casual', 'educational'])}
   - Key messaging: {random.choice(['product benefits', 'brand values', 'industry expertise'])}
   - Visual style: {random.choice(['lifestyle', 'product-focused', 'educational'])}
   - Hashtag strategy: {random.choice(['branded + campaign', 'industry + trending'])}""",
                "metadata": {
                    "type": ExampleType.INFLUENCER_MARKETING.value,
                    "influencer_type": influencer_type,
                    "industry": industry,
                    "platform": platform,
                    "metrics": metrics.__dict__
                }
            })
        return examples
    
    def generate_email_campaign_examples(self) -> List[Dict[str, Any]]:
        examples = []
        for _ in range(10):
            email_type = random.choice(self.email_types)
            industry = random.choice(self.industries)
            goal = random.choice(self.marketing_goals)
            metrics = self.generate_metrics()
            
            examples.append({
                "instruction": f"Create an {email_type} campaign for {industry} industry to achieve {goal}",
                "input": f"Email Type: {email_type}\nIndustry: {industry}\nGoal: {goal}",
                "output": f"""Here's a comprehensive {email_type} campaign strategy:

1. Campaign Overview:
   - Type: {email_type}
   - Industry: {industry}
   - Primary goal: {goal}
   - Target audience: {random.choice(self.target_audiences)}
   - Campaign duration: {random.choice(['2 weeks', '1 month', '3 months'])}
   - Email frequency: {random.choice(['weekly', 'bi-weekly', 'daily'])}

2. Content Strategy:
   - Subject line approach: {random.choice(['benefit-focused', 'curiosity-driven', 'urgency-based'])}
   - Email structure: {random.choice(['scannable sections', 'story-based', 'visual-heavy'])}
   - Call-to-action: {random.choice(['primary + secondary', 'single focused', 'multiple options'])}
   - Personalization level: {random.choice(['basic', 'advanced', 'dynamic'])}

3. Performance Metrics:
   - Open rate target: {metrics.email_open_rate:.2%}
   - Click-through goal: {metrics.email_click_rate:.2%}
   - Unsubscribe threshold: <{metrics.email_unsubscribe_rate:.2%}
   - Conversion target: {metrics.conversion_rate:.2%}
   - Revenue goal: ${metrics.ltv:.2f} per subscriber

4. Technical Setup:
   - Email service provider: {random.choice(['Mailchimp', 'SendGrid', 'HubSpot'])}
   - Automation workflow: {random.choice(['welcome series', 'abandoned cart', 're-engagement'])}
   - A/B testing: {random.choice(['subject lines', 'content layout', 'CTA placement'])}
   - List segmentation: {random.choice(['demographics', 'behavior', 'engagement level'])}""",
                "metadata": {
                    "type": ExampleType.EMAIL_CAMPAIGN.value,
                    "email_type": email_type,
                    "industry": industry,
                    "goal": goal,
                    "metrics": metrics.__dict__
                }
            })
        return examples

def main():
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = MarketingExampleGenerator()
    
    # Generate examples
    all_examples = []
    all_examples.extend(generator.generate_video_content_examples())
    all_examples.extend(generator.generate_social_media_examples())
    all_examples.extend(generator.generate_marketing_strategy_examples())
    all_examples.extend(generator.generate_content_creation_examples())
    all_examples.extend(generator.generate_influencer_marketing_examples())
    all_examples.extend(generator.generate_email_campaign_examples())
    
    # Save examples
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"specialized_examples_{timestamp}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(all_examples)} specialized marketing examples")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main() 