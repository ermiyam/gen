from typing import Dict, Any, Optional
from datetime import datetime
import random

class FlexibleResponder:
    def __init__(self):
        """Initialize the flexible response generator with marketing-specific styles."""
        self.current_date = datetime.now().strftime("%B %d, %Y")
        self.learning_history = []
        
        # Enhanced marketing-specific intros
        self.intros = {
            "analysis": [
                "Time to dive into the data like a marketing archaeologist—let's find those golden insights.",
                "Ready to dissect this marketing puzzle with some AI-powered precision.",
                "Let's break this down with the analytical power of a marketing microscope.",
                "Time to put this marketing data under the microscope and extract some insights.",
                "Let's analyze this marketing challenge with surgical precision."
            ],
            "optimization": [
                "Time to turbocharge your marketing engine with some AI magic.",
                "Let's optimize this campaign like we're tuning a high-performance engine.",
                "Ready to push your marketing metrics into overdrive.",
                "Let's fine-tune your marketing strategy for maximum impact.",
                "Time to optimize your marketing machine for peak performance."
            ],
            "prediction": [
                "Let's peer into the marketing crystal ball with some AI-powered foresight.",
                "Time to forecast your marketing future with data-driven precision.",
                "Ready to predict your next marketing win with some AI wizardry.",
                "Let's gaze into the marketing future with our AI-powered crystal ball.",
                "Time to predict your marketing success with data-backed confidence."
            ],
            "recommendation": [
                "Let me serve up some marketing wisdom with a side of AI insight.",
                "Time to dish out some strategic marketing recommendations.",
                "Ready to drop some knowledge bombs on your marketing strategy.",
                "Let's craft some marketing recommendations that hit the bullseye.",
                "Time to share some marketing wisdom backed by AI analysis."
            ],
            "question": [
                "Got a marketing question? Let's crack it open with some AI expertise.",
                "Time to answer your marketing query with some data-backed wisdom.",
                "Ready to tackle your marketing question with precision and clarity.",
                "Let's dive into your marketing question with AI-powered insight.",
                "Time to answer your marketing question with data-driven precision."
            ],
            "learning": [
                "Let's learn from this marketing challenge together.",
                "Time to add some marketing wisdom to our knowledge base.",
                "Ready to learn and improve our marketing strategies.",
                "Let's learn from this marketing experience and grow stronger.",
                "Time to expand our marketing knowledge together."
            ],
            "creative": [
                "Let's unleash some creative marketing magic!",
                "Time to get creative with your marketing strategy.",
                "Ready to craft some marketing brilliance.",
                "Let's create some marketing magic together.",
                "Time to let our creative marketing juices flow."
            ],
            "general": [
                "Let's tackle this marketing challenge with some AI-powered ingenuity.",
                "Time to apply some marketing magic with a dash of AI intelligence.",
                "Ready to help you navigate the marketing landscape with AI precision.",
                "Let's work on this marketing challenge together.",
                "Time to bring some AI-powered marketing expertise to the table."
            ]
        }
        
        # Enhanced marketing-specific outros
        self.outros = {
            "analysis": [
                "That's the data-driven truth—now go make those marketing moves!",
                "Insights served hot—time to turn them into marketing gold.",
                "Analysis complete—your marketing strategy just got an AI upgrade.",
                "Data analyzed—time to put these insights into action!",
                "Analysis served—go make those marketing waves!"
            ],
            "optimization": [
                "Your marketing engine is now optimized and ready to roar!",
                "Optimization complete—time to watch those metrics soar.",
                "Your marketing strategy just got a turbo boost—enjoy the ride!",
                "Optimization done—your marketing machine is running at peak performance!",
                "Fine-tuning complete—your marketing strategy is ready to shine!"
            ],
            "prediction": [
                "The future of your marketing is looking bright—let's make it happen!",
                "Prediction served—time to turn foresight into marketing action.",
                "Your marketing crystal ball is clear—go make those predictions reality!",
                "Future forecasted—time to make those marketing predictions come true!",
                "Predictions ready—let's turn them into marketing success!"
            ],
            "recommendation": [
                "Marketing wisdom served—time to put it into action!",
                "Recommendations ready—go make those marketing waves!",
                "Strategic insights delivered—your marketing game just got stronger.",
                "Recommendations served—time to implement these marketing strategies!",
                "Marketing advice delivered—go make those marketing moves!"
            ],
            "question": [
                "Question answered—now go make those marketing moves!",
                "Knowledge served—time to put it to work in your marketing!",
                "Answer delivered—your marketing strategy just got smarter.",
                "Question answered—time to apply this marketing knowledge!",
                "Answer served—go make those marketing decisions!"
            ],
            "learning": [
                "Learning complete—your marketing knowledge just grew!",
                "Knowledge gained—time to apply these marketing lessons!",
                "Learning served—your marketing expertise just leveled up!",
                "Knowledge acquired—go make those marketing improvements!",
                "Learning complete—your marketing skills just got stronger!"
            ],
            "creative": [
                "Creative magic unleashed—go make those marketing waves!",
                "Creativity served—time to implement these marketing ideas!",
                "Creative brilliance delivered—your marketing just got more innovative!",
                "Creative ideas ready—go make those marketing dreams reality!",
                "Creativity unleashed—your marketing just got more exciting!"
            ],
            "general": [
                "Marketing insights served—time to make them work for you!",
                "AI wisdom delivered—go make those marketing waves!",
                "Strategy served—your marketing game just got an upgrade.",
                "Marketing help delivered—time to put it into action!",
                "AI assistance complete—go make those marketing moves!"
            ]
        }

    def detect_intent(self, text: str) -> str:
        """Enhanced intent detection with more specific marketing intents."""
        text = text.lower()
        
        # Check for specific marketing intents with more detailed patterns
        if any(word in text for word in ["analyze", "evaluate", "assess", "review", "examine", "look at", "check"]):
            return "analysis"
        elif any(word in text for word in ["optimize", "improve", "enhance", "boost", "maximize", "better", "upgrade"]):
            return "optimization"
        elif any(word in text for word in ["predict", "forecast", "project", "anticipate", "estimate", "future", "will"]):
            return "prediction"
        elif any(word in text for word in ["recommend", "suggest", "propose", "advise", "guide", "help", "should"]):
            return "recommendation"
        elif any(word in text for word in ["learn", "teach", "understand", "explain", "how", "why", "what"]):
            return "learning"
        elif any(word in text for word in ["create", "design", "make", "build", "develop", "generate", "write"]):
            return "creative"
        elif "?" in text:
            return "question"
        else:
            return "general"

    def grok_intro(self, intent: str) -> str:
        """Generate a Grok-style opening with marketing focus."""
        return random.choice(self.intros.get(intent, self.intros["general"]))

    def chatgpt_body(self, content: Dict[str, Any], intent: str) -> str:
        """Generate a ChatGPT-style structured response body."""
        body = "Here's what I've got for you:\n\n"
        
        # Format based on intent and content
        if intent == "analysis":
            if "metrics" in content:
                body += "📊 **Key Metrics**:\n"
                for metric, value in content["metrics"].items():
                    body += f"- {metric}: {value}\n"
                body += "\n"
            
            if "insights" in content:
                body += "💡 **Key Insights**:\n"
                for insight in content["insights"]:
                    body += f"- {insight}\n"
                body += "\n"
            
            if "recommendations" in content:
                body += "🎯 **Recommendations**:\n"
                for rec in content["recommendations"]:
                    body += f"- {rec}\n"
                    
        elif intent == "optimization":
            if "improvements" in content:
                body += "📈 **Improvements Made**:\n"
                for imp in content["improvements"]:
                    body += f"- {imp}\n"
                body += "\n"
            
            if "metrics" in content:
                body += "📊 **Performance Metrics**:\n"
                for metric, value in content["metrics"].items():
                    body += f"- {metric}: {value}\n"
                body += "\n"
            
            if "next_steps" in content:
                body += "🎯 **Next Steps**:\n"
                for step in content["next_steps"]:
                    body += f"- {step}\n"
                    
        elif intent == "prediction":
            if "predictions" in content:
                body += "🔮 **Predictions**:\n"
                for pred in content["predictions"]:
                    body += f"- {pred}\n"
                body += "\n"
            
            if "confidence" in content:
                body += "📊 **Confidence Levels**:\n"
                for item, conf in content["confidence"].items():
                    body += f"- {item}: {conf}%\n"
                body += "\n"
            
            if "factors" in content:
                body += "🎯 **Key Factors**:\n"
                for factor in content["factors"]:
                    body += f"- {factor}\n"
                    
        elif intent == "recommendation":
            if "strategies" in content:
                body += "🎯 **Strategic Recommendations**:\n"
                for strategy in content["strategies"]:
                    body += f"- {strategy}\n"
                body += "\n"
            
            if "benefits" in content:
                body += "✨ **Expected Benefits**:\n"
                for benefit in content["benefits"]:
                    body += f"- {benefit}\n"
                body += "\n"
            
            if "implementation" in content:
                body += "⚙️ **Implementation Steps**:\n"
                for step in content["implementation"]:
                    body += f"- {step}\n"
                    
        elif intent == "question":
            if "answer" in content:
                body += f"{content['answer']}\n\n"
            
            if "details" in content:
                body += "📚 **Additional Details**:\n"
                for detail in content["details"]:
                    body += f"- {detail}\n"
                body += "\n"
            
            if "related" in content:
                body += "🔗 **Related Topics**:\n"
                for topic in content["related"]:
                    body += f"- {topic}\n"
                    
        else:  # general
            if "message" in content:
                body += f"{content['message']}\n\n"
            
            if "suggestions" in content:
                body += "💡 **Suggestions**:\n"
                for suggestion in content["suggestions"]:
                    body += f"- {suggestion}\n"
                body += "\n"
            
            if "next_steps" in content:
                body += "🎯 **Next Steps**:\n"
                for step in content["next_steps"]:
                    body += f"- {step}\n"
                    
        return body

    def grok_outro(self, intent: str) -> str:
        """Generate a Grok-style closing with marketing focus."""
        return random.choice(self.outros.get(intent, self.outros["general"]))

    def learn_from_interaction(self, user_input: str, content: Dict[str, Any], response: str):
        """Learn from interactions to improve future responses."""
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "content": content,
            "response": response,
            "intent": self.detect_intent(user_input)
        })
        
        # Keep only last 1000 interactions
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]

    def generate_response(self, text: str, content: Dict[str, Any]) -> str:
        """Generate a complete hybrid response with learning capability."""
        # Detect intent
        intent = self.detect_intent(text)
        
        # Generate response components
        intro = self.grok_intro(intent)
        body = self.chatgpt_body(content, intent)
        outro = self.grok_outro(intent)
        
        # Combine components
        response = f"{intro}\n\n{body}\n\n{outro}"
        
        # Add timestamp
        response = f"AI Marketing Assistant ({self.current_date}):\n\n{response}"
        
        # Learn from this interaction
        self.learn_from_interaction(text, content, response)
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the response generator
    responder = FlexibleResponder()
    
    # Test different types of responses
    test_cases = [
        {
            "text": "Analyze my marketing campaign performance",
            "content": {
                "metrics": {
                    "reach": "10,000",
                    "engagement": "5%",
                    "conversions": "100"
                },
                "insights": [
                    "Strong engagement on social media",
                    "Email campaign needs optimization",
                    "Video content performing well"
                ],
                "recommendations": [
                    "Increase social media budget",
                    "Revise email subject lines",
                    "Create more video content"
                ]
            }
        },
        {
            "text": "What's the best time to post on social media?",
            "content": {
                "answer": "Based on your audience data, the optimal posting times are 9 AM and 6 PM EST.",
                "details": [
                    "Highest engagement during morning commute",
                    "Evening posts perform well with working professionals"
                ],
                "related": [
                    "Content scheduling",
                    "Audience engagement patterns",
                    "Social media analytics"
                ]
            }
        },
        {
            "text": "Help me create a viral marketing campaign",
            "content": {
                "strategies": [
                    "Leverage trending topics",
                    "Create shareable content",
                    "Engage with influencers"
                ],
                "benefits": [
                    "Increased brand visibility",
                    "Higher engagement rates",
                    "Greater reach potential"
                ],
                "implementation": [
                    "Research current trends",
                    "Design shareable assets",
                    "Plan influencer outreach"
                ]
            }
        }
    ]
    
    # Generate and print responses
    for test in test_cases:
        response = responder.generate_response(test["text"], test["content"])
        print("\n" + "="*50)
        print(response)
        print("="*50 + "\n") 