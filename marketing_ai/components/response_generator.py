from typing import Dict, Any, Optional
from datetime import datetime
import re

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator with marketing-specific styles."""
        self.current_date = datetime.now().strftime("%B %d, %Y")
        
        # Marketing-specific intents
        self.intents = {
            "analysis": ["analyze", "evaluate", "assess", "review", "examine"],
            "optimization": ["optimize", "improve", "enhance", "boost", "maximize"],
            "prediction": ["predict", "forecast", "project", "anticipate", "estimate"],
            "recommendation": ["recommend", "suggest", "propose", "advise", "guide"],
            "question": ["what", "how", "why", "when", "where", "which", "?"]
        }

    def detect_intent(self, text: str) -> str:
        """Detect the intent of the user's input."""
        text = text.lower()
        
        for intent, keywords in self.intents.items():
            if any(keyword in text for keyword in keywords):
                return intent
                
        return "general"

    def grok_intro(self, intent: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a Grok-style opening with marketing focus."""
        intros = {
            "analysis": [
                "Time to dive into the data like a marketing archaeologist—let's find those golden insights.",
                "Ready to dissect this marketing puzzle with some AI-powered precision.",
                "Let's break this down with the analytical power of a marketing microscope."
            ],
            "optimization": [
                "Time to turbocharge your marketing engine with some AI magic.",
                "Let's optimize this campaign like we're tuning a high-performance engine.",
                "Ready to push your marketing metrics into overdrive."
            ],
            "prediction": [
                "Let's peer into the marketing crystal ball with some AI-powered foresight.",
                "Time to forecast your marketing future with data-driven precision.",
                "Ready to predict your next marketing win with some AI wizardry."
            ],
            "recommendation": [
                "Let me serve up some marketing wisdom with a side of AI insight.",
                "Time to dish out some strategic marketing recommendations.",
                "Ready to drop some knowledge bombs on your marketing strategy."
            ],
            "question": [
                "Got a marketing question? Let's crack it open with some AI expertise.",
                "Time to answer your marketing query with some data-backed wisdom.",
                "Ready to tackle your marketing question with precision and clarity."
            ],
            "general": [
                "Let's tackle this marketing challenge with some AI-powered ingenuity.",
                "Time to apply some marketing magic with a dash of AI intelligence.",
                "Ready to help you navigate the marketing landscape with AI precision."
            ]
        }
        
        # Select a random intro for the detected intent
        import random
        return random.choice(intros.get(intent, intros["general"]))

    def chatgpt_body(self, intent: str, content: Dict[str, Any]) -> str:
        """Generate a ChatGPT-style structured response body."""
        if intent == "analysis":
            return self._format_analysis_body(content)
        elif intent == "optimization":
            return self._format_optimization_body(content)
        elif intent == "prediction":
            return self._format_prediction_body(content)
        elif intent == "recommendation":
            return self._format_recommendation_body(content)
        elif intent == "question":
            return self._format_question_body(content)
        else:
            return self._format_general_body(content)

    def grok_outro(self, intent: str) -> str:
        """Generate a Grok-style closing with marketing focus."""
        outros = {
            "analysis": [
                "That's the data-driven truth—now go make those marketing moves!",
                "Insights served hot—time to turn them into marketing gold.",
                "Analysis complete—your marketing strategy just got an AI upgrade."
            ],
            "optimization": [
                "Your marketing engine is now optimized and ready to roar!",
                "Optimization complete—time to watch those metrics soar.",
                "Your marketing strategy just got a turbo boost—enjoy the ride!"
            ],
            "prediction": [
                "The future of your marketing is looking bright—let's make it happen!",
                "Prediction served—time to turn foresight into marketing action.",
                "Your marketing crystal ball is clear—go make those predictions reality!"
            ],
            "recommendation": [
                "Marketing wisdom served—time to put it into action!",
                "Recommendations ready—go make those marketing waves!",
                "Strategic insights delivered—your marketing game just got stronger."
            ],
            "question": [
                "Question answered—now go make those marketing moves!",
                "Knowledge served—time to put it to work in your marketing!",
                "Answer delivered—your marketing strategy just got smarter."
            ],
            "general": [
                "Marketing insights served—time to make them work for you!",
                "AI wisdom delivered—go make those marketing waves!",
                "Strategy served—your marketing game just got an upgrade."
            ]
        }
        
        import random
        return random.choice(outros.get(intent, outros["general"]))

    def _format_analysis_body(self, content: Dict[str, Any]) -> str:
        """Format analysis results in a structured way."""
        body = "Here's what I found:\n\n"
        
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
                
        return body

    def _format_optimization_body(self, content: Dict[str, Any]) -> str:
        """Format optimization results in a structured way."""
        body = "Optimization complete! Here's what we've done:\n\n"
        
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
                
        return body

    def _format_prediction_body(self, content: Dict[str, Any]) -> str:
        """Format prediction results in a structured way."""
        body = "Here's what the future holds:\n\n"
        
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
                
        return body

    def _format_recommendation_body(self, content: Dict[str, Any]) -> str:
        """Format recommendations in a structured way."""
        body = "Here are your marketing recommendations:\n\n"
        
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
                
        return body

    def _format_question_body(self, content: Dict[str, Any]) -> str:
        """Format question answers in a structured way."""
        body = "Here's your answer:\n\n"
        
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
                
        return body

    def _format_general_body(self, content: Dict[str, Any]) -> str:
        """Format general responses in a structured way."""
        body = "Here's what I've got for you:\n\n"
        
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

    def generate_response(self, text: str, content: Dict[str, Any]) -> str:
        """Generate a complete hybrid response."""
        # Detect intent
        intent = self.detect_intent(text)
        
        # Generate response components
        intro = self.grok_intro(intent)
        body = self.chatgpt_body(intent, content)
        outro = self.grok_outro(intent)
        
        # Combine components
        response = f"{intro}\n\n{body}\n\n{outro}"
        
        # Add timestamp
        response = f"AI Marketing Assistant ({self.current_date}):\n\n{response}"
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize the response generator
    generator = ResponseGenerator()
    
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
        }
    ]
    
    # Generate and print responses
    for test in test_cases:
        response = generator.generate_response(test["text"], test["content"])
        print("\n" + "="*50)
        print(response)
        print("="*50 + "\n") 