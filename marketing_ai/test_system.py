from advanced_ai_system import MarketingAI
import json

def test_marketing_ai():
    """Test the Marketing AI system with various scenarios."""
    # Initialize the system with test tokens
    marketing_ai = MarketingAI(
        twitter_bearer_token="test_twitter_token",
        google_api_key="test_google_key"
    )
    
    print("\n=== Marketing AI System Test Suite ===")
    
    # Test 1: Content Analysis
    print("\n1. Testing Content Analysis...")
    test_content = {
        "text": "Check out our amazing new product! Limited time offer - 50% off!",
        "type": "social_post",
        "user_id": "test_user_1",
        "platform": "twitter"
    }
    analysis_result = marketing_ai.analyze_content(test_content)
    print("\nContent Analysis Results:")
    print(json.dumps(analysis_result, indent=2))
    
    # Test 2: Interactive Tutorial
    print("\n2. Testing Interactive Tutorial...")
    tutorial_result = marketing_ai._handle_tutorial_request("tutorial social media")
    print("\nTutorial Response:")
    print(json.dumps(tutorial_result, indent=2))
    
    # Test 3: Learning Mode
    print("\n3. Testing Learning Mode...")
    learning_result = marketing_ai._handle_learning_request("learn marketing")
    print("\nLearning Mode Response:")
    print(json.dumps(learning_result, indent=2))
    
    # Test 4: Question Handling
    print("\n4. Testing Question Handling...")
    question_result = marketing_ai._handle_question("What's the best time to post on social media?")
    print("\nQuestion Response:")
    print(json.dumps(question_result, indent=2))
    
    # Test 5: Analysis Request
    print("\n5. Testing Analysis Request...")
    analysis_request = marketing_ai._handle_analysis_request("analyze my social media post")
    print("\nAnalysis Request Response:")
    print(json.dumps(analysis_request, indent=2))
    
    # Test 6: Optimization Request
    print("\n6. Testing Optimization Request...")
    optimization_result = marketing_ai._handle_optimization_request("optimize my campaign")
    print("\nOptimization Response:")
    print(json.dumps(optimization_result, indent=2))
    
    # Test 7: Prediction Request
    print("\n7. Testing Prediction Request...")
    prediction_result = marketing_ai._handle_prediction_request("predict campaign performance")
    print("\nPrediction Response:")
    print(json.dumps(prediction_result, indent=2))
    
    # Test 8: Recommendation Request
    print("\n8. Testing Recommendation Request...")
    recommendation_result = marketing_ai._handle_recommendation_request("recommend marketing strategies")
    print("\nRecommendation Response:")
    print(json.dumps(recommendation_result, indent=2))
    
    # Test 9: Creative Request
    print("\n9. Testing Creative Request...")
    creative_result = marketing_ai._handle_creative_request("create a viral campaign")
    print("\nCreative Response:")
    print(json.dumps(creative_result, indent=2))
    
    # Test 10: Interactive Session
    print("\n10. Testing Interactive Session...")
    print("\nStarting interactive session (type 'exit' to end):")
    marketing_ai.start_interactive_session("test_user_1")

if __name__ == "__main__":
    test_marketing_ai() 