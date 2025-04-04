import json
import random
from typing import List, Dict

def generate_marketing_examples(num_examples: int = 50) -> List[Dict]:
    """Generate additional marketing training examples."""
    
    # Template categories
    categories = {
        "social_media": [
            "Write a TikTok script for {product}",
            "Create an Instagram caption for {topic}",
            "Write a LinkedIn post about {topic}",
            "Create a Twitter thread about {topic}",
            "Write a Facebook ad for {product}"
        ],
        "email": [
            "Write a subject line for {product}",
            "Create an email body for {product}",
            "Write a newsletter introduction about {topic}",
            "Create an abandoned cart email for {product}"
        ],
        "content": [
            "Write a blog post title about {topic}",
            "Create a YouTube video description for {topic}",
            "Write a podcast episode description about {topic}",
            "Create a landing page headline for {product}"
        ],
        "product": [
            "Write product descriptions for {product}",
            "Create feature highlights for {product}",
            "Write benefits copy for {product}",
            "Create a product launch announcement for {product}"
        ]
    }
    
    # Sample topics and products
    topics = [
        "digital marketing", "social media strategy", "content creation",
        "brand building", "customer engagement", "lead generation",
        "email marketing", "SEO optimization", "paid advertising",
        "influencer marketing", "brand storytelling", "customer retention"
    ]
    
    products = [
        "AI marketing tool", "fitness app", "meal prep service",
        "eco-friendly clothing", "smart home device", "online course",
        "subscription box", "mobile app", "software platform",
        "wellness product", "sustainable product", "tech gadget"
    ]
    
    examples = []
    
    for _ in range(num_examples):
        # Select random category and template
        category = random.choice(list(categories.keys()))
        template = random.choice(categories[category])
        
        # Fill in template
        if "{product}" in template:
            instruction = template.format(product=random.choice(products))
        else:
            instruction = template.format(topic=random.choice(topics))
        
        # Generate appropriate output based on category
        if "TikTok" in instruction:
            output = f"🎥 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} your {random.choice(['game', 'life', 'business', 'success'])} with this game-changing tip!\n\n{random.choice(['Watch', 'Learn', 'Discover', 'See'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "Instagram" in instruction:
            output = f"✨ {random.choice(['Level up', 'Transform', 'Elevate', 'Boost'])} your {random.choice(['game', 'life', 'business', 'success'])} with this {random.choice(['game-changing', 'powerful', 'effective', 'proven'])} strategy!\n\n{random.choice(['Swipe', 'Save', 'Share', 'Tag'])}, {random.choice(['learn', 'discover', 'see', 'find out'])} {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} tip {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "LinkedIn" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} your {random.choice(['business', 'career', 'success', 'growth'])} with this {random.choice(['game-changing', 'powerful', 'effective', 'proven'])} strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} tip {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "Twitter" in instruction:
            output = f"🧵 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} your {random.choice(['game', 'life', 'business', 'success'])} with this {random.choice(['game-changing', 'powerful', 'effective', 'proven'])} strategy!\n\n1/ {random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} tip {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "Facebook" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} your {random.choice(['game', 'life', 'business', 'success'])} with this {random.choice(['game-changing', 'powerful', 'effective', 'proven'])} strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} tip {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "subject line" in instruction:
            output = f"🚀 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!"
        elif "email body" in instruction:
            output = f"Hi there,\n\nI wanted to share something {random.choice(['exciting', 'game-changing', 'powerful', 'effective'])} with you.\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\nBest regards,\n[Your Name]"
        elif "blog post" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!"
        elif "YouTube" in instruction:
            output = f"🎥 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\nIn this video, I'll show you {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "podcast" in instruction:
            output = f"🎙️ {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\nIn this episode, I'll share {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "landing page" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!"
        elif "product description" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "feature highlights" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "benefits copy" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        elif "product launch" in instruction:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        else:
            output = f"🎯 {random.choice(['Transform', 'Revolutionize', 'Elevate', 'Boost'])} Your {random.choice(['Game', 'Life', 'Business', 'Success'])} with This {random.choice(['Game-Changing', 'Powerful', 'Effective', 'Proven'])} Strategy!\n\n{random.choice(['Learn', 'Discover', 'See', 'Find out'])}, {random.choice(['how', 'why', 'what', 'when'])} this {random.choice(['simple', 'powerful', 'effective', 'proven'])} strategy {random.choice(['works', 'helps', 'transforms', 'improves'])}.\n\n#MarketingTips #{random.choice(['Business', 'Success', 'Growth', 'Strategy'])}"
        
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })
    
    return examples

def main():
    # Generate 50 new examples
    new_examples = generate_marketing_examples(50)
    
    # Load existing examples
    try:
        with open("data/training/nexgencreators_training_v1.jsonl", "r", encoding="utf-8") as f:
            existing_examples = [json.loads(line) for line in f]
    except FileNotFoundError:
        existing_examples = []
    
    # Combine existing and new examples
    all_examples = existing_examples + new_examples
    
    # Save combined dataset
    with open("data/training/nexgencreators_training_v1.jsonl", "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Generated {len(new_examples)} new examples")
    print(f"Total examples: {len(all_examples)}")

if __name__ == "__main__":
    main() 