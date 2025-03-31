const natural = require('natural');
const axios = require('axios');
const LearningSystem = require('./LearningSystem');

class AISystem {
    constructor() {
        this.initialized = false;
        this.context = new Map();
        
        // Base knowledge for common questions and topics
        this.knowledge = {
            greetings: ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
            identity: ['are you real', 'who are you', 'what are you', 'are you ai', 'are you human'],
            capabilities: ['what can you do', 'help', 'what do you know', 'what are your capabilities'],
            marketing: ['marketing', 'advertise', 'promote', 'brand', 'social media', 'campaign']
        };

        this.initialize();
    }

    initialize() {
        try {
            this.initialized = true;
            console.log('✅ AI System initialized');
        } catch (error) {
            console.error('AI System initialization error:', error);
        }
    }

    async processMessage(message) {
        try {
            if (!message) return 'Please provide a message.';
            
            const lowerMessage = message.toLowerCase().trim();
            
            // Analyze message intent
            const intent = this.analyzeIntent(lowerMessage);
            
            // Generate contextual response
            return this.generateResponse(message, intent);
            
        } catch (error) {
            console.error('Message processing error:', error);
            return 'I apologize, but I encountered an error. Please try again.';
        }
    }

    analyzeIntent(message) {
        // Check for different types of intents
        if (this.knowledge.greetings.some(g => message.includes(g))) {
            return 'greeting';
        }
        if (this.knowledge.identity.some(i => message.includes(i))) {
            return 'identity';
        }
        if (this.knowledge.capabilities.some(c => message.includes(c))) {
            return 'capabilities';
        }
        if (this.knowledge.marketing.some(m => message.includes(m))) {
            return 'marketing';
        }
        
        // If no specific intent is found, treat as general conversation
        return 'general';
    }

    generateResponse(message, intent) {
        switch (intent) {
            case 'greeting':
                return this.getRandomResponse([
                    "Hello! I'm your AI marketing assistant. How can I help you today?",
                    "Hi there! Ready to discuss your marketing needs!",
                    "Greetings! I'm here to help with your marketing strategies!"
                ]);

            case 'identity':
                return this.getRandomResponse([
                    "I'm an AI marketing assistant, designed to help you with marketing strategies and ideas. While I'm not human, I'm knowledgeable about marketing and eager to help!",
                    "I'm an artificial intelligence specialized in marketing. I can provide real, practical marketing advice and strategies.",
                    "I'm an AI assistant focused on marketing. I'm direct about being AI, but my marketing knowledge and assistance are very real!"
                ]);

            case 'capabilities':
                return this.getRandomResponse([
                    "I can help with marketing strategies, content creation, social media planning, brand development, and campaign optimization. What specific area interests you?",
                    "My capabilities include marketing analysis, strategy development, content suggestions, and campaign planning. What would you like to focus on?",
                    "I can assist with various marketing aspects - from social media strategies to brand development. What's your current marketing challenge?"
                ]);

            case 'marketing':
                return this.getRandomResponse([
                    "Let's talk about your marketing goals! Are you looking to increase brand awareness, drive sales, or engage with your audience better?",
                    "Marketing is my specialty! Would you like to discuss specific strategies for your business?",
                    "Great to discuss marketing! What's your current biggest marketing challenge?"
                ]);

            case 'general':
                return this.generateContextualResponse(message);
        }
    }

    generateContextualResponse(message) {
        // Generate more natural responses for general conversation
        return this.getRandomResponse([
            `I understand you're asking about "${message}". Could you tell me more about how this relates to your marketing goals?`,
            `Interesting question about "${message}". How does this fit into your marketing strategy?`,
            `Let's explore "${message}" from a marketing perspective. What specific aspects would you like to focus on?`,
            `That's an interesting point about "${message}". Would you like to discuss how we can incorporate this into your marketing approach?`
        ]);
    }

    getRandomResponse(responses) {
        return responses[Math.floor(Math.random() * responses.length)];
    }

    getStatus() {
        return {
            initialized: this.initialized,
            knowledgeBaseSize: Object.keys(this.knowledge).length
        };
    }

    tokenizer = new natural.WordTokenizer();
    memory = new Map();
    conversationContext = [];
    personality = {
        friendly: 0.9,
        professional: 0.8,
        helpful: 1.0,
        enthusiastic: 0.7
    };

    updateContext(message) {
        this.conversationContext.push({
            message,
            timestamp: Date.now(),
            analyzed: this.quickAnalysis(message)
        });

        // Keep context manageable
        if (this.conversationContext.length > 10) {
            this.conversationContext.shift();
        }
    }

    quickAnalysis(message) {
        const lowercase = message.toLowerCase();
        return {
            isQuestion: message.includes('?'),
            hasUrgency: /urgent|asap|quickly|now/i.test(message),
            isPositive: /great|good|awesome|nice|thanks|perfect/i.test(message),
            isNegative: /bad|wrong|not working|error|fail/i.test(message)
        };
    }

    analyzeSentiment(message) {
        const positive = /great|good|awesome|nice|thanks|perfect|yes|sure|okay/i;
        const negative = /bad|wrong|not|no|error|fail|cant|cannot|don't|dont/i;
        
        let score = 0;
        if (positive.test(message)) score += 1;
        if (negative.test(message)) score -= 1;
        
        return {
            score,
            isPositive: score > 0,
            isNegative: score < 0,
            isNeutral: score === 0
        };
    }

    analyzeTopic(message) {
        const topics = {
            socialMedia: /(facebook|instagram|twitter|tiktok|linkedin|social\s*media)/i,
            viral: /(viral|trending|trend|popular)/i,
            content: /(content|post|video|photo|blog|article)/i,
            strategy: /(strategy|plan|approach|campaign)/i,
            analytics: /(analytics|metrics|numbers|statistics|data)/i,
            audience: /(audience|followers|customers|users|people)/i
        };

        return Object.entries(topics)
            .filter(([_, regex]) => regex.test(message))
            .map(([topic, _]) => topic);
    }

    analyzeUrgency(message) {
        const urgentTerms = /urgent|asap|quickly|now|fast|immediate/i;
        return urgentTerms.test(message);
    }

    async generateEnhancedResponse(message, analysis) {
        // Handle multiple intents
        if (analysis.intent.length > 0) {
            const responses = await Promise.all(
                analysis.intent.map(intent => this.getResponseForIntent(intent, message, analysis))
            );

            // Combine responses intelligently
            return this.combineResponses(responses, analysis);
        }

        return this.generateGeneralResponse(message, analysis);
    }

    async getResponseForIntent(intent, message, analysis) {
        const responses = {
            greeting: [
                "Hey there! 👋 I'm excited to help with your marketing journey!",
                "Hello! Ready to create some marketing magic together?",
                "Hi! Let's make your marketing goals a reality!",
                "Greetings! I'm your AI marketing partner - what shall we tackle today?"
            ],
            wellBeing: [
                "I'm doing great and ready to help! Thanks for asking. 😊",
                "I'm functioning perfectly and excited to assist with your marketing needs!",
                "All systems go! I'm here and ready to help you succeed!"
            ],
            marketing: {
                social: "Let's boost your social media presence! Here's what works best:\n" +
                       "1️⃣ Create engaging, shareable content\n" +
                       "2️⃣ Post consistently (I recommend 3-5 times per week)\n" +
                       "3️⃣ Engage with your audience actively\n" +
                       "4️⃣ Use trending hashtags strategically\n" +
                       "5️⃣ Analyze and adapt your strategy\n\n" +
                       "Would you like me to elaborate on any of these points?",
                
                viral: "Want to go viral? Here's your action plan:\n" +
                      "🚀 Study trending content in your niche\n" +
                      "🎯 Create emotional, relatable content\n" +
                      "⏰ Post at peak engagement times\n" +
                      "🔄 Encourage sharing and interaction\n" +
                      "📈 Leverage trending sounds/topics\n\n" +
                      "Which aspect would you like to explore further?",
                
                strategy: "Here's a proven marketing strategy framework:\n" +
                         "1. Define clear, measurable goals\n" +
                         "2. Identify your target audience\n" +
                         "3. Choose the right platforms\n" +
                         "4. Create compelling content\n" +
                         "5. Measure and optimize results\n\n" +
                         "Shall we dive deeper into any of these steps?"
            }
        };

        // Select appropriate response based on intent and context
        let response = responses[intent];
        if (Array.isArray(response)) {
            response = response[Math.floor(Math.random() * response.length)];
        } else if (typeof response === 'object') {
            // Select specific marketing response based on topic
            const topic = analysis.topic[0] || 'strategy';
            response = response[topic] || response.strategy;
        }

        return response;
    }

    combineResponses(responses, analysis) {
        // Remove duplicates and empty responses
        const uniqueResponses = [...new Set(responses.filter(r => r))];
        
        // If we have multiple responses, combine them naturally
        if (uniqueResponses.length > 1) {
            return uniqueResponses.join('\n\n');
        }

        return uniqueResponses[0] || this.generateGeneralResponse(message, analysis);
    }

    generateGeneralResponse(message, analysis) {
        const generalResponses = [
            "I understand you're interested in marketing. Could you tell me more about your specific goals?",
            "Let's focus on your marketing objectives. What's your main challenge right now?",
            "I'm here to help with any marketing needs. What aspect would you like to explore?",
            "Tell me more about your marketing goals, and I'll help you create a solid strategy!"
        ];

        return generalResponses[Math.floor(Math.random() * generalResponses.length)];
    }

    generateErrorResponse() {
        const errorResponses = [
            "I apologize, but I encountered a small hiccup. Could you rephrase that?",
            "Oops! Let me recalibrate. Mind trying that again?",
            "Even AIs have their moments! Could you say that differently?",
            "I want to help, but I'm having trouble understanding. Could you rephrase that?"
        ];

        return errorResponses[Math.floor(Math.random() * errorResponses.length)];
    }

    async searchInternet(query) {
        try {
            // Implement safe internet searching here if needed
            return null;
        } catch (error) {
            console.error('Search error:', error);
            return null;
        }
    }
}

// Export the class directly
module.exports = AISystem; 