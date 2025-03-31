const natural = require('natural');
const tokenizer = new natural.WordTokenizer();

class ConversationalLayer {
    constructor() {
        this.conversationHistory = [];
        this.personalityTraits = {
            friendliness: 0.8,
            enthusiasm: 0.7,
            helpfulness: 0.9,
            creativity: 0.8
        };
        
        this.conversationPatterns = {
            greetings: ['hi', 'hello', 'hey', 'howdy', 'good morning', 'good afternoon', 'good evening'],
            farewells: ['bye', 'goodbye', 'see you', 'talk to you later', 'cya'],
            thanks: ['thank you', 'thanks', 'appreciate it', 'thx'],
            questions: ['what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'will']
        };
    }

    async processInput(input, context = {}) {
        const response = {
            type: 'general',
            content: '',
            confidence: 0,
            followUp: null
        };

        // Normalize input
        const normalizedInput = input.toLowerCase().trim();
        const tokens = tokenizer.tokenize(normalizedInput);

        // Determine input type and generate response
        if (this.isGreeting(normalizedInput)) {
            response.type = 'greeting';
            response.content = this.generateGreeting(context);
            response.confidence = 0.9;
        } 
        else if (this.isQuestion(tokens)) {
            response.type = 'question';
            response.content = await this.generateAnswer(normalizedInput, context);
            response.followUp = this.generateFollowUp(normalizedInput);
        }
        else if (this.isFarewell(normalizedInput)) {
            response.type = 'farewell';
            response.content = this.generateFarewell(context);
            response.confidence = 0.9;
        }
        else {
            response.content = await this.generateConversationalResponse(normalizedInput, context);
        }

        // Add to conversation history
        this.updateConversationHistory(input, response);

        return response;
    }

    isGreeting(input) {
        return this.conversationPatterns.greetings.some(greeting => 
            input.includes(greeting));
    }

    isQuestion(tokens) {
        return this.conversationPatterns.questions.some(question => 
            tokens.includes(question));
    }

    isFarewell(input) {
        return this.conversationPatterns.farewells.some(farewell => 
            input.includes(farewell));
    }

    generateGreeting(context) {
        const greetings = [
            "Hi there! 👋 How can I help you today?",
            "Hello! I'm excited to chat with you!",
            "Hey! What's on your mind?",
            "Hi! I'm ready to help with anything you need!"
        ];
        return this.selectRandomResponse(greetings);
    }

    async generateAnswer(input, context) {
        // Check if it's a viral marketing question
        if (input.includes('viral') || input.includes('trending')) {
            return this.generateViralMarketingAdvice();
        }

        // Handle other types of questions
        const response = await this.analyzeAndRespondToQuestion(input);
        return response;
    }

    async generateViralMarketingAdvice() {
        const viralStrategies = [
            {
                strategy: "Create a TikTok Challenge",
                details: "Design a fun, easy-to-replicate challenge that resonates with your target audience. Make it unique and engaging!",
                examples: ["Dance challenges", "Transformation videos", "Skill demonstrations"]
            },
            {
                strategy: "Leverage Trending Sounds/Music",
                details: "Use popular songs or trending sounds in your content to increase visibility and engagement.",
                examples: ["Popular song remixes", "Viral sound effects", "Trending background music"]
            },
            {
                strategy: "Emotional Storytelling",
                details: "Create content that triggers strong emotions - happiness, surprise, or inspiration work best for viral content.",
                examples: ["Behind-the-scenes stories", "Transformation journeys", "Unexpected plot twists"]
            }
        ];

        const selected = viralStrategies[Math.floor(Math.random() * viralStrategies.length)];
        
        return `Here's a powerful strategy to help you go viral: 

${selected.strategy}! 🚀

${selected.details}

For example, you could try:
${selected.examples.map(ex => `• ${ex}`).join('\n')}

Would you like me to elaborate on this strategy or suggest another one?`;
    }

    async analyzeAndRespondToQuestion(input) {
        // Determine question type and context
        const questionType = this.determineQuestionType(input);
        
        switch (questionType) {
            case 'how-to':
                return this.generateHowToResponse(input);
            case 'what-is':
                return this.generateDefinitionResponse(input);
            case 'comparison':
                return this.generateComparisonResponse(input);
            default:
                return this.generateGeneralResponse(input);
        }
    }

    generateFollowUp(input) {
        const followUps = [
            "Would you like to know more about this?",
            "Should I elaborate on any part?",
            "What specific aspect interests you most?",
            "Would you like some examples?"
        ];
        return this.selectRandomResponse(followUps);
    }

    selectRandomResponse(responses) {
        return responses[Math.floor(Math.random() * responses.length)];
    }

    updateConversationHistory(input, response) {
        this.conversationHistory.push({
            timestamp: Date.now(),
            input,
            response,
            context: {}
        });
    }
}

module.exports = new ConversationalLayer(); 