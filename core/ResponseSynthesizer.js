const natural = require('natural');
const { Configuration, OpenAIApi } = require('openai');

class ResponseSynthesizer {
    constructor() {
        this.openai = new OpenAIApi(new Configuration({
            apiKey: process.env.OPENAI_API_KEY
        }));
        this.tokenizer = new natural.WordTokenizer();
    }

    async synthesizeResponse(data) {
        const { searchResults, context, sentiment } = data;
        
        // Combine different types of information
        const synthesized = await this.combineInformation(searchResults);
        
        // Enhance with AI processing
        const enhanced = await this.enhanceWithAI(synthesized);
        
        // Format based on context and sentiment
        return this.formatResponse(enhanced, context, sentiment);
    }

    async combineInformation(results) {
        // Process and combine information from different sources
        const processed = results.map(result => {
            return {
                content: this.extractMainPoints(result.content),
                source: result.source,
                confidence: this.calculateConfidence(result)
            };
        });

        return this.mergeInformation(processed);
    }

    async enhanceWithAI(information) {
        try {
            const completion = await this.openai.createCompletion({
                model: "text-davinci-003",
                prompt: `Enhance and organize this information: ${JSON.stringify(information)}`,
                max_tokens: 500
            });
            return completion.data.choices[0].text;
        } catch (error) {
            console.error('AI enhancement error:', error);
            return information;
        }
    }

    formatResponse(content, context, sentiment) {
        // Format response with markdown and emojis
        let formatted = `## Response\n\n`;
        
        if (sentiment.score < 0) {
            formatted += `I understand this might be challenging. Let me help! 🤝\n\n`;
        }

        formatted += `${content}\n\n`;

        if (context.sources.length > 0) {
            formatted += `\n### Sources:\n`;
            context.sources.forEach(source => {
                formatted += `- ${source}\n`;
            });
        }

        return formatted;
    }
} 