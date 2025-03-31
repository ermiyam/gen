const { InternetFetcher } = require('./InternetFetcher');
const natural = require('natural');

class AIConsciousness {
    constructor() {
        this.internetFetcher = new InternetFetcher();
        this.knowledge = new Map();
        this.tokenizer = new natural.WordTokenizer();
        this.tfidf = new natural.TfIdf();
    }

    async learn(input) {
        try {
            // Check if input is a URL
            if (this.isUrl(input)) {
                if (input.includes('youtube.com') || input.includes('youtu.be')) {
                    const videoId = this.internetFetcher.extractYoutubeId(input);
                    if (videoId) {
                        const content = await this.internetFetcher.fetchYoutubeTranscript(videoId);
                        if (content) {
                            await this.processContent(content, 'youtube', videoId);
                        }
                    }
                } else {
                    const content = await this.internetFetcher.fetchWebContent(input);
                    if (content) {
                        await this.processContent(content, 'web', input);
                    }
                }
            }

            // Process direct input
            await this.processContent(input, 'direct', Date.now());

            return true;
        } catch (error) {
            console.error('Learning error:', error);
            return false;
        }
    }

    async processContent(content, type, id) {
        const tokens = this.tokenizer.tokenize(content.toLowerCase());
        this.tfidf.addDocument(tokens);

        this.knowledge.set(id, {
            type,
            content,
            tokens,
            timestamp: Date.now(),
            processed: true
        });
    }

    isUrl(string) {
        try {
            new URL(string);
            return true;
        } catch {
            return false;
        }
    }

    getKnowledge() {
        return Array.from(this.knowledge.entries());
    }
}

module.exports = { AIConsciousness }; 