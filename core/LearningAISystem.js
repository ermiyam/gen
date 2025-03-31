const axios = require('axios');
const natural = require('natural');
const { YoutubeTranscript } = require('youtube-transcript');
const cheerio = require('cheerio');
const fs = require('fs-extra');
const path = require('path');

class LearningAISystem {
    constructor() {
        this.knowledge = {
            topics: new Map(),
            sources: new Map(),
            interactions: [],
            youtube: new Map(),
            websites: new Map()
        };
        
        this.classifier = new natural.BayesClassifier();
        this.tokenizer = new natural.WordTokenizer();
        
        this.knowledgePath = path.join(__dirname, 'knowledge_base.json');
        this.initialize();
    }

    async initialize() {
        try {
            // Load existing knowledge if available
            await this.loadKnowledge();
            console.log('✅ Learning AI System initialized');
        } catch (error) {
            console.error('Initialization error:', error);
            // Start with fresh knowledge base
            await this.saveKnowledge();
        }
    }

    async learn(input, source = 'user') {
        try {
            switch (source) {
                case 'youtube':
                    await this.learnFromYoutube(input); // input = video URL
                    break;
                case 'website':
                    await this.learnFromWebsite(input); // input = website URL
                    break;
                case 'user':
                    await this.learnFromUser(input); // input = user message
                    break;
            }

            // Save updated knowledge
            await this.saveKnowledge();
            return true;
        } catch (error) {
            console.error('Learning error:', error);
            return false;
        }
    }

    async learnFromYoutube(videoUrl) {
        try {
            // Extract video ID from URL
            const videoId = videoUrl.split('v=')[1];
            
            // Get video transcript
            const transcript = await YoutubeTranscript.fetchTranscript(videoId);
            
            // Process transcript
            const text = transcript.map(item => item.text).join(' ');
            const topics = this.extractTopics(text);
            
            // Store in knowledge base
            this.knowledge.youtube.set(videoId, {
                url: videoUrl,
                topics,
                timestamp: Date.now()
            });

            console.log(`✅ Learned from YouTube video: ${videoUrl}`);
            return true;
        } catch (error) {
            console.error('YouTube learning error:', error);
            return false;
        }
    }

    async learnFromWebsite(url) {
        try {
            // Fetch website content
            const response = await axios.get(url);
            const $ = cheerio.load(response.data);
            
            // Extract main content (remove ads, navigation, etc.)
            const content = $('article, main, .content')
                .text()
                .replace(/\s+/g, ' ')
                .trim();
            
            // Process content
            const topics = this.extractTopics(content);
            
            // Store in knowledge base
            this.knowledge.websites.set(url, {
                content: content.substring(0, 1000), // Store preview
                topics,
                timestamp: Date.now()
            });

            console.log(`✅ Learned from website: ${url}`);
            return true;
        } catch (error) {
            console.error('Website learning error:', error);
            return false;
        }
    }

    async learnFromUser(input) {
        try {
            // Process user input
            const tokens = this.tokenizer.tokenize(input.toLowerCase());
            const topics = this.extractTopics(input);
            
            // Store interaction
            this.knowledge.interactions.push({
                input,
                topics,
                timestamp: Date.now()
            });

            // Update classifier
            this.classifier.addDocument(tokens, topics[0]);
            await this.classifier.train();

            console.log('✅ Learned from user interaction');
            return true;
        } catch (error) {
            console.error('User learning error:', error);
            return false;
        }
    }

    extractTopics(text) {
        const tokens = this.tokenizer.tokenize(text.toLowerCase());
        const topics = new Set();
        
        // Extract key phrases and topics
        for (let i = 0; i < tokens.length; i++) {
            if (this.isRelevantTopic(tokens[i])) {
                topics.add(tokens[i]);
            }
            if (i < tokens.length - 1) {
                const phrase = `${tokens[i]} ${tokens[i + 1]}`;
                if (this.isRelevantTopic(phrase)) {
                    topics.add(phrase);
                }
            }
        }
        
        return Array.from(topics);
    }

    isRelevantTopic(topic) {
        // Add your topic relevance logic here
        const stopWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at'];
        return !stopWords.includes(topic) && topic.length > 3;
    }

    async generateResponse(input) {
        try {
            const topics = this.extractTopics(input);
            let response = '';
            
            // Find relevant knowledge
            const relevantYoutube = this.findRelevantYoutubeContent(topics);
            const relevantWebsites = this.findRelevantWebsites(topics);
            const relevantInteractions = this.findRelevantInteractions(topics);
            
            // Combine knowledge to generate response
            response = this.synthesizeResponse(
                input,
                relevantYoutube,
                relevantWebsites,
                relevantInteractions
            );
            
            return response;
        } catch (error) {
            console.error('Response generation error:', error);
            return 'I apologize, but I encountered an error generating a response.';
        }
    }

    findRelevantYoutubeContent(topics) {
        const relevant = [];
        this.knowledge.youtube.forEach((content, videoId) => {
            if (topics.some(topic => content.topics.includes(topic))) {
                relevant.push(content);
            }
        });
        return relevant;
    }

    findRelevantWebsites(topics) {
        const relevant = [];
        this.knowledge.websites.forEach((content, url) => {
            if (topics.some(topic => content.topics.includes(topic))) {
                relevant.push(content);
            }
        });
        return relevant;
    }

    findRelevantInteractions(topics) {
        return this.knowledge.interactions.filter(interaction =>
            topics.some(topic => interaction.topics.includes(topic))
        );
    }

    synthesizeResponse(input, youtube, websites, interactions) {
        let response = `Based on what I've learned about "${input}":\n\n`;
        
        if (youtube.length > 0) {
            response += "From YouTube content: " + this.summarizeContent(youtube) + "\n\n";
        }
        
        if (websites.length > 0) {
            response += "From website content: " + this.summarizeContent(websites) + "\n\n";
        }
        
        if (interactions.length > 0) {
            response += "From previous interactions: " + this.summarizeInteractions(interactions) + "\n";
        }
        
        return response || "I'm still learning about this topic. Could you teach me more?";
    }

    summarizeContent(contents) {
        // Implement your content summarization logic here
        return contents.map(c => c.topics.join(', ')).join('; ');
    }

    summarizeInteractions(interactions) {
        return interactions
            .slice(-3)
            .map(i => i.topics.join(', '))
            .join('; ');
    }

    async loadKnowledge() {
        if (await fs.exists(this.knowledgePath)) {
            const data = await fs.readJson(this.knowledgePath);
            this.knowledge = {
                ...this.knowledge,
                ...data
            };
        }
    }

    async saveKnowledge() {
        await fs.writeJson(this.knowledgePath, this.knowledge);
    }
}

module.exports = { LearningAISystem }; 