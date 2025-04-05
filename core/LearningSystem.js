const fse = require('fs-extra');
const path = require('path');
const axios = require('axios');
const natural = require('natural');
const fsPromises = require('fs').promises;
const mongoose = require('mongoose');

class LearningSystem {
    constructor() {
        this.isLearning = false;
        this.learningInterval = null;
        this.knowledgeBase = {
            conversations: [],
            patterns: [],
            facts: [],
            timestamp: Date.now()
        };
        this.dbConnection = null;
        this.learningConfig = {
            interval: 1000 * 60 * 30, // 30 minutes
            maxConcurrent: 3,
            focusAreas: ['marketing', 'social media', 'content creation']
        };
        this.learningRate = 0.1;
        this.initialized = false;
        this.classifier = new natural.BayesClassifier();
        
        // Fix mongoose deprecation warning
        mongoose.set('strictQuery', true);
        
        this.initialize();
    }

    async initialize() {
        try {
            // Ensure knowledge.json exists
            const knowledgePath = path.join(__dirname, 'knowledge.json');
            await fse.ensureFile(knowledgePath);
            
            try {
                const data = await fse.readJson(knowledgePath);
                this.knowledgeBase = data;
            } catch (error) {
                // If file is empty or invalid, write default knowledge base
                await fse.writeJson(knowledgePath, this.knowledgeBase);
            }

            // Try connecting to MongoDB with fallback
            try {
                await mongoose.connect('mongodb://127.0.0.1:27017/ai_marketing', {
                    useNewUrlParser: true,
                    useUnifiedTopology: true,
                    serverSelectionTimeoutMS: 5000 // 5 second timeout
                });
                console.log('✅ Connected to MongoDB');
            } catch (mongoError) {
                console.log('MongoDB connection failed, using file storage fallback');
                // Continue with file-based storage
            }

            this.initialized = true;
            console.log('✅ Learning System initialized with knowledge base');
        } catch (error) {
            console.error('Learning System initialization error:', error);
            // Continue with basic functionality
        }
    }

    async connectToDatabase() {
        try {
            await mongoose.connect('mongodb://localhost:27017/ai_marketing', {
                useNewUrlParser: true,
                useUnifiedTopology: true
            });
            console.log('✅ Connected to MongoDB');
        } catch (error) {
            console.error('MongoDB connection error:', error);
            throw error;
        }
    }

    async initializeFallback() {
        console.log('Using fallback storage system');
        // Implement fallback storage using local files
    }

    async startContinuousLearning() {
        if (this.isLearning) return;
        
        this.isLearning = true;
        console.log('Starting continuous learning process...');

        this.learningInterval = setInterval(async () => {
            await this.learnIteration();
        }, this.learningConfig.interval);
    }

    async learnIteration() {
        try {
            // Get current focus areas
            const topics = await this.getCurrentLearningTopics();
            
            // Learn from each topic concurrently
            const learningPromises = topics.map(topic => 
                this.learnAboutTopic(topic));
            
            const results = await Promise.all(learningPromises);
            
            // Save new knowledge
            await this.saveKnowledge(results);
            
            // Log learning progress
            await this.logLearningProgress(results);
            
        } catch (error) {
            console.error('Learning iteration error:', error);
        }
    }

    async learnAboutTopic(topic) {
        const sources = await this.gatherSources(topic);
        const newKnowledge = await this.processInformation(sources);
        return {
            topic,
            knowledge: newKnowledge,
            timestamp: Date.now()
        };
    }

    async gatherSources(topic) {
        const sources = [];
        
        // Gather from multiple sources
        try {
            // RSS Feeds
            const rssFeeds = await this.fetchRSSFeeds(topic);
            sources.push(...rssFeeds);

            // API Sources
            const apiData = await this.fetchAPIData(topic);
            sources.push(...apiData);

            // Web Scraping
            const webData = await this.scrapeWebData(topic);
            sources.push(...webData);

        } catch (error) {
            console.error(`Error gathering sources for ${topic}:`, error);
        }

        return sources;
    }

    async saveKnowledge(newKnowledge) {
        try {
            // Save to MongoDB
            await KnowledgeModel.insertMany(newKnowledge);
            
            // Update in-memory knowledge base
            newKnowledge.forEach(item => {
                this.knowledgeBase.set(item.topic, item);
            });

            // Backup to file system
            await this.backupKnowledge();
        } catch (error) {
            console.error('Error saving knowledge:', error);
        }
    }

    async backupKnowledge() {
        const backupPath = path.join(__dirname, '../data/knowledge_backup.json');
        const knowledge = Array.from(this.knowledgeBase.entries());
        await fse.writeJson(backupPath, knowledge);
    }

    async learn(input, response, wasSuccessful = true) {
        try {
            this.knowledgeBase.conversations.push({
                input,
                response,
                timestamp: Date.now()
            });
            
            await this.saveKnowledge();
            return true;
        } catch (error) {
            console.error('Learning error:', error);
            return false;
        }
    }

    async searchAndLearn(query) {
        try {
            const response = await axios.get(
                `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json`
            );
            
            if (response.data && response.data.AbstractText) {
                return {
                    fact: response.data.AbstractText,
                    source: 'DuckDuckGo',
                    timestamp: Date.now()
                };
            }
            return null;
        } catch (error) {
            console.error('Search error:', error);
            return null;
        }
    }

    extractPatterns(input) {
        const tokens = input.toLowerCase().split(' ');
        const patterns = new Set();
        
        // Single words
        tokens.forEach(token => patterns.add(token));
        
        // Word pairs
        for (let i = 0; i < tokens.length - 1; i++) {
            patterns.add(`${tokens[i]} ${tokens[i + 1]}`);
        }
        
        return Array.from(patterns);
    }

    async generateResponse(input) {
        // Check patterns for learned responses
        const patterns = this.extractPatterns(input);
        const potentialResponses = new Map();

        patterns.forEach(pattern => {
            if (this.knowledgeBase.patterns.has(pattern)) {
                const patternData = this.knowledgeBase.patterns.get(pattern);
                patternData.successfulResponses.forEach((count, response) => {
                    const currentCount = potentialResponses.get(response) || 0;
                    potentialResponses.set(response, currentCount + count);
                });
            }
        });

        // Get the best learned response
        let bestResponse = null;
        let bestCount = 0;
        potentialResponses.forEach((count, response) => {
            if (count > bestCount) {
                bestCount = count;
                bestResponse = response;
            }
        });

        // Check facts
        const relevantFact = this.knowledgeBase.facts.get(input);
        if (relevantFact && relevantFact.fact) {
            return {
                response: relevantFact.fact,
                confidence: 0.8,
                source: 'learned'
            };
        }

        return bestResponse ? {
            response: bestResponse,
            confidence: bestCount / patterns.length,
            source: 'learned'
        } : null;
    }

    shouldSearchInternet(input) {
        return input.length > 10 && 
               !this.knowledgeBase.facts.has(input) &&
               /what|how|why|when|where|who/i.test(input);
    }

    async saveKnowledge() {
        try {
            const knowledgePath = path.join(__dirname, 'knowledge.json');
            await fse.writeJson(knowledgePath, this.knowledgeBase);
        } catch (error) {
            console.error('Error saving knowledge:', error);
        }
    }

    async loadKnowledge() {
        try {
            const data = await fsPromises.readFile(
                path.join(__dirname, 'knowledge.json'),
                'utf8'
            );
            const knowledge = JSON.parse(data);
            
            this.knowledgeBase.conversations = knowledge.conversations;
            this.knowledgeBase.patterns = new Map(knowledge.patterns);
            this.knowledgeBase.facts = new Map(knowledge.facts);
            
            console.log('Loaded knowledge base:', {
                conversations: this.knowledgeBase.conversations.length,
                patterns: this.knowledgeBase.patterns.size,
                facts: this.knowledgeBase.facts.size
            });
        } catch (error) {
            console.error('Error loading knowledge:', error);
        }
    }

    getStats() {
        return {
            conversationsLearned: this.knowledgeBase.conversations.length,
            patternsLearned: this.knowledgeBase.patterns.size,
            factsLearned: this.knowledgeBase.facts.size,
            lastUpdated: new Date().toISOString()
        };
    }
}

// MongoDB Schema
const KnowledgeSchema = new mongoose.Schema({
    topic: String,
    knowledge: mongoose.Schema.Types.Mixed,
    timestamp: Date,
    confidence: Number,
    sources: [String]
});

const KnowledgeModel = mongoose.model('Knowledge', KnowledgeSchema);

module.exports = { LearningSystem }; 