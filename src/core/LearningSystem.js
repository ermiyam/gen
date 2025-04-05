const axios = require('axios');
const cheerio = require('cheerio');
const natural = require('natural');

class LearningSystem {
    constructor() {
        this.learningQueue = new Set();
        this.processedUrls = new Set();
        this.knowledge = new Map();
        this.tokenizer = new natural.WordTokenizer();
        this.tfidf = new natural.TfIdf();
        
        // Learning parameters
        this.learningRate = 0.1;
        this.minConfidence = 0.7;
    }

    async learn() {
        try {
            // Process learning queue
            for (const url of this.learningQueue) {
                if (!this.processedUrls.has(url)) {
                    await this.learnFromUrl(url);
                    this.processedUrls.add(url);
                }
            }

            // Generate new insights
            this.generateInsights();

            // Update learning parameters
            this.updateLearningParameters();

        } catch (error) {
            console.error('Learning error:', error);
        }
    }

    async learnFromUrl(url) {
        try {
            const { data } = await axios.get(url);
            const $ = cheerio.load(data);
            
            // Extract text content
            const content = $('p, h1, h2, h3, h4, h5, h6')
                .map((_, el) => $(el).text())
                .get()
                .join(' ');

            // Process content
            const tokens = this.tokenizer.tokenize(content.toLowerCase());
            this.tfidf.addDocument(tokens);

            // Extract key concepts
            const concepts = this.extractConcepts(tokens);
            
            // Store knowledge
            this.storeKnowledge(url, concepts);

        } catch (error) {
            console.error(`Error learning from ${url}:`, error);
        }
    }

    extractConcepts(tokens) {
        const concepts = new Map();
        
        // Calculate term frequency
        tokens.forEach(token => {
            concepts.set(token, (concepts.get(token) || 0) + 1);
        });

        // Filter significant concepts
        return new Map(
            Array.from(concepts.entries())
                .filter(([, freq]) => freq > 1)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 100)
        );
    }

    storeKnowledge(source, concepts) {
        this.knowledge.set(source, {
            concepts,
            timestamp: Date.now(),
            confidence: this.calculateConfidence(concepts)
        });
    }

    calculateConfidence(concepts) {
        return Math.min(
            1,
            Array.from(concepts.values()).reduce((sum, freq) => sum + freq, 0) / 1000
        );
    }

    generateInsights() {
        const allConcepts = new Map();
        
        // Combine concepts from all sources
        for (const { concepts } of this.knowledge.values()) {
            for (const [concept, freq] of concepts) {
                allConcepts.set(
                    concept,
                    (allConcepts.get(concept) || 0) + freq
                );
            }
        }

        // Generate insights from top concepts
        const insights = Array.from(allConcepts.entries())
            .sort(([, a], [, b]) => b - a)
            .slice(0, 10)
            .map(([concept]) => concept);

        return insights;
    }

    updateLearningParameters() {
        // Adjust learning rate based on success
        const successRate = Array.from(this.knowledge.values())
            .filter(k => k.confidence > this.minConfidence)
            .length / this.knowledge.size;

        this.learningRate = Math.max(0.01, Math.min(0.5, this.learningRate * (1 + (successRate - 0.5))));
    }

    addToLearningQueue(url) {
        if (this.isValidUrl(url)) {
            this.learningQueue.add(url);
            return true;
        }
        return false;
    }

    isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch {
            return false;
        }
    }

    getStats() {
        return {
            queueSize: this.learningQueue.size,
            processedUrls: this.processedUrls.size,
            knowledgeSize: this.knowledge.size,
            learningRate: this.learningRate,
            confidence: this.calculateAverageConfidence()
        };
    }

    calculateAverageConfidence() {
        if (this.knowledge.size === 0) return 0;
        
        return Array.from(this.knowledge.values())
            .reduce((sum, k) => sum + k.confidence, 0) / this.knowledge.size;
    }
}

module.exports = { LearningSystem }; 