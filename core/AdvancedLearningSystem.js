const natural = require('natural');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');
const cheerio = require('cheerio');
const sentiment = require('sentiment');
const { WordNet } = require('natural');
const wordnet = new WordNet();
const TfIdf = natural.TfIdf;

class AdvancedLearningSystem {
    constructor() {
        this.knowledgeBase = {
            conversations: [],
            patterns: new Map(),
            facts: new Map(),
            relationships: new Map(),
            sources: new Set()
        };
        this.learningRate = 0.1;
        this.tfidf = new TfIdf();
        this.visualData = {
            learningProgress: [],
            topicClusters: new Map(),
            confidenceScores: []
        };
        this.classifier = new natural.BayesClassifier();
        this.tokenizer = new natural.WordTokenizer();
        this.sentimentAnalyzer = new sentiment();
        this.contextMemory = [];
        this.topicClusters = new Map();
        this.thinkingStrategies = [
            this.analyzeContext.bind(this),
            this.searchInternet.bind(this),
            this.findSimilarTopics.bind(this),
            this.synthesizeResponse.bind(this)
        ];
        this.initializeLearning();
    }

    async initializeLearning() {
        try {
            await this.loadKnowledge();
            this.startContinuousLearning();
            console.log('Advanced learning system initialized');
        } catch (error) {
            console.log('Starting fresh learning system:', error);
        }
    }

    startContinuousLearning() {
        // Continuously learn in the background
        setInterval(async () => {
            await this.learnFromSources();
        }, 1000 * 60 * 30); // Every 30 minutes
    }

    async learnFromSources() {
        try {
            const sources = [
                this.learnFromReddit(),
                this.learnFromTwitter(),
                this.learnFromNews(),
                this.learnFromBlogs()
            ];

            const results = await Promise.allSettled(sources);
            console.log('Background learning completed:', results);
        } catch (error) {
            console.error('Background learning error:', error);
        }
    }

    async learn(input, response, feedback = null) {
        try {
            // Extract key information
            const analysis = await this.analyzeContent(input);
            
            // Learn patterns
            await this.learnPatterns(input, response, analysis);
            
            // Learn from internet if needed
            if (this.shouldSearchInternet(input)) {
                await this.learnFromInternet(input, analysis);
            }

            // Update relationships
            this.updateRelationships(analysis);

            // Update visualization data
            this.updateVisualization(analysis);

            // Save progress
            await this.saveKnowledge();

            return {
                success: true,
                learningProgress: this.calculateLearningProgress(),
                newPatterns: analysis.patterns.length
            };
        } catch (error) {
            console.error('Learning error:', error);
            return { success: false, error: error.message };
        }
    }

    async analyzeContent(input) {
        const tokens = new natural.WordTokenizer().tokenize(input.toLowerCase());
        const tfidf = new TfIdf();
        tfidf.addDocument(input);

        return {
            tokens,
            patterns: this.extractPatterns(input),
            topics: this.extractTopics(tfidf),
            sentiment: this.analyzeSentiment(input),
            entities: await this.extractEntities(input)
        };
    }

    async learnFromInternet(input, analysis) {
        const sources = {
            google: this.searchGoogle(input),
            reddit: this.searchReddit(input),
            twitter: this.searchTwitter(input),
            news: this.searchNews(input)
        };

        const results = await Promise.allSettled(Object.values(sources));
        const validResults = results
            .filter(r => r.status === 'fulfilled' && r.value)
            .map(r => r.value);

        // Process and store new knowledge
        validResults.forEach(result => {
            this.processNewKnowledge(result, analysis);
        });
    }

    async searchGoogle(query) {
        try {
            const response = await axios.get(
                `https://www.googleapis.com/customsearch/v1?key=${process.env.GOOGLE_API_KEY}&cx=${process.env.GOOGLE_CX}&q=${encodeURIComponent(query)}`
            );
            return this.processSearchResults(response.data.items);
        } catch (error) {
            console.error('Google search error:', error);
            return null;
        }
    }

    async scrapeWebContent(url) {
        try {
            const response = await axios.get(url);
            const $ = cheerio.load(response.data);
            
            // Remove unwanted elements
            $('script, style, nav, footer, header').remove();
            
            // Extract main content
            const content = $('main, article, .content, #content')
                .first()
                .text()
                .replace(/\s+/g, ' ')
                .trim();

            return content;
        } catch (error) {
            console.error('Scraping error:', error);
            return null;
        }
    }

    updateVisualization(analysis) {
        // Update learning progress
        this.visualData.learningProgress.push({
            timestamp: Date.now(),
            patterns: analysis.patterns.length,
            confidence: this.calculateConfidence(analysis)
        });

        // Update topic clusters
        analysis.topics.forEach(topic => {
            if (!this.visualData.topicClusters.has(topic)) {
                this.visualData.topicClusters.set(topic, new Set());
            }
            this.visualData.topicClusters.get(topic).add(analysis.patterns[0]);
        });

        // Update confidence scores
        this.visualData.confidenceScores.push({
            timestamp: Date.now(),
            score: this.calculateConfidence(analysis)
        });

        // Keep visualization data manageable
        if (this.visualData.learningProgress.length > 1000) {
            this.visualData.learningProgress = this.visualData.learningProgress.slice(-1000);
        }
    }

    generateVisualization() {
        return {
            learningProgress: {
                labels: this.visualData.learningProgress.map(p => new Date(p.timestamp).toLocaleTimeString()),
                data: this.visualData.learningProgress.map(p => p.patterns)
            },
            topicClusters: Array.from(this.visualData.topicClusters.entries()).map(([topic, patterns]) => ({
                topic,
                size: patterns.size,
                examples: Array.from(patterns).slice(0, 5)
            })),
            confidenceOverTime: {
                labels: this.visualData.confidenceScores.map(s => new Date(s.timestamp).toLocaleTimeString()),
                data: this.visualData.confidenceScores.map(s => s.score)
            }
        };
    }

    calculateConfidence(analysis) {
        const factors = {
            patternMatch: this.calculatePatternMatchConfidence(analysis.patterns),
            topicRelevance: this.calculateTopicRelevance(analysis.topics),
            sentimentClarity: Math.abs(analysis.sentiment) / 5,
            entityRecognition: analysis.entities.length > 0 ? 0.8 : 0.4
        };

        return Object.values(factors).reduce((sum, value) => sum + value, 0) / Object.keys(factors).length;
    }

    async getInsights() {
        return {
            totalPatterns: this.knowledgeBase.patterns.size,
            topTopics: Array.from(this.visualData.topicClusters.entries())
                .sort((a, b) => b[1].size - a[1].size)
                .slice(0, 5),
            recentLearning: this.visualData.learningProgress.slice(-5),
            averageConfidence: this.calculateAverageConfidence(),
            visualization: this.generateVisualization()
        };
    }

    async think(input) {
        console.log('Thinking about:', input);
        
        const results = {
            context: await this.analyzeContext(input),
            sentiment: this.analyzeSentiment(input),
            topics: await this.extractTopics(input),
            searchResults: await this.searchInternet(input),
            similarCases: this.findSimilarCases(input)
        };

        // Synthesize all information
        const response = await this.synthesizeResponse(results);
        
        // Learn from this thinking process
        this.learnFromThinking(input, results, response);

        return response;
    }

    async analyzeContext(input) {
        // Analyze recent conversation history
        const recentContext = this.contextMemory.slice(-5);
        const contextTopics = new Set();
        
        for (const memory of recentContext) {
            const topics = await this.extractTopics(memory.input);
            topics.forEach(topic => contextTopics.add(topic));
        }

        return {
            topics: Array.from(contextTopics),
            sentiment: this.analyzeSentiment(input),
            timeContext: new Date().getHours()
        };
    }

    async searchInternet(query) {
        try {
            // Search multiple sources
            const [googleResults, wikiResults, redditResults] = await Promise.all([
                this.searchGoogle(query),
                this.searchWikipedia(query),
                this.searchReddit(query)
            ]);

            return this.combineSearchResults([
                ...googleResults,
                ...wikiResults,
                ...redditResults
            ]);
        } catch (error) {
            console.error('Search error:', error);
            return [];
        }
    }

    synthesizeFromResults(results) {
        const relevantInfo = [];
        let confidence = 0;

        // Process search results
        results.searchResults.forEach(result => {
            if (this.isReliableSource(result.source)) {
                relevantInfo.push(result.snippet);
                confidence += 0.2;
            }
        });

        // Process similar cases
        results.similarCases.forEach(case_ => {
            if (case_.success) {
                relevantInfo.push(case_.response);
                confidence += 0.15;
            }
        });

        // Combine information
        const answer = this.combineInformation(relevantInfo);
        
        return {
            answer,
            confidence: Math.min(confidence, 1),
            sources: results.searchResults.map(r => r.source)
        };
    }

    async learnFromThinking(input, results, response) {
        // Store in knowledge base
        this.knowledgeBase.patterns[input] = {
            response: response.mainAnswer,
            confidence: response.confidence,
            context: results.context,
            timestamp: Date.now()
        };

        // Update topic clusters
        results.topics.forEach(topic => {
            if (!this.topicClusters.has(topic)) {
                this.topicClusters.set(topic, new Set());
            }
            this.topicClusters.get(topic).add(input);
        });

        // Update context memory
        this.contextMemory.push({
            input,
            response: response.mainAnswer,
            timestamp: Date.now(),
            topics: results.topics
        });

        // Trim context memory if too long
        if (this.contextMemory.length > 100) {
            this.contextMemory = this.contextMemory.slice(-50);
        }

        // Save to persistent storage
        this.saveKnowledge();
    }

    analyzeSentiment(text) {
        return this.sentimentAnalyzer.analyze(text);
    }

    async extractTopics(text) {
        const tfidf = new TfIdf();
        tfidf.addDocument(text);
        
        return new Promise((resolve) => {
            const topics = [];
            tfidf.listTerms(0).slice(0, 5).forEach(item => {
                topics.push(item.term);
            });
            resolve(topics);
        });
    }

    findSimilarCases(input) {
        const inputTopics = new Set(this.tokenizer.tokenize(input.toLowerCase()));
        const similarCases = [];

        this.contextMemory.forEach(memory => {
            const memoryTopics = new Set(this.tokenizer.tokenize(memory.input.toLowerCase()));
            const similarity = this.calculateJaccardSimilarity(inputTopics, memoryTopics);
            
            if (similarity > 0.3) {
                similarCases.push({
                    input: memory.input,
                    response: memory.response,
                    similarity
                });
            }
        });

        return similarCases.sort((a, b) => b.similarity - a.similarity);
    }

    calculateJaccardSimilarity(set1, set2) {
        const intersection = new Set([...set1].filter(x => set2.has(x)));
        const union = new Set([...set1, ...set2]);
        return intersection.size / union.size;
    }
}

module.exports = new AdvancedLearningSystem(); 