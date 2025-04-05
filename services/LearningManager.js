const axios = require('axios');
const natural = require('natural');
const { Configuration, OpenAIApi } = require('openai');
const cheerio = require('cheerio');

class LearningManager {
    constructor() {
        this.knowledgeBase = new Map();
        this.learningQueue = [];
        this.searchAPIs = {
            google: process.env.GOOGLE_API_KEY,
            openai: process.env.OPENAI_API_KEY
        };
        this.openai = new OpenAIApi(new Configuration({
            apiKey: this.searchAPIs.openai
        }));
    }

    async learnFromQuery(query) {
        try {
            // 1. Extract keywords from query
            const keywords = this.extractKeywords(query);
            
            // 2. Search internet for relevant information
            const searchResults = await this.searchMultipleSources(keywords);
            
            // 3. Process and validate information
            const processedInfo = await this.processInformation(searchResults);
            
            // 4. Store in knowledge base
            this.updateKnowledgeBase(keywords, processedInfo);

            return processedInfo;
        } catch (error) {
            console.error('Learning error:', error);
            throw error;
        }
    }

    extractKeywords(query) {
        const tokenizer = new natural.WordTokenizer();
        const tokens = tokenizer.tokenize(query.toLowerCase());
        const stopwords = natural.stopwords;
        return tokens.filter(token => !stopwords.includes(token));
    }

    async searchMultipleSources(keywords) {
        const results = [];
        
        // Search Google Custom Search API
        try {
            const googleResults = await axios.get(
                `https://www.googleapis.com/customsearch/v1?key=${this.searchAPIs.google}&cx=YOUR_SEARCH_ENGINE_ID&q=${keywords.join('+')}`
            );
            results.push(...googleResults.data.items);
        } catch (error) {
            console.warn('Google search failed:', error.message);
        }

        // Get insights from OpenAI
        try {
            const openaiResponse = await this.openai.createCompletion({
                model: "gpt-3.5-turbo",
                messages: [{
                    role: "system",
                    content: "Analyze and provide insights about: " + keywords.join(' ')
                }]
            });
            results.push({ source: 'openai', content: openaiResponse.data.choices[0].text });
        } catch (error) {
            console.warn('OpenAI query failed:', error.message);
        }

        return results;
    }

    async processInformation(searchResults) {
        const processedInfo = {
            summary: '',
            sources: [],
            confidence: 0,
            timestamp: Date.now()
        };

        for (const result of searchResults) {
            try {
                // Fetch and parse webpage content
                if (result.link) {
                    const response = await axios.get(result.link);
                    const $ = cheerio.load(response.data);
                    const pageText = $('p').text();
                    
                    // Add to processed info
                    processedInfo.sources.push({
                        url: result.link,
                        title: result.title,
                        snippet: pageText.substring(0, 500)
                    });
                }
            } catch (error) {
                console.warn(`Failed to process source ${result.link}:`, error.message);
            }
        }

        // Use OpenAI to generate a summary
        try {
            const summary = await this.openai.createCompletion({
                model: "gpt-3.5-turbo",
                messages: [{
                    role: "system",
                    content: "Create a concise summary of this information: " + 
                            processedInfo.sources.map(s => s.snippet).join('\n')
                }]
            });
            processedInfo.summary = summary.data.choices[0].text;
        } catch (error) {
            console.warn('Summary generation failed:', error.message);
        }

        return processedInfo;
    }

    updateKnowledgeBase(keywords, processedInfo) {
        const key = keywords.join('_');
        this.knowledgeBase.set(key, {
            ...processedInfo,
            accessCount: 0,
            lastAccessed: Date.now()
        });
    }

    async focusOnMarketing() {
        // Filter and prioritize marketing-related content
        const marketingKeywords = [
            'marketing', 'advertising', 'branding', 'social media',
            'SEO', 'content strategy', 'lead generation', 'conversion',
            'customer acquisition', 'market research', 'analytics'
        ];

        // Update learning priorities
        this.learningPriorities = {
            domains: [
                'blog.hubspot.com',
                'moz.com',
                'searchenginejournal.com',
                'marketingweek.com'
            ],
            topics: marketingKeywords
        };
    }
}

module.exports = new LearningManager(); 