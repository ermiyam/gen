const axios = require('axios');
const cheerio = require('cheerio');
const puppeteer = require('puppeteer');

class EnhancedSearchSystem {
    constructor() {
        this.browser = null;
        this.searchCache = new Map();
        this.initializeBrowser();
    }

    async initializeBrowser() {
        this.browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox']
        });
    }

    async searchAll(query) {
        const cacheKey = query.toLowerCase();
        if (this.searchCache.has(cacheKey)) {
            return this.searchCache.get(cacheKey);
        }

        const results = await Promise.all([
            this.searchGoogle(query),
            this.searchYouTube(query),
            this.searchStackOverflow(query),
            this.searchTwitter(query),
            this.searchReddit(query)
        ]);

        const combinedResults = this.combineResults(results);
        this.searchCache.set(cacheKey, combinedResults);
        return combinedResults;
    }

    async searchGoogle(query) {
        // ... (existing Google search code) ...
    }

    async searchYouTube(query) {
        try {
            const response = await axios.get(
                `https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(query)}&key=${process.env.YOUTUBE_API_KEY}`
            );
            return response.data.items.map(item => ({
                type: 'video',
                title: item.snippet.title,
                description: item.snippet.description,
                url: `https://youtube.com/watch?v=${item.id.videoId}`,
                source: 'YouTube'
            }));
        } catch (error) {
            console.error('YouTube search error:', error);
            return [];
        }
    }

    // ... (other search methods) ...
} 