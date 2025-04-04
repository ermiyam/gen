const axios = require('axios');
const cheerio = require('cheerio');
const { google } = require('googleapis');
const { YoutubeTranscript } = require('youtube-transcript-api');

class InternetFetcher {
    constructor() {
        this.sources = {
            google: {
                enabled: true,
                apiKey: process.env.GOOGLE_API_KEY,
                searchEngineId: process.env.GOOGLE_CUSTOM_SEARCH_ID
            },
            reddit: {
                enabled: true,
                clientId: process.env.REDDIT_CLIENT_ID,
                clientSecret: process.env.REDDIT_CLIENT_SECRET
            },
            twitter: {
                enabled: true,
                apiKey: process.env.TWITTER_API_KEY
            }
        };
        this.youtube = google.youtube({
            version: 'v3',
            auth: this.sources.google.apiKey
        });
        this.cache = new Map();
    }

    async searchInternet(query) {
        try {
            // Check cache first
            if (this.cache.has(query)) {
                return this.cache.get(query);
            }

            // Use DuckDuckGo API (doesn't require API key)
            const searchUrl = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json`;
            const response = await axios.get(searchUrl);

            const results = {
                success: true,
                totalResults: 0,
                results: []
            };

            if (response.data && response.data.RelatedTopics) {
                results.results = response.data.RelatedTopics
                    .filter(topic => topic.Text && topic.FirstURL)
                    .map(topic => ({
                        title: topic.Text,
                        url: topic.FirstURL,
                        summary: topic.Text,
                        source: 'DuckDuckGo'
                    }))
                    .slice(0, 5);

                results.totalResults = results.results.length;
            }

            // Cache the results
            this.cache.set(query, results);
            return results;

        } catch (error) {
            console.error('Search error:', error);
            return {
                success: false,
                error: 'Failed to search the internet',
                results: []
            };
        }
    }

    async scrapeWebSearch(query) {
        try {
            // Use DuckDuckGo as it doesn't require API keys
            const searchUrl = `https://duckduckgo.com/?q=${encodeURIComponent(query)}&format=json`;
            const response = await axios.get(searchUrl);
            
            const results = [];
            
            // Process each search result
            if (response.data && response.data.RelatedTopics) {
                for (const topic of response.data.RelatedTopics) {
                    if (topic.Text && topic.FirstURL) {
                        try {
                            const content = await this.scrapeWebsite(topic.FirstURL);
                            if (content) {
                                results.push({
                                    title: topic.Text,
                                    url: topic.FirstURL,
                                    content: content,
                                    source: 'web'
                                });
                            }
                        } catch (error) {
                            console.error(`Failed to scrape ${topic.FirstURL}:`, error);
                        }
                    }
                }
            }

            return results;

        } catch (error) {
            console.error('Web scraping error:', error);
            return [];
        }
    }

    async scrapeWebsite(url) {
        try {
            const response = await axios.get(url, {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                timeout: 5000
            });

            const $ = cheerio.load(response.data);

            // Remove unwanted elements
            $('script, style, nav, footer, header, aside').remove();

            // Get main content
            const content = $('main, article, .content, #content, body')
                .first()
                .text()
                .replace(/\s+/g, ' ')
                .trim();

            return content || null;

        } catch (error) {
            console.error('Website scraping error:', error);
            return null;
        }
    }

    processResults(results) {
        // Remove duplicates and invalid results
        const uniqueResults = results.filter(result => 
            result && result.content && result.content.length > 100
        );

        // Sort by content length (assuming longer content is more informative)
        uniqueResults.sort((a, b) => b.content.length - a.content.length);

        // Take top 5 results
        const topResults = uniqueResults.slice(0, 5);

        return {
            success: true,
            totalResults: topResults.length,
            results: topResults.map(result => ({
                title: result.title,
                url: result.url,
                summary: result.content.substring(0, 200) + '...',
                source: result.source
            }))
        };
    }

    async fetchYoutubeTranscript(videoId) {
        try {
            if (this.cache.has(`yt-${videoId}`)) {
                return this.cache.get(`yt-${videoId}`);
            }

            const transcript = await YoutubeTranscript.fetchTranscript(videoId);
            const text = transcript.map(item => item.text).join(' ');
            
            this.cache.set(`yt-${videoId}`, text);
            return text;
        } catch (error) {
            console.error('YouTube transcript error:', error);
            return null;
        }
    }

    async fetchWebContent(url) {
        try {
            if (this.cache.has(url)) {
                return this.cache.get(url);
            }

            const { data } = await axios.get(url);
            const $ = cheerio.load(data);
            const text = $('p, h1, h2, h3, h4, h5, h6')
                .map((_, el) => $(el).text())
                .get()
                .join(' ');

            this.cache.set(url, text);
            return text;
        } catch (error) {
            console.error('Web fetch error:', error);
            return null;
        }
    }

    extractYoutubeId(url) {
        const regex = /(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/;
        const match = url.match(regex);
        return match ? match[1] : null;
    }
}

module.exports = { InternetFetcher }; 