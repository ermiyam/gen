const axios = require('axios');
const cheerio = require('cheerio');
const { Configuration, OpenAIApi } = require('openai');
const puppeteer = require('puppeteer');

class InternetExplorer {
    constructor() {
        this.sources = {
            searchEngines: [
                'google',
                'bing',
                'duckduckgo'
            ],
            academicSources: [
                'scholar.google.com',
                'arxiv.org',
                'researchgate.net'
            ],
            trustworthyDomains: [
                'wikipedia.org',
                '.edu',
                '.gov',
                'medium.com',
                'stackoverflow.com'
            ]
        };

        this.browser = null;
        this.initializeBrowser();
    }

    async initializeBrowser() {
        this.browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox']
        });
    }

    async search(query, depth = 'medium') {
        const searchResults = {
            directResults: [],
            relatedInfo: [],
            timestamp: Date.now(),
            confidence: 0
        };

        try {
            // Parallel search across multiple sources
            const [webResults, apiResults, academicResults] = await Promise.all([
                this.searchWeb(query),
                this.searchAPIs(query),
                this.searchAcademic(query)
            ]);

            // Combine and validate results
            const combinedResults = await this.validateAndCombineResults(
                webResults,
                apiResults,
                academicResults
            );

            searchResults.directResults = combinedResults.direct;
            searchResults.relatedInfo = combinedResults.related;
            searchResults.confidence = combinedResults.confidence;

            return searchResults;

        } catch (error) {
            console.error('Search error:', error);
            throw error;
        }
    }

    async searchWeb(query) {
        const results = [];

        // Create a browser page
        const page = await this.browser.newPage();
        
        try {
            // Search Google
            await page.goto(`https://www.google.com/search?q=${encodeURIComponent(query)}`);
            const googleResults = await page.evaluate(() => {
                const links = Array.from(document.querySelectorAll('.g'));
                return links.map(link => ({
                    title: link.querySelector('h3')?.textContent,
                    url: link.querySelector('a')?.href,
                    snippet: link.querySelector('.VwiC3b')?.textContent
                }));
            });
            results.push(...googleResults);

            // Search additional sources if needed
            if (results.length < 5) {
                await this.searchAlternativeSources(query, results);
            }

        } catch (error) {
            console.error('Web search error:', error);
        } finally {
            await page.close();
        }

        return results;
    }

    async extractContent(url) {
        try {
            const page = await this.browser.newPage();
            await page.goto(url, { waitUntil: 'networkidle0' });

            // Extract main content
            const content = await page.evaluate(() => {
                // Remove unwanted elements
                document.querySelectorAll('nav, footer, aside, ads').forEach(el => el.remove());

                // Get main content
                const main = document.querySelector('main, article, .content, #content');
                return main ? main.textContent : document.body.textContent;
            });

            await page.close();
            return this.cleanContent(content);

        } catch (error) {
            console.error('Content extraction error:', error);
            return null;
        }
    }

    async validateAndCombineResults(...allResults) {
        const combined = {
            direct: [],
            related: [],
            confidence: 0
        };

        // Flatten and deduplicate results
        const uniqueResults = new Map();
        allResults.flat().forEach(result => {
            if (result.url && !uniqueResults.has(result.url)) {
                uniqueResults.set(result.url, result);
            }
        });

        // Validate each result
        for (const [url, result] of uniqueResults) {
            const validation = await this.validateSource(url, result);
            
            if (validation.score > 0.7) {
                combined.direct.push({
                    ...result,
                    validationScore: validation.score,
                    lastVerified: Date.now()
                });
            } else if (validation.score > 0.4) {
                combined.related.push({
                    ...result,
                    validationScore: validation.score,
                    lastVerified: Date.now()
                });
            }
        }

        // Calculate overall confidence
        combined.confidence = this.calculateConfidence(combined.direct);

        return combined;
    }

    async validateSource(url, content) {
        const validation = {
            score: 0,
            factors: {}
        };

        // Check domain trustworthiness
        validation.factors.domainTrust = this.checkDomainTrust(url);

        // Check content quality
        validation.factors.contentQuality = await this.assessContentQuality(content);

        // Check for citations and references
        validation.factors.citations = await this.checkCitations(content);

        // Calculate final score
        validation.score = Object.values(validation.factors)
            .reduce((acc, val) => acc + val, 0) / Object.keys(validation.factors).length;

        return validation;
    }

    cleanContent(content) {
        return content
            .replace(/\s+/g, ' ')
            .replace(/[^\w\s.,!?-]/g, '')
            .trim();
    }

    calculateConfidence(results) {
        if (results.length === 0) return 0;

        const avgValidationScore = results.reduce((acc, r) => acc + r.validationScore, 0) / results.length;
        const sourceVariety = new Set(results.map(r => new URL(r.url).hostname)).size / results.length;

        return (avgValidationScore * 0.7 + sourceVariety * 0.3);
    }
}

module.exports = new InternetExplorer(); 