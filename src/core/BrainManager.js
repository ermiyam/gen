const fs = require('fs').promises;
const path = require('path');
const natural = require('natural');

class BrainManager {
    constructor() {
        this.shortTermMemory = new Map();
        this.longTermMemory = new Map();
        this.connections = new Map();
        this.tokenizer = new natural.WordTokenizer();
        this.tfidf = new natural.TfIdf();
        
        // Memory limits
        this.MAX_SHORT_TERM = 1000;
        this.MAX_LONG_TERM = 1000000;
    }

    async initialize() {
        try {
            await this.loadMemory();
            this.startMemoryManagement();
            console.log('Brain initialized successfully');
        } catch (error) {
            console.error('Brain initialization error:', error);
            // Create fresh memory if loading fails
            await this.saveMemory();
        }
    }

    async loadMemory() {
        const brainPath = path.join(__dirname, '../../brain/memory.json');
        try {
            const data = await fs.readFile(brainPath, 'utf8');
            const memory = JSON.parse(data);
            this.longTermMemory = new Map(memory.longTerm);
            this.connections = new Map(memory.connections);
        } catch (error) {
            console.log('No existing memory found, starting fresh');
        }
    }

    async saveMemory() {
        const brainPath = path.join(__dirname, '../../brain/memory.json');
        const memory = {
            longTerm: Array.from(this.longTermMemory.entries()),
            connections: Array.from(this.connections.entries())
        };
        await fs.writeFile(brainPath, JSON.stringify(memory, null, 2));
    }

    startMemoryManagement() {
        // Consolidate short-term to long-term memory every minute
        setInterval(() => this.consolidateMemory(), 60000);
        
        // Save memory to disk every 5 minutes
        setInterval(() => this.saveMemory(), 300000);
        
        // Clean up old memories every hour
        setInterval(() => this.cleanupMemory(), 3600000);
    }

    async learn(input, response) {
        const tokens = this.tokenizer.tokenize(input.toLowerCase());
        const timestamp = Date.now();

        // Add to short-term memory
        this.shortTermMemory.set(timestamp, {
            input,
            response,
            tokens,
            accessed: 1,
            created: timestamp
        });

        // Update TF-IDF
        this.tfidf.addDocument(tokens);

        // Create connections
        this.updateConnections(tokens);

        // Manage memory limits
        if (this.shortTermMemory.size > this.MAX_SHORT_TERM) {
            await this.consolidateMemory();
        }

        return true;
    }

    updateConnections(tokens) {
        tokens.forEach((token, i) => {
            if (!this.connections.has(token)) {
                this.connections.set(token, new Set());
            }
            
            // Connect with nearby words
            for (let j = Math.max(0, i - 2); j <= Math.min(tokens.length - 1, i + 2); j++) {
                if (i !== j) {
                    this.connections.get(token).add(tokens[j]);
                }
            }
        });
    }

    async consolidateMemory() {
        for (const [timestamp, memory] of this.shortTermMemory) {
            if (memory.accessed > 2 || Date.now() - timestamp > 3600000) {
                this.longTermMemory.set(timestamp, {
                    ...memory,
                    consolidated: Date.now()
                });
            }
        }

        // Clear short-term memory
        this.shortTermMemory.clear();

        // Manage long-term memory size
        if (this.longTermMemory.size > this.MAX_LONG_TERM) {
            this.cleanupMemory();
        }
    }

    cleanupMemory() {
        const memories = Array.from(this.longTermMemory.entries())
            .sort(([, a], [, b]) => b.accessed - a.accessed)
            .slice(0, this.MAX_LONG_TERM);

        this.longTermMemory = new Map(memories);
    }

    getStats() {
        return {
            shortTermSize: this.shortTermMemory.size,
            longTermSize: this.longTermMemory.size,
            connections: this.connections.size,
            lastSaved: this.lastSaved
        };
    }
}

module.exports = { BrainManager }; 