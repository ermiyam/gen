// Define our core types for better organization
class MarketingConcept {
    constructor(name, description, examples, metrics, strategies) {
        this.name = name;
        this.description = description;
        this.examples = examples;
        this.metrics = metrics;
        this.strategies = strategies;
        this.relatedConcepts = new Set();
    }
}

class VideoContent {
    constructor(id, url, title, timestamp) {
        this.id = id;
        this.url = url;
        this.title = title;
        this.timestamp = timestamp;
        this.concepts = new Map();
        this.keyPoints = [];
        this.summary = '';
        this.categories = new Set();
        this.difficulty = 'intermediate';
        this.audienceType = ['marketers', 'business owners'];
    }
}

class ConversationContext {
    constructor() {
        this.currentTopic = null;
        this.currentVideo = null;
        this.context = [];
        this.lastInteraction = Date.now();
        this.topicDepth = 0;
        this.userPreferences = {
            detailLevel: 'detailed',
            focusAreas: new Set(),
            examplesRequested: true
        };
    }

    addMessage(role, content, metadata = {}) {
        this.context.push({
            role,
            content,
            timestamp: Date.now(),
            metadata
        });
        
        // Keep last 15 messages for context
        if (this.context.length > 15) {
            this.context.shift();
        }
    }

    updateTopicDepth(message) {
        if (message.includes('more') || message.includes('detail')) {
            this.topicDepth++;
        } else if (message.includes('summarize') || message.includes('briefly')) {
            this.topicDepth = 0;
        }
    }
}

export { MarketingConcept, VideoContent, ConversationContext }; 