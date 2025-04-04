const LearningSystem = require('./LearningSystem');

class LearningManager {
    constructor() {
        this.learningSystem = LearningSystem;
        this.status = {
            isActive: false,
            lastUpdate: null,
            currentFocus: [],
            statistics: {
                topicsLearned: 0,
                knowledgeSize: 0
            }
        };
    }

    async configure(config) {
        try {
            // Update learning configuration
            this.learningSystem.learningConfig = {
                ...this.learningSystem.learningConfig,
                ...config
            };

            // Restart learning process with new config
            await this.restartLearning();

            return {
                success: true,
                message: 'Learning configuration updated',
                currentConfig: this.learningSystem.learningConfig
            };
        } catch (error) {
            console.error('Configuration error:', error);
            throw error;
        }
    }

    async getStatus() {
        return {
            ...this.status,
            knowledgeSize: this.learningSystem.knowledgeBase.size,
            lastUpdate: await this.learningSystem.getLastUpdateTime()
        };
    }
}

module.exports = new LearningManager(); 