let fse;
try {
    fse = require('fs-extra');
} catch (error) {
    console.error('fs-extra not found, installing...');
    require('child_process').execSync('npm install fs-extra --save', { stdio: 'inherit' });
    fse = require('fs-extra');
}

const path = require('path');
const AISystem = require('./AISystem');
const { LearningSystem } = require('./LearningSystem');
const { ErrorHandler } = require('./ErrorHandler');

class SystemManager {
    constructor() {
        this.aiSystem = new AISystem();
        this.learningSystem = new LearningSystem();
        this.errorHandler = new ErrorHandler();
        this.status = {
            isReady: false,
            lastError: null,
            startTime: Date.now()
        };

        this.initialize();
    }

    async initialize() {
        try {
            await this.verifyDependencies();
            await this.initializeSystems();
            this.status.isReady = true;
            console.log('✅ System Manager initialized successfully');
        } catch (error) {
            this.handleError(error);
        }
    }

    async verifyDependencies() {
        const requiredDeps = [
            'express',
            'cors',
            'fs-extra',
            'natural',
            'axios',
            'path'
        ];

        for (const dep of requiredDeps) {
            try {
                require(dep);
            } catch (error) {
                console.log(`Installing missing dependency: ${dep}`);
                require('child_process').execSync(`npm install ${dep} --save`, { stdio: 'inherit' });
            }
        }
    }

    async initializeSystems() {
        await Promise.all([
            this.aiSystem.initialize(),
            this.learningSystem.initialize()
        ]);
    }

    async processMessage(message) {
        try {
            const response = await this.aiSystem.processMessage(message);
            await this.learningSystem.learn(message, response);
            return response;
        } catch (error) {
            this.handleError(error);
            return "I apologize, but I encountered an error. Please try again.";
        }
    }

    handleError(error) {
        this.status.lastError = {
            message: error.message,
            timestamp: Date.now()
        };
        return this.errorHandler.handle(error);
    }

    getStatus() {
        return {
            status: this.status.isReady ? 'ready' : 'initializing',
            uptime: Date.now() - this.status.startTime,
            lastError: this.status.lastError
        };
    }

    getDetailedStatus() {
        return {
            ...this.getStatus(),
            ai: this.aiSystem.getStatus(),
            learning: this.learningSystem.getStatus(),
            errors: this.errorHandler.getStatus()
        };
    }
}

module.exports = { SystemManager }; 