const { Worker } = require('worker_threads');
const path = require('path');
const { BrainManager } = require('./BrainManager');
const { LearningSystem } = require('./LearningSystem');

class CoreAI {
    constructor() {
        this.name = "Gen";
        this.version = "1.0.0";
        this.brain = new BrainManager();
        this.learning = new LearningSystem();
        this.workers = new Map();
        this.initialize();
    }

    async initialize() {
        console.log('Initializing AI Core Systems...');
        await this.brain.initialize();
        await this.initializeWorkers();
        this.startContinuousLearning();
        console.log('AI Core Systems Ready');
    }

    async initializeWorkers() {
        const workerTypes = ['pattern', 'learning', 'analysis', 'neural'];
        
        for (const type of workerTypes) {
            const worker = new Worker(path.join(__dirname, `../workers/${type}Worker.js`));
            this.setupWorkerHandlers(worker, type);
            this.workers.set(type, worker);
        }
    }

    setupWorkerHandlers(worker, type) {
        worker.on('message', (data) => {
            this.handleWorkerMessage(type, data);
        });

        worker.on('error', (error) => {
            console.error(`Worker ${type} error:`, error);
            this.restartWorker(type);
        });
    }

    async think(input) {
        const startTime = process.hrtime();

        // Distribute work to workers
        const results = await Promise.all([
            this.processWithWorker('pattern', input),
            this.processWithWorker('learning', input),
            this.processWithWorker('analysis', input),
            this.processWithWorker('neural', input)
        ]);

        // Synthesize results
        const response = this.synthesizeResults(results);

        // Calculate processing time
        const [seconds, nanoseconds] = process.hrtime(startTime);
        const processingTime = seconds * 1000 + nanoseconds / 1000000;

        // Update brain with new knowledge
        await this.brain.learn(input, response);

        return {
            thought: response,
            stats: this.getStats(),
            processingTime: `${processingTime.toFixed(2)}ms`
        };
    }

    processWithWorker(type, input) {
        return new Promise((resolve) => {
            const worker = this.workers.get(type);
            worker.postMessage({ input, timestamp: Date.now() });
            worker.once('message', resolve);
        });
    }

    synthesizeResults(results) {
        // Combine and weight results from different workers
        const synthesis = results.reduce((acc, result) => {
            if (result && result.confidence > 0.5) {
                acc.push(result.output);
            }
            return acc;
        }, []);

        return synthesis.join(' ');
    }

    startContinuousLearning() {
        setInterval(() => {
            this.learning.learn();
        }, 5000); // Learn every 5 seconds
    }

    getStats() {
        return {
            brain: this.brain.getStats(),
            learning: this.learning.getStats(),
            workers: Array.from(this.workers.keys()),
            uptime: process.uptime()
        };
    }
}

module.exports = { CoreAI }; 