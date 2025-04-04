const { parentPort } = require('worker_threads');

class PatternAnalyzer {
    analyze(input) {
        const patterns = this.findPatterns(input);
        return {
            type: 'pattern',
            patterns,
            confidence: this.calculateConfidence(patterns),
            timestamp: Date.now()
        };
    }

    findPatterns(input) {
        const words = input.toLowerCase().split(/\W+/);
        const patterns = new Map();

        // Find repeating word patterns
        for (let i = 0; i < words.length - 1; i++) {
            const pattern = `${words[i]} ${words[i + 1]}`;
            patterns.set(pattern, (patterns.get(pattern) || 0) + 1);
        }

        return Array.from(patterns.entries())
            .filter(([, count]) => count > 1);
    }

    calculateConfidence(patterns) {
        return Math.min(1, patterns.length / 10);
    }
}

const analyzer = new PatternAnalyzer();

parentPort.on('message', data => {
    const result = analyzer.analyze(data.input);
    parentPort.postMessage(result);
}); 