const EventEmitter = require('events');

class MonitoringSystem extends EventEmitter {
    constructor() {
        super();
        this.metrics = new Map();
        this.alerts = new Set();
        this.thresholds = {
            cpu: 80, // 80% CPU usage
            memory: 85, // 85% memory usage
            responseTime: 1000 // 1 second
        };
    }

    async monitor() {
        setInterval(async () => {
            const metrics = await this.gatherMetrics();
            this.checkThresholds(metrics);
            this.emit('metrics', metrics);
        }, 5000);
    }

    async gatherMetrics() {
        const metrics = {
            timestamp: Date.now(),
            cpu: process.cpuUsage(),
            memory: process.memoryUsage(),
            uptime: process.uptime()
        };

        this.metrics.set(metrics.timestamp, metrics);
        return metrics;
    }

    checkThresholds(metrics) {
        if (metrics.cpu.user > this.thresholds.cpu) {
            this.createAlert('High CPU Usage', metrics);
        }

        if ((metrics.memory.heapUsed / metrics.memory.heapTotal) * 100 > this.thresholds.memory) {
            this.createAlert('High Memory Usage', metrics);
        }
    }

    createAlert(type, data) {
        const alert = {
            type,
            data,
            timestamp: Date.now()
        };

        this.alerts.add(alert);
        this.emit('alert', alert);
    }
}

module.exports = new MonitoringSystem(); 