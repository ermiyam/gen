const express = require('express');
const cors = require('cors');
const compression = require('compression');
const { createServer } = require('http');
const { Server } = require('socket.io');
const fs = require('fs');
require('dotenv').config();

class MarketingAI {
    constructor() {
        this.app = express();
        this.httpServer = createServer(this.app);
        this.io = new Server(this.httpServer, {
            cors: { origin: "*" }
        });

        // Initialize stats
        this.stats = {
            conceptsLearned: 0,
            insightsGenerated: 0,
            patternsIdentified: 0,
            connectionsMade: 0
        };

        this.setupServer();
        this.startLearning();
    }

    setupServer() {
        this.app.use(cors());
        this.app.use(compression());
        this.app.use(express.json());
        this.app.use(express.static('public'));

        // Health check route
        this.app.get('/', (req, res) => {
            res.json({ status: 'AI Marketing Server Running' });
        });

        this.setupWebSocket();
    }

    setupWebSocket() {
        this.io.on('connection', (socket) => {
            console.log('Client connected');
            socket.emit('stats-update', this.stats);
        });
    }

    startLearning() {
        setInterval(() => {
            this.stats.conceptsLearned++;
            this.io.emit('stats-update', this.stats);
        }, 2000);

        setInterval(() => {
            this.stats.insightsGenerated++;
            this.io.emit('stats-update', this.stats);
        }, 3000);
    }

    start() {
        const port = process.env.PORT || 3000;
        this.httpServer.listen(port, '0.0.0.0', () => {
            console.log(`Server running on port ${port}`);
        });
    }
}

const server = new MarketingAI();
server.start(); 