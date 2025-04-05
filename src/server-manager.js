const express = require('express');
const { exec } = require('child_process');
const path = require('path');
const chokidar = require('chokidar');
const git = require('simple-git');
const WebSocket = require('ws');

class ServerManager {
    constructor() {
        this.app = express();
        this.wss = new WebSocket.Server({ port: 8080 });
        this.clients = new Set();
        this.repoPath = process.env.REPO_PATH || './';
        
        this.setupGitWatcher();
        this.setupWebSocket();
        this.startAutoSync();
    }

    setupGitWatcher() {
        // Watch for file changes
        chokidar.watch(['src/**/*', 'public/**/*'], {
            ignored: /(^|[\/\\])\../,
            persistent: true
        }).on('all', (event, path) => {
            this.handleFileChange(event, path);
        });
    }

    setupWebSocket() {
        this.wss.on('connection', (ws) => {
            this.clients.add(ws);
            console.log('Client connected');

            ws.on('message', async (message) => {
                const data = JSON.parse(message);
                if (data.type === 'code-update') {
                    await this.updateCode(data.files);
                }
            });

            ws.on('close', () => {
                this.clients.delete(ws);
                console.log('Client disconnected');
            });
        });
    }

    async handleFileChange(event, filePath) {
        console.log(`File ${event}: ${filePath}`);
        
        try {
            // Commit changes
            await git(this.repoPath)
                .add(filePath)
                .commit(`Auto-update: ${event} in ${filePath}`);

            // Push changes
            await git(this.repoPath).push();

            // Notify all clients
            this.broadcastUpdate(filePath);
        } catch (error) {
            console.error('Git sync error:', error);
        }
    }

    broadcastUpdate(filePath) {
        const message = JSON.stringify({
            type: 'file-update',
            path: filePath,
            timestamp: Date.now()
        });

        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
    }

    startAutoSync() {
        // Pull changes every 30 seconds
        setInterval(async () => {
            try {
                await git(this.repoPath).pull();
                console.log('Auto-sync completed');
            } catch (error) {
                console.error('Auto-sync error:', error);
            }
        }, 30000);
    }

    start() {
        const port = process.env.PORT || 3001;
        this.app.listen(port, () => {
            console.log(`\n🔄 Server Manager running on port ${port}`);
            console.log(`📡 WebSocket server running on port 8080`);
            console.log(`👀 Watching for file changes...`);
        });
    }
}

const manager = new ServerManager();
manager.start(); 