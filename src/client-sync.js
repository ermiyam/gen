const WebSocket = require('ws');
const { exec } = require('child_process');
const git = require('simple-git');
const path = require('path');
const fs = require('fs');

class ClientSync {
    constructor() {
        this.ws = null;
        this.serverUrl = process.env.SERVER_URL || 'ws://your-server-ip:8080';
        this.repoPath = process.env.REPO_PATH || './';
        
        this.connect();
        this.setupLocalWatcher();
    }

    connect() {
        this.ws = new WebSocket(this.serverUrl);

        this.ws.on('open', () => {
            console.log('Connected to sync server');
        });

        this.ws.on('message', async (data) => {
            const message = JSON.parse(data);
            if (message.type === 'file-update') {
                await this.syncChanges();
            }
        });

        this.ws.on('close', () => {
            console.log('Disconnected from server, reconnecting...');
            setTimeout(() => this.connect(), 5000);
        });
    }

    setupLocalWatcher() {
        fs.watch(this.repoPath, { recursive: true }, async (event, filename) => {
            if (filename) {
                await this.handleLocalChange(filename);
            }
        });
    }

    async handleLocalChange(filename) {
        try {
            const status = await git(this.repoPath).status();
            if (status.modified.includes(filename)) {
                this.ws.send(JSON.stringify({
                    type: 'code-update',
                    files: [filename]
                }));
            }
        } catch (error) {
            console.error('Local change handling error:', error);
        }
    }

    async syncChanges() {
        try {
            await git(this.repoPath).pull();
            console.log('Code synchronized');
            
            // Restart the AI server if needed
            this.restartAIServer();
        } catch (error) {
            console.error('Sync error:', error);
        }
    }

    restartAIServer() {
        exec('npm restart', (error) => {
            if (error) {
                console.error('Server restart error:', error);
                return;
            }
            console.log('AI server restarted');
        });
    }
}

const client = new ClientSync(); 