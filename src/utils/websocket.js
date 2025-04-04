function setupWebSocket(wss, ai) {
    wss.on('connection', (ws) => {
        console.log('New client connected');
        
        // Send initial stats
        sendStats(ws, ai);

        // Set up interval to send stats updates
        const statsInterval = setInterval(() => {
            sendStats(ws, ai);
        }, 1000);

        ws.on('close', () => {
            clearInterval(statsInterval);
            console.log('Client disconnected');
        });

        ws.on('error', (error) => {
            console.error('WebSocket error:', error);
        });
    });
}

function sendStats(ws, ai) {
    try {
        const stats = ai.getStats();
        ws.send(JSON.stringify({
            type: 'stats',
            data: stats,
            timestamp: Date.now()
        }));
    } catch (error) {
        console.error('Error sending stats:', error);
    }
}

module.exports = { setupWebSocket }; 