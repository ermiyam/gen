const express = require('express');
const cors = require('cors');

class BasicAI {
    constructor() {
        this.responses = new Map([
            ['hello', 'Hello! How can I help you with marketing today?'],
            ['help', 'I can help you with marketing strategies and content creation.'],
            ['marketing', 'Let me suggest some marketing strategies for you...']
        ]);
    }

    getResponse(message) {
        const lowerMessage = message.toLowerCase();
        return this.responses.get(lowerMessage) || 'I understand. How else can I help you?';
    }
}

// Only if this works, update index.js to:
const updatedIndex = `
const express = require('express');
const cors = require('cors');
const { BasicAI } = require('./core/BasicAI');

const app = express();
const ai = new BasicAI();

app.use(cors());
app.use(express.json());

app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

app.post('/chat', (req, res) => {
    try {
        const { message } = req.body;
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }
        const response = ai.getResponse(message);
        res.json({ response });
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(\`Server running on port \${PORT}\`);
});
`;

Let me know if this basic version starts without crashing, and then we can add more features gradually. The key is to:

1. Start with a minimal working version
2. Add features one at a time
3. Test each addition
4. Keep what works

Would you like me to:
1. Show how to add more features safely?
2. Create a testing system?
3. Add error recovery?
4. Include monitoring? 