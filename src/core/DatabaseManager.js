const mongoose = require('mongoose');

class DatabaseManager {
    constructor() {
        this.isConnected = false;
    }

    async connect() {
        try {
            // Local MongoDB connection - no API key needed for local development
            await mongoose.connect('mongodb://127.0.0.1:27017/ai_learning', {
                useNewUrlParser: true,
                useUnifiedTopology: true,
                serverSelectionTimeoutMS: 5000
            });
            
            this.isConnected = true;
            console.log('Connected to MongoDB successfully');
        } catch (error) {
            console.error('MongoDB connection error:', error);
            // Instead of continuing without MongoDB, we'll exit
            process.exit(1);
        }
    }

    async disconnect() {
        if (this.isConnected) {
            await mongoose.disconnect();
            this.isConnected = false;
            console.log('Disconnected from MongoDB');
        }
    }
}

module.exports = new DatabaseManager(); 