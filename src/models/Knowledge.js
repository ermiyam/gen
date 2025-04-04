const mongoose = require('mongoose');

const KnowledgeSchema = new mongoose.Schema({
    topic: { type: String, required: true, index: true },
    content: { type: mongoose.Schema.Types.Mixed, required: true },
    confidence: { type: Number, required: true, min: 0, max: 1 },
    sources: [{ type: String }],
    lastUpdated: { type: Date, default: Date.now },
    category: { type: String, index: true },
    usage: { type: Number, default: 0 }
});

module.exports = mongoose.model('Knowledge', KnowledgeSchema); 