const mongoose = require('mongoose');

const AnalyticsSchema = new mongoose.Schema({
    category: { type: String, required: true },
    metric: { type: String, required: true },
    value: { type: mongoose.Schema.Types.Mixed, required: true },
    timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Analytics', AnalyticsSchema); 