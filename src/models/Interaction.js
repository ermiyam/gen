const mongoose = require('mongoose');

const InteractionSchema = new mongoose.Schema({
    type: { type: String, required: true },
    input: { type: String, required: true },
    response: { type: mongoose.Schema.Types.Mixed },
    timestamp: { type: Date, default: Date.now },
    success: { type: Boolean, default: true },
    processingTime: { type: Number },
    confidence: { type: Number }
});

module.exports = mongoose.model('Interaction', InteractionSchema); 