module.exports = {
    learningThresholds: {
        minConfidence: 0.7,
        maxSourceAge: 30 * 24 * 60 * 60 * 1000, // 30 days in milliseconds
        minSourceCount: 3
    },
    
    marketingFocus: {
        enabled: false, // Will be enabled later
        domains: [
            'blog.hubspot.com',
            'moz.com',
            'searchenginejournal.com',
            'marketingweek.com'
        ],
        topics: [
            'digital marketing',
            'content strategy',
            'SEO',
            'social media marketing',
            'email marketing',
            'PPC',
            'marketing analytics'
        ]
    }
}; 