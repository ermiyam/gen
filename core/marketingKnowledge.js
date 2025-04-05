import { MarketingConcept } from './types.js';

class MarketingKnowledgeBase {
    constructor() {
        this.concepts = new Map();
        this.initializeKnowledge();
    }

    initializeKnowledge() {
        // Brand Building
        const brandBuilding = new MarketingConcept(
            'Brand Building',
            'The strategic process of creating and strengthening your brand identity and perception',
            [
                'Nike\'s "Just Do It" campaign consistency',
                'Apple\'s minimalist design philosophy',
                'Coca-Cola\'s emotional storytelling'
            ],
            [
                'Brand awareness',
                'Brand recall',
                'Brand sentiment',
                'Net Promoter Score (NPS)',
                'Social media mentions'
            ],
            [
                {
                    name: 'Visual Identity Development',
                    steps: [
                        'Define brand values and personality',
                        'Create consistent visual elements',
                        'Develop brand guidelines',
                        'Implement across all channels'
                    ]
                },
                {
                    name: 'Content Strategy',
                    steps: [
                        'Define brand voice',
                        'Create content pillars',
                        'Develop editorial calendar',
                        'Measure content performance'
                    ]
                }
            ]
        );

        // Social Media Marketing
        const socialMedia = new MarketingConcept(
            'Social Media Marketing',
            'Strategic use of social media platforms to connect with audiences and achieve marketing objectives',
            [
                'Wendy\'s Twitter personality',
                'GoPro\'s Instagram UGC strategy',
                'Glossier\'s community-first approach'
            ],
            [
                'Engagement rate',
                'Reach',
                'Click-through rate',
                'Conversion rate',
                'Share of voice'
            ],
            [
                {
                    name: 'Platform Optimization',
                    steps: [
                        'Identify target platforms',
                        'Optimize content format',
                        'Schedule for peak engagement',
                        'Monitor performance metrics'
                    ]
                },
                {
                    name: 'Community Management',
                    steps: [
                        'Develop response guidelines',
                        'Create engagement calendar',
                        'Monitor mentions and tags',
                        'Build relationships with followers'
                    ]
                }
            ]
        );

        // Add more concepts...
        this.concepts.set('brand_building', brandBuilding);
        this.concepts.set('social_media', socialMedia);
        
        // Establish relationships
        brandBuilding.relatedConcepts.add('social_media');
        socialMedia.relatedConcepts.add('brand_building');
    }

    getConcept(name) {
        return this.concepts.get(name);
    }

    getRelatedConcepts(name) {
        const concept = this.getConcept(name);
        return concept ? 
            Array.from(concept.relatedConcepts).map(c => this.getConcept(c)) 
            : [];
    }
}

export const marketingKnowledge = new MarketingKnowledgeBase(); 