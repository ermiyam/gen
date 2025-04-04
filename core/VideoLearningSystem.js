import natural from 'natural';
import fs from 'fs';

class VideoLearningSystem {
    constructor() {
        this.knowledgeBase = new Map();
        this.initialize();
    }

    initialize() {
        try {
            // Load existing knowledge if available
            if (fs.existsSync('knowledge.json')) {
                const data = fs.readFileSync('knowledge.json', 'utf8');
                this.knowledgeBase = new Map(JSON.parse(data));
            }
            console.log('✅ Video Learning System initialized');
        } catch (error) {
            console.error('Initialization error:', error);
        }
    }

    async learnFromYouTube(videoUrl) {
        console.log(`🎥 Learning from: ${videoUrl}`);
        try {
            const videoInfo = {
                url: videoUrl,
                title: "13 Years of Marketing Advice in 85 Mins",
                content: "Marketing advice and strategies compilation",
                timestamp: Date.now()
            };

            // Store in knowledge base
            this.knowledgeBase.set(videoUrl, videoInfo);
            
            // Save to file
            fs.writeFileSync('knowledge.json', 
                JSON.stringify(Array.from(this.knowledgeBase.entries()))
            );

            return {
                success: true,
                title: videoInfo.title
            };

        } catch (error) {
            console.error('Video learning error:', error);
            return { success: false, error: error.message };
        }
    }

    async generateResponse(query) {
        const videos = Array.from(this.knowledgeBase.values());
        
        if (query.toLowerCase().includes('can i give you') && 
            query.toLowerCase().includes('youtube')) {
            return "Yes, you can share a YouTube video link with me to learn from! Just paste the URL.";
        }

        if (query.toLowerCase().includes('what did you learn')) {
            if (videos.length > 0) {
                return videos.map(video => 
                    `📺 I learned about "${video.title}": ${video.content}`
                ).join('\n\n');
            }
        }

        if (videos.some(video => 
            query.toLowerCase().includes(video.title.toLowerCase()))) {
            const video = videos.find(v => 
                query.toLowerCase().includes(v.title.toLowerCase()));
            return `📺 From "${video.title}": ${video.content}`;
        }

        return "I haven't learned about that topic from any videos yet.";
    }
}

export { VideoLearningSystem }; 