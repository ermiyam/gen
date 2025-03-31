import fetch from 'node-fetch';
import readline from 'readline';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function addVideo() {
    try {
        // Ask for YouTube URL
        const videoUrl = await new Promise(resolve => {
            rl.question('Enter YouTube video URL: ', (answer) => {
                resolve(answer.trim());
            });
        });

        console.log('🎥 Learning from video...');
        
        const response = await fetch('http://localhost:3000/learn/video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ videoUrl })
        });

        const result = await response.json();
        
        if (result.success) {
            console.log('✅ Successfully learned from video!');
            console.log('Title:', result.title);
        } else {
            console.log('❌ Failed to learn from video:', result.error);
        }

    } catch (error) {
        console.error('❌ Error:', error.message);
    } finally {
        rl.close();
    }
}

addVideo();
