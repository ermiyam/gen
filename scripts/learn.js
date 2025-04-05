const fetch = require('node-fetch');

const videoUrl = 'https://youtu.be/reisEL_D7xc?si=iBbx65iIdG1Bw2Se';

async function learnVideo() {
    try {
        console.log('🎥 Starting to learn from video...');
        const response = await fetch('http://localhost:3000/learn/video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ videoUrl })
        });

        const result = await response.json();
        console.log('✅ Result:', result);
    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

learnVideo();
