import { readFile, writeFile } from 'fs/promises';
import { existsSync, mkdirSync } from 'fs';
import { execSync } from 'child_process';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🔧 Starting system preparation...');

async function main() {
    try {
        // Kill any existing processes on port 3000
        try {
            const processes = execSync('lsof -i :3000 -t').toString().trim().split('\n');
            processes.forEach(pid => {
                try {
                    execSync(`kill -9 ${pid}`);
                } catch (e) {
                    // Ignore if process already gone
                }
            });
        } catch (e) {
            // No processes found on port 3000
        }

        // Read package.json
        const packagePath = join(__dirname, '..', 'package.json');
        const packageData = JSON.parse(await readFile(packagePath));

        // Ensure type: "module" is set
        if (!packageData.type || packageData.type !== 'module') {
            packageData.type = 'module';
            await writeFile(packagePath, JSON.stringify(packageData, null, 2));
        }

        // Create knowledge directory if it doesn't exist
        const knowledgeDir = join(__dirname, '..', 'knowledge');
        if (!existsSync(knowledgeDir)) {
            mkdirSync(knowledgeDir);
        }

        // Initialize empty knowledge file if it doesn't exist
        const knowledgeFile = join(knowledgeDir, 'video_knowledge.json');
        if (!existsSync(knowledgeFile)) {
            await writeFile(knowledgeFile, '[]');
        }

        console.log('✅ System prepared successfully');
    } catch (error) {
        console.error('❌ Error during system preparation:', error);
        process.exit(1);
    }
}

main(); 