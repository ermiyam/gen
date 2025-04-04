const fs = require('fs');
const { execSync } = require('child_process');
const path = require('path');

class InstantFixer {
    constructor() {
        this.fixed = false;
        this.fix();
    }

    async fix() {
        console.log('🔧 Starting instant fix...');
        
        try {
            // Step 1: Fix package.json
            this.fixPackageJson();

            // Step 2: Clear problematic files
            this.clearProblematicFiles();

            // Step 3: Install dependencies
            this.installDependencies();

            // Step 4: Verify installation
            this.verifyInstallation();

            console.log('✅ System fixed successfully!');
            this.fixed = true;

        } catch (error) {
            console.error('❌ Fix failed:', error);
            this.emergencyFix();
        }
    }

    fixPackageJson() {
        const packagePath = path.join(process.cwd(), 'package.json');
        const packageContent = {
            name: "ai-project",
            version: "1.0.0",
            main: "index.js",
            scripts: {
                "start": "node index.js",
                "dev": "nodemon index.js"
            },
            dependencies: {
                "express": "^4.17.1",
                "cors": "^2.8.5",
                "axios": "^0.21.1",
                "natural": "^5.1.13",
                "fs-extra": "^10.0.0",
                "chokidar": "^3.5.2",
                "nodemon": "^2.0.7"
            }
        };

        fs.writeFileSync(packagePath, JSON.stringify(packageContent, null, 2));
        console.log('✅ package.json fixed');
    }

    clearProblematicFiles() {
        const filesToRemove = [
            'node_modules',
            'package-lock.json',
            'yarn.lock'
        ];

        filesToRemove.forEach(file => {
            try {
                if (fs.existsSync(file)) {
                    if (fs.lstatSync(file).isDirectory()) {
                        fs.rmSync(file, { recursive: true, force: true });
                    } else {
                        fs.unlinkSync(file);
                    }
                }
            } catch (error) {
                console.error(`Error removing ${file}:`, error);
            }
        });

        console.log('✅ Problematic files cleared');
    }

    installDependencies() {
        try {
            console.log('📦 Installing dependencies...');
            execSync('npm install', { stdio: 'inherit' });
        } catch (error) {
            console.log('⚠️ Normal install failed, trying alternative...');
            this.alternativeInstall();
        }
    }

    alternativeInstall() {
        try {
            // Clear npm cache first
            execSync('npm cache clean --force', { stdio: 'inherit' });
            
            // Install dependencies one by one
            const dependencies = [
                'express',
                'cors',
                'axios',
                'natural',
                'fs-extra',
                'chokidar',
                'nodemon'
            ];

            dependencies.forEach(dep => {
                try {
                    execSync(`npm install ${dep}`, { stdio: 'inherit' });
                    console.log(`✅ Installed ${dep}`);
                } catch (error) {
                    console.error(`Failed to install ${dep}:`, error);
                }
            });
        } catch (error) {
            throw new Error('Alternative install failed: ' + error.message);
        }
    }

    verifyInstallation() {
        const requiredFiles = [
            'node_modules',
            'package.json',
            'package-lock.json'
        ];

        requiredFiles.forEach(file => {
            if (!fs.existsSync(file)) {
                throw new Error(`Missing required file: ${file}`);
            }
        });

        console.log('✅ Installation verified');
    }

    emergencyFix() {
        console.log('🚨 Performing emergency fix...');
        
        try {
            // Remove everything and start fresh
            this.clearProblematicFiles();
            
            // Create minimal package.json
            this.fixPackageJson();
            
            // Try installing only essential dependencies
            execSync('npm install express cors', { stdio: 'inherit' });
            
            console.log('✅ Emergency fix completed');
        } catch (error) {
            console.error('💥 Emergency fix failed. Please try:');
            console.log(`
1. Delete the project folder
2. Create a new folder
3. Copy your source files (excluding node_modules)
4. Run: npm init -y
5. Run: npm install
            `);
        }
    }
}

// Create and run the fixer
new InstantFixer(); 