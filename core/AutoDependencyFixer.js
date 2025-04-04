const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class AutoDependencyFixer {
    constructor() {
        this.requiredDependencies = new Set([
            'fs-extra',
            'express',
            'cors',
            'axios',
            'natural',
            'cheerio',
            'ws',
            'esprima',
            'escodegen',
            'acorn',
            'chokidar'
        ]);
        
        this.initialize();
    }

    async initialize() {
        try {
            await this.fixMissingDependencies();
            this.watchForDependencyIssues();
            console.log('🔧 Auto Dependency Fixer initialized');
        } catch (error) {
            console.error('Initialization error:', error);
            await this.emergencyFix();
        }
    }

    async fixMissingDependencies() {
        const packageJson = this.readPackageJson();
        const missing = this.findMissingDependencies(packageJson);
        
        if (missing.length > 0) {
            console.log('📦 Installing missing dependencies:', missing.join(', '));
            this.installDependencies(missing);
        }
    }

    readPackageJson() {
        try {
            return require(path.join(process.cwd(), 'package.json'));
        } catch (error) {
            console.log('Creating package.json...');
            this.initializePackageJson();
            return this.readPackageJson();
        }
    }

    initializePackageJson() {
        execSync('npm init -y');
        const packageJson = require(path.join(process.cwd(), 'package.json'));
        packageJson.scripts = {
            ...packageJson.scripts,
            "start": "node index.js",
            "dev": "nodemon index.js"
        };
        fs.writeFileSync(
            path.join(process.cwd(), 'package.json'),
            JSON.stringify(packageJson, null, 2)
        );
    }

    findMissingDependencies(packageJson) {
        const installed = new Set([
            ...Object.keys(packageJson.dependencies || {}),
            ...Object.keys(packageJson.devDependencies || {})
        ]);

        return Array.from(this.requiredDependencies)
            .filter(dep => !installed.has(dep));
    }

    installDependencies(dependencies) {
        try {
            execSync(`npm install ${dependencies.join(' ')}`, { stdio: 'inherit' });
            console.log('✅ Dependencies installed successfully');
        } catch (error) {
            console.error('Failed to install dependencies:', error);
            this.tryAlternativeInstall(dependencies);
        }
    }

    tryAlternativeInstall(dependencies) {
        try {
            // Try installing one by one
            dependencies.forEach(dep => {
                try {
                    execSync(`npm install ${dep}`, { stdio: 'inherit' });
                    console.log(`✅ Installed ${dep}`);
                } catch (error) {
                    console.error(`Failed to install ${dep}:`, error);
                }
            });
        } catch (error) {
            console.error('Alternative installation failed:', error);
        }
    }

    watchForDependencyIssues() {
        const watcher = require('chokidar').watch([
            'package.json',
            '**/*.js',
            '**/*.json'
        ], {
            ignored: /(node_modules|\.git)/,
            persistent: true
        });

        watcher.on('change', async (filepath) => {
            if (path.basename(filepath) === 'package.json') {
                await this.fixMissingDependencies();
            } else {
                await this.checkFileForDependencies(filepath);
            }
        });
    }

    async checkFileForDependencies(filepath) {
        try {
            const content = fs.readFileSync(filepath, 'utf8');
            const requiredModules = this.extractRequiredModules(content);
            
            if (requiredModules.length > 0) {
                const missing = requiredModules.filter(mod => {
                    try {
                        require.resolve(mod);
                        return false;
                    } catch (error) {
                        return true;
                    }
                });

                if (missing.length > 0) {
                    console.log(`📦 Installing modules required by ${filepath}:`, missing.join(', '));
                    this.installDependencies(missing);
                }
            }
        } catch (error) {
            console.error(`Error checking dependencies in ${filepath}:`, error);
        }
    }

    extractRequiredModules(content) {
        const requireRegex = /require\(['"]([^'"]+)['"]\)/g;
        const importRegex = /from\s+['"]([^'"]+)['"]/g;
        const modules = new Set();

        let match;
        while ((match = requireRegex.exec(content)) !== null) {
            if (!match[1].startsWith('.')) {
                modules.add(match[1]);
            }
        }
        while ((match = importRegex.exec(content)) !== null) {
            if (!match[1].startsWith('.')) {
                modules.add(match[1]);
            }
        }

        return Array.from(modules);
    }

    async emergencyFix() {
        console.log('🚨 Performing emergency fix...');
        
        try {
            // Clear node_modules and package-lock.json
            execSync('rm -rf node_modules package-lock.json');
            
            // Reinstall all dependencies
            await this.fixMissingDependencies();
            
            console.log('✅ Emergency fix completed');
        } catch (error) {
            console.error('Emergency fix failed:', error);
        }
    }
}

// Create and export the Auto Dependency Fixer
const autoDependencyFixer = new AutoDependencyFixer();
module.exports = autoDependencyFixer; 