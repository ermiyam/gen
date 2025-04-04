const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const chokidar = require('chokidar');

class SuperAIFixer {
    constructor() {
        this.isActive = true;
        this.fixCount = 0;
        this.lastError = null;
        this.projectRoot = process.cwd();
        
        // Essential dependencies that must be present
        this.criticalDependencies = [
            'express',
            'cors',
            'axios',
            'natural',
            'fs-extra',
            'chokidar',
            'nodemon'
        ];

        this.initialize();
    }

    async initialize() {
        console.log('🤖 Super AI Fixer is starting...');
        
        try {
            // Immediate fixes
            await this.performInitialFixes();
            
            // Start monitoring
            this.startActiveMonitoring();
            
            console.log('✅ Super AI Fixer is now active and monitoring');
        } catch (error) {
            console.error('Initialization error:', error);
            await this.emergencyRepair();
        }
    }

    async performInitialFixes() {
        console.log('🔍 Performing initial system check...');

        // Check and fix package.json
        await this.fixPackageJson();

        // Ensure all critical dependencies
        await this.ensureDependencies();

        // Fix common Node.js issues
        await this.fixNodeIssues();

        // Clear any corrupted modules
        await this.cleanModules();

        console.log('✅ Initial fixes completed');
    }

    async fixPackageJson() {
        const packagePath = path.join(this.projectRoot, 'package.json');
        
        try {
            let packageJson = {};
            
            // Try to read existing package.json
            try {
                packageJson = require(packagePath);
            } catch {
                // Create new if doesn't exist
                packageJson = {
                    name: "ai-project",
                    version: "1.0.0",
                    main: "index.js"
                };
            }

            // Ensure scripts
            packageJson.scripts = {
                start: "node index.js",
                dev: "nodemon index.js",
                fix: "node ./core/SuperAIFixer.js",
                ...packageJson.scripts
            };

            // Ensure dependencies
            packageJson.dependencies = {
                ...packageJson.dependencies,
                express: "^4.17.1",
                cors: "^2.8.5",
                axios: "^0.21.1",
                natural: "^5.1.13",
                "fs-extra": "^10.0.0",
                chokidar: "^3.5.2"
            };

            // Write updated package.json
            fs.writeFileSync(
                packagePath,
                JSON.stringify(packageJson, null, 2)
            );

            console.log('✅ package.json fixed');
        } catch (error) {
            console.error('Error fixing package.json:', error);
            await this.emergencyRepair();
        }
    }

    async ensureDependencies() {
        console.log('📦 Checking dependencies...');

        try {
            // Remove problematic files
            if (fs.existsSync('package-lock.json')) {
                fs.unlinkSync('package-lock.json');
            }
            
            // Install critical dependencies
            for (const dep of this.criticalDependencies) {
                try {
                    require(dep);
                } catch {
                    console.log(`Installing ${dep}...`);
                    execSync(`npm install ${dep} --save`, { stdio: 'inherit' });
                }
            }

            console.log('✅ Dependencies installed');
        } catch (error) {
            console.error('Dependency installation error:', error);
            await this.forceDependencyInstall();
        }
    }

    async forceDependencyInstall() {
        console.log('🔨 Forcing dependency installation...');
        
        try {
            // Remove all node_modules
            execSync('rm -rf node_modules');
            
            // Clear npm cache
            execSync('npm cache clean --force');
            
            // Reinstall everything
            execSync('npm install --force', { stdio: 'inherit' });
            
            console.log('✅ Force installation complete');
        } catch (error) {
            console.error('Force installation failed:', error);
            throw error;
        }
    }

    async fixNodeIssues() {
        try {
            // Check Node.js version
            const nodeVersion = process.version;
            const requiredVersion = 'v14.0.0';
            
            if (this.compareVersions(nodeVersion, requiredVersion) < 0) {
                console.log('⚠️ Node.js version is too old. Updating...');
                await this.updateNodeVersion();
            }

            // Fix potential permission issues
            if (process.platform !== 'win32') {
                execSync('chmod -R 755 .');
            }

        } catch (error) {
            console.error('Node.js fix error:', error);
        }
    }

    async cleanModules() {
        try {
            const modulesPath = path.join(this.projectRoot, 'node_modules');
            
            if (fs.existsSync(modulesPath)) {
                console.log('🧹 Cleaning node_modules...');
                
                // Remove problematic modules
                const problematicModules = await this.findProblematicModules();
                for (const mod of problematicModules) {
                    const modPath = path.join(modulesPath, mod);
                    if (fs.existsSync(modPath)) {
                        fs.rmdirSync(modPath, { recursive: true });
                    }
                }
                
                console.log('✅ Modules cleaned');
            }
        } catch (error) {
            console.error('Module cleaning error:', error);
        }
    }

    startActiveMonitoring() {
        // Monitor file changes
        const watcher = chokidar.watch([
            '**/*.js',
            'package.json',
            'node_modules'
        ], {
            ignored: /(^|[\/\\])\../,
            persistent: true
        });

        watcher
            .on('change', path => this.handleFileChange(path))
            .on('error', path => this.handleFileError(path))
            .on('unlink', path => this.handleFileDeletion(path));

        // Monitor process errors
        process.on('uncaughtException', error => this.handleError(error));
        process.on('unhandledRejection', error => this.handleError(error));

        console.log('👀 Active monitoring started');
    }

    async handleError(error) {
        console.log('🚨 Error detected:', error.message);
        
        this.lastError = error;
        this.fixCount++;

        try {
            // Analyze error
            const errorType = this.analyzeError(error);
            
            // Apply specific fix
            await this.applyFix(errorType, error);
            
            // Verify fix
            await this.verifyFix();
            
            console.log('✅ Error fixed successfully');
        } catch (fixError) {
            console.error('Fix failed:', fixError);
            await this.emergencyRepair();
        }
    }

    async emergencyRepair() {
        console.log('🚨 EMERGENCY REPAIR INITIATED');
        
        try {
            // Stop all processes
            this.stopAllProcesses();
            
            // Clear everything
            execSync('rm -rf node_modules package-lock.json');
            
            // Reinstall fresh
            await this.performInitialFixes();
            
            // Restart application
            this.restartApp();
            
            console.log('✅ Emergency repair completed');
        } catch (error) {
            console.error('Emergency repair failed:', error);
            console.log('Please manually run: npm cache clean --force && npm install');
        }
    }

    restartApp() {
        console.log('🔄 Restarting application...');
        
        const args = process.argv.slice(1);
        
        const child = spawn(process.argv[0], args, {
            detached: true,
            stdio: 'inherit'
        });

        child.unref();
        process.exit();
    }

    getStatus() {
        return {
            active: this.isActive,
            fixCount: this.fixCount,
            lastError: this.lastError,
            dependencies: this.criticalDependencies,
            timestamp: new Date().toISOString()
        };
    }
}

// Create and export the Super AI Fixer
const superAIFixer = new SuperAIFixer();
module.exports = superAIFixer; 