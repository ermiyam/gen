const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

class TerminalProblemSolver {
    constructor() {
        this.currentErrors = new Set();
        this.fixAttempts = new Map();
        this.isFixing = false;
        
        // Common error patterns and their fixes
        this.errorPatterns = new Map([
            ['ENOENT', this.handleFileNotFound],
            ['Cannot find module', this.handleMissingModule],
            ['Port', this.handlePortConflict],
            ['Permission denied', this.handlePermissionError],
            ['Unexpected token', this.handleSyntaxError],
            ['TypeError', this.handleTypeError]
        ]);

        this.initialize();
    }

    async initialize() {
        try {
            // Clear any existing error logs
            await this.clearErrorLogs();
            
            // Start monitoring terminal output
            this.startErrorMonitoring();
            
            // Perform initial system check
            await this.performSystemCheck();
            
            console.log('🔍 Terminal Problem Solver is active');
        } catch (error) {
            await this.handleInitializationError(error);
        }
    }

    startErrorMonitoring() {
        // Monitor process errors
        process.on('uncaughtException', async (error) => {
            await this.handleError(error);
        });

        process.on('unhandledRejection', async (error) => {
            await this.handleError(error);
        });

        // Monitor stderr
        process.stderr.on('data', async (data) => {
            await this.analyzeError(data.toString());
        });
    }

    async handleError(error) {
        if (this.isFixing) return; // Prevent recursive fixing
        this.isFixing = true;

        try {
            console.log('🔧 Analyzing error:', error.message);

            // Find matching error pattern
            for (const [pattern, handler] of this.errorPatterns) {
                if (error.message.includes(pattern)) {
                    await handler.call(this, error);
                    break;
                }
            }

            // If no specific handler found, try general fix
            if (!this.errorFixed) {
                await this.attemptGeneralFix(error);
            }

            await this.verifyFix();
        } catch (fixError) {
            console.error('Fix attempt failed:', fixError);
            await this.emergencyRecovery();
        } finally {
            this.isFixing = false;
        }
    }

    async handleMissingModule(error) {
        const moduleName = error.message.match(/Cannot find module '([^']+)'/)?.[1];
        if (moduleName) {
            console.log(`📦 Installing missing module: ${moduleName}`);
            try {
                // Try npm install first
                execSync(`npm install ${moduleName}`, { stdio: 'inherit' });
            } catch (installError) {
                // If npm install fails, try alternative approaches
                await this.alternativeModuleInstall(moduleName);
            }
        }
    }

    async alternativeModuleInstall(moduleName) {
        try {
            // Try yarn if available
            execSync('yarn --version', { stdio: 'ignore' });
            execSync(`yarn add ${moduleName}`, { stdio: 'inherit' });
        } catch (yarnError) {
            try {
                // Try installing with --force
                execSync(`npm install ${moduleName} --force`, { stdio: 'inherit' });
            } catch (npmError) {
                throw new Error(`Failed to install ${moduleName}`);
            }
        }
    }

    async handlePortConflict(error) {
        const port = error.message.match(/port\s+(\d+)/i)?.[1];
        if (port) {
            try {
                // Find process using the port
                const command = process.platform === 'win32' 
                    ? `netstat -ano | findstr :${port}`
                    : `lsof -i :${port}`;
                
                const result = execSync(command).toString();
                
                // Kill the process
                const pid = result.match(/(\d+)\s*$/m)?.[1];
                if (pid) {
                    process.platform === 'win32'
                        ? execSync(`taskkill /F /PID ${pid}`)
                        : execSync(`kill -9 ${pid}`);
                    
                    console.log(`✅ Freed port ${port}`);
                }
            } catch (error) {
                // If we can't kill the process, try using a different port
                this.updatePortInFiles(port, this.findAvailablePort());
            }
        }
    }

    async handlePermissionError(error) {
        const file = error.message.match(/['"]([^'"]+)['"]/)?.[1];
        if (file) {
            try {
                if (process.platform !== 'win32') {
                    execSync(`chmod 777 ${file}`);
                } else {
                    execSync(`icacls "${file}" /grant Everyone:F`);
                }
                console.log(`✅ Fixed permissions for ${file}`);
            } catch (error) {
                console.error('Permission fix failed:', error);
            }
        }
    }

    async handleSyntaxError(error) {
        const file = error.stack.match(/at\s+(.+?):(\d+)/)?.[1];
        if (file) {
            try {
                let content = fs.readFileSync(file, 'utf8');
                const fixedContent = await this.fixSyntax(content);
                fs.writeFileSync(file, fixedContent);
                console.log(`✅ Fixed syntax in ${file}`);
            } catch (error) {
                console.error('Syntax fix failed:', error);
            }
        }
    }

    async fixSyntax(content) {
        // Common syntax fixes
        return content
            .replace(/([{[])\s*,\s*([}\]])/g, '$1$2') // Remove trailing commas
            .replace(/;+/g, ';') // Fix multiple semicolons
            .replace(/([^=!])===/g, '$1==') // Fix strict equality
            .replace(/\s+/g, ' ') // Normalize whitespace
            .trim();
    }

    async performSystemCheck() {
        try {
            // Check Node.js version
            const nodeVersion = process.version;
            if (!this.isCompatibleVersion(nodeVersion)) {
                await this.upgradeNode();
            }

            // Check npm
            execSync('npm --version');

            // Verify package.json
            this.verifyPackageJson();

            // Check node_modules
            this.verifyNodeModules();

            console.log('✅ System check completed');
        } catch (error) {
            console.error('System check failed:', error);
            await this.emergencyRecovery();
        }
    }

    async emergencyRecovery() {
        console.log('🚨 Starting emergency recovery...');
        
        try {
            // Clear problematic files
            execSync('rm -rf node_modules package-lock.json');
            
            // Reinstall dependencies
            execSync('npm install');
            
            // Clear require cache
            this.clearRequireCache();
            
            // Restart the application
            this.restartApp();
            
            console.log('✅ Emergency recovery completed');
        } catch (error) {
            console.error('Emergency recovery failed:', error);
            console.log('Please run: npm cache clean --force && npm install');
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
}

// Create and export the Terminal Problem Solver
const terminalProblemSolver = new TerminalProblemSolver();
module.exports = terminalProblemSolver; 