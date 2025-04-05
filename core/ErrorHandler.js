const { execSync } = require('child_process');
const fse = require('fs-extra');
const path = require('path');

class ErrorHandler {
    constructor() {
        this.errors = [];
        this.fixes = new Map();
        this.initialize();
    }

    initialize() {
        this.setupErrorPatterns();
        this.watchForErrors();
    }

    setupErrorPatterns() {
        this.errorPatterns = new Map([
            ['MODULE_NOT_FOUND', this.handleModuleError.bind(this)],
            ['ENOENT', this.handleFileError.bind(this)],
            ['SyntaxError', this.handleSyntaxError.bind(this)],
            ['TypeError', this.handleTypeError.bind(this)]
        ]);
    }

    watchForErrors() {
        process.on('uncaughtException', this.handle.bind(this));
        process.on('unhandledRejection', this.handle.bind(this));
    }

    async handle(error) {
        try {
            console.error('🚨 Error detected:', error.message);
            this.errors.push({
                timestamp: Date.now(),
                error: error.message,
                stack: error.stack
            });

            // Find and execute appropriate fix
            const handler = this.errorPatterns.get(error.code || error.name);
            if (handler) {
                await handler(error);
            } else {
                await this.handleGenericError(error);
            }

            return true;
        } catch (handlingError) {
            console.error('Error handler failed:', handlingError);
            return false;
        }
    }

    async handleModuleError(error) {
        const module = error.message.match(/Cannot find module '([^']+)'/)?.[1];
        if (module) {
            await this.installDependency(module);
        }
    }

    async installDependency(module) {
        try {
            console.log(`📦 Installing ${module}...`);
            execSync(`npm install ${module} --save`, { stdio: 'inherit' });
            console.log(`✅ Installed ${module}`);
            return true;
        } catch (error) {
            console.error(`Failed to install ${module}:`, error);
            return false;
        }
    }

    async handleFileError(error) {
        const filePath = error.path;
        if (filePath && !fse.existsSync(filePath)) {
            try {
                await fse.ensureFile(filePath);
                console.log(`✅ Created missing file: ${filePath}`);
            } catch (createError) {
                console.error(`Failed to create file ${filePath}:`, createError);
            }
        }
    }

    async handleSyntaxError(error) {
        console.error('Syntax error detected:', error.message);
        // Log the error for manual review
        await this.logError(error);
    }

    async handleTypeError(error) {
        console.error('Type error detected:', error.message);
        // Log the error for manual review
        await this.logError(error);
    }

    async handleGenericError(error) {
        console.error('Generic error:', error.message);
        await this.logError(error);
    }

    async logError(error) {
        try {
            const logPath = path.join(process.cwd(), 'error.log');
            const logEntry = `${new Date().toISOString()} - ${error.stack}\n`;
            await fse.appendFile(logPath, logEntry);
        } catch (logError) {
            console.error('Failed to log error:', logError);
        }
    }

    getStatus() {
        return {
            totalErrors: this.errors.length,
            recentErrors: this.errors.slice(-5),
            fixes: Array.from(this.fixes.entries())
        };
    }
}

module.exports = { ErrorHandler }; 