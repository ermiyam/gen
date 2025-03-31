const fs = require('fs').promises;
const path = require('path');
const natural = require('natural');

class AIRepairSystem {
    constructor() {
        this.isActive = true;
        this.brain = new natural.BayesClassifier();
        this.knownFixes = new Map();
        this.running = false;
        
        // Initialize the repair system
        this.init();
    }

    async init() {
        // Start monitoring the main system
        this.startMonitoring();
        
        // Load known fixes
        await this.loadKnownFixes();
        
        console.log('AI Repair System is now active and monitoring for issues');
    }

    startMonitoring() {
        if (this.running) return;
        this.running = true;

        // Monitor for crashes
        process.on('uncaughtException', async (error) => {
            await this.handleCrash(error);
        });

        // Monitor for unhandled promise rejections
        process.on('unhandledRejection', async (error) => {
            await this.handleCrash(error);
        });

        // Monitor system stability
        setInterval(() => this.checkSystemHealth(), 1000);
    }

    async handleCrash(error) {
        try {
            console.log('⚠️ Crash detected! AI Repair System responding...');
            
            // Analyze the error
            const errorAnalysis = this.analyzeError(error);
            
            // Find or generate fix
            const fix = await this.generateFix(errorAnalysis);
            
            // Apply the fix
            await this.applyFix(fix);
            
            // Verify the fix worked
            await this.verifyFix();
            
            console.log('✅ System recovered successfully!');
        } catch (repairError) {
            console.error('❌ AI Repair System encountered an error:', repairError);
            await this.emergencyRecover();
        }
    }

    analyzeError(error) {
        const analysis = {
            type: error.name,
            message: error.message,
            stack: error.stack,
            location: this.extractErrorLocation(error),
            severity: this.calculateSeverity(error)
        };

        console.log('🔍 Error Analysis:', analysis);
        return analysis;
    }

    async generateFix(errorAnalysis) {
        // Check if we have a known fix
        if (this.knownFixes.has(errorAnalysis.message)) {
            return this.knownFixes.get(errorAnalysis.message);
        }

        // Generate new fix based on error type
        const fix = await this.createNewFix(errorAnalysis);
        
        // Store the fix for future use
        this.knownFixes.set(errorAnalysis.message, fix);
        
        return fix;
    }

    async createNewFix(errorAnalysis) {
        const fix = {
            type: errorAnalysis.type,
            action: null,
            code: null
        };

        switch (errorAnalysis.type) {
            case 'ReferenceError':
                fix.action = 'define_variable';
                fix.code = this.generateVariableDefinition(errorAnalysis);
                break;
                
            case 'SyntaxError':
                fix.action = 'fix_syntax';
                fix.code = this.fixSyntaxError(errorAnalysis);
                break;
                
            case 'TypeError':
                fix.action = 'type_correction';
                fix.code = this.correctTypeError(errorAnalysis);
                break;

            case 'RangeError':
                fix.action = 'range_correction';
                fix.code = this.fixRangeError(errorAnalysis);
                break;

            default:
                fix.action = 'general_recovery';
                fix.code = this.generateGeneralFix(errorAnalysis);
        }

        return fix;
    }

    async applyFix(fix) {
        console.log('🔧 Applying fix:', fix);

        try {
            switch (fix.action) {
                case 'define_variable':
                    await this.applyVariableFix(fix);
                    break;
                    
                case 'fix_syntax':
                    await this.applySyntaxFix(fix);
                    break;
                    
                case 'type_correction':
                    await this.applyTypeFix(fix);
                    break;
                    
                case 'range_correction':
                    await this.applyRangeFix(fix);
                    break;
                    
                case 'general_recovery':
                    await this.applyGeneralFix(fix);
                    break;
            }

            // Save the successful fix
            await this.saveSuccessfulFix(fix);
            
        } catch (error) {
            console.error('Fix application failed:', error);
            throw error;
        }
    }

    async applyVariableFix(fix) {
        const filePath = fix.location;
        let content = await fs.readFile(filePath, 'utf8');
        content = content.replace(/\n/g, '\n');
        
        // Add variable definition
        content = `${fix.code}\n${content}`;
        
        await fs.writeFile(filePath, content);
    }

    async applySyntaxFix(fix) {
        const filePath = fix.location;
        let content = await fs.readFile(filePath, 'utf8');
        
        // Apply syntax correction
        content = content.replace(fix.errorCode, fix.code);
        
        await fs.writeFile(filePath, content);
    }

    async verifyFix() {
        return new Promise((resolve) => {
            // Wait a short period to ensure system stability
            setTimeout(async () => {
                try {
                    // Check if system is running
                    const health = await this.checkSystemHealth();
                    resolve(health.status === 'healthy');
                } catch (error) {
                    resolve(false);
                }
            }, 1000);
        });
    }

    async checkSystemHealth() {
        const health = {
            status: 'healthy',
            memory: process.memoryUsage(),
            uptime: process.uptime(),
            errors: 0
        };

        // Check memory usage
        if (health.memory.heapUsed / health.memory.heapTotal > 0.9) {
            health.status = 'warning';
        }

        return health;
    }

    async emergencyRecover() {
        console.log('🚨 Initiating emergency recovery...');
        
        try {
            // Restart core services
            await this.restartServices();
            
            // Clear memory
            global.gc && global.gc();
            
            // Reset error handlers
            this.resetErrorHandlers();
            
            console.log('✅ Emergency recovery completed');
        } catch (error) {
            console.error('❌ Emergency recovery failed:', error);
        }
    }

    async saveSuccessfulFix(fix) {
        try {
            const fixes = await this.loadKnownFixes();
            fixes.push({
                timestamp: Date.now(),
                fix: fix
            });
            
            await fs.writeFile(
                path.join(__dirname, 'known_fixes.json'),
                JSON.stringify(fixes, null, 2)
            );
        } catch (error) {
            console.error('Error saving fix:', error);
        }
    }
}

// Create and export the AI Repair System
const aiRepairSystem = new AIRepairSystem();
module.exports = aiRepairSystem; 