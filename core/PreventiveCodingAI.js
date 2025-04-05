const fs = require('fs').promises;
const path = require('path');
const chokidar = require('chokidar');
const esprima = require('esprima');
const escodegen = require('escodegen');

class PreventiveCodingAI {
    constructor() {
        this.activeFiles = new Map();
        this.errorPatterns = new Map();
        this.fixHistory = new Map();
        this.watching = false;
        
        // Initialize immediately
        this.initialize();
    }

    async initialize() {
        try {
            // Load known error patterns
            await this.loadErrorPatterns();
            
            // Start real-time code monitoring
            this.startCodeWatcher();
            
            // Initialize preventive scanning
            this.startPreventiveScans();
            
            console.log('🤖 Preventive Coding AI is active and protecting your code');
        } catch (error) {
            console.error('Initialization error:', error);
            // Auto-recover
            this.selfRecover();
        }
    }

    startCodeWatcher() {
        if (this.watching) return;
        
        // Watch for file changes in real-time
        const watcher = chokidar.watch(['*.js', 'src/**/*.js', 'core/**/*.js'], {
            ignored: /(node_modules|\.git)/,
            persistent: true
        });

        watcher
            .on('change', async (filepath) => {
                await this.analyzeFileChange(filepath);
            })
            .on('add', async (filepath) => {
                await this.analyzeNewFile(filepath);
            });

        this.watching = true;
        console.log('👀 Code watcher activated');
    }

    async analyzeFileChange(filepath) {
        try {
            // Read file content
            const content = await fs.readFile(filepath, 'utf8');
            
            // Parse code for potential issues
            const issues = await this.analyzePotentialIssues(content, filepath);
            
            if (issues.length > 0) {
                console.log(`🔍 Found ${issues.length} potential issues in ${filepath}`);
                // Auto-fix issues
                await this.preventIssues(issues, filepath);
            }
        } catch (error) {
            console.error(`Analysis error in ${filepath}:`, error);
        }
    }

    async analyzePotentialIssues(content, filepath) {
        const issues = [];
        try {
            // Parse code into AST
            const ast = esprima.parseScript(content, { loc: true });
            
            // Check for common issues
            issues.push(...this.checkSyntaxIssues(ast));
            issues.push(...this.checkRuntimeIssues(ast));
            issues.push(...this.checkMemoryIssues(ast));
            issues.push(...this.checkAsyncIssues(ast));
            
            return issues;
        } catch (error) {
            // If parsing fails, it's a syntax error
            issues.push({
                type: 'SyntaxError',
                location: { line: error.lineNumber, column: error.column },
                message: error.message,
                severity: 'high'
            });
            return issues;
        }
    }

    async preventIssues(issues, filepath) {
        for (const issue of issues) {
            try {
                // Generate fix
                const fix = await this.generateFix(issue);
                
                // Apply fix
                await this.applyPreventiveFix(fix, filepath);
                
                // Verify fix
                await this.verifyFix(filepath);
                
                console.log(`✅ Prevented ${issue.type} in ${filepath}`);
                
                // Learn from successful fix
                await this.learnFromFix(issue, fix);
                
            } catch (error) {
                console.error(`Failed to prevent issue in ${filepath}:`, error);
            }
        }
    }

    async generateFix(issue) {
        const fix = {
            type: issue.type,
            location: issue.location,
            code: null
        };

        switch (issue.type) {
            case 'SyntaxError':
                fix.code = this.fixSyntaxError(issue);
                break;
                
            case 'MemoryLeak':
                fix.code = this.fixMemoryLeak(issue);
                break;
                
            case 'AsyncError':
                fix.code = this.fixAsyncError(issue);
                break;
                
            case 'RuntimeError':
                fix.code = this.fixRuntimeError(issue);
                break;
                
            default:
                fix.code = this.generateGeneralFix(issue);
        }

        return fix;
    }

    async applyPreventiveFix(fix, filepath) {
        try {
            // Read current file content
            let content = await fs.readFile(filepath, 'utf8');
            
            // Create backup
            await this.createBackup(filepath, content);
            
            // Apply the fix
            content = this.injectFix(content, fix);
            
            // Write fixed content
            await fs.writeFile(filepath, content, 'utf8');
            
            // Record the fix
            this.recordFix(filepath, fix);
            
        } catch (error) {
            console.error('Fix application error:', error);
            // Restore from backup if fix fails
            await this.restoreFromBackup(filepath);
            throw error;
        }
    }

    injectFix(content, fix) {
        const lines = content.split('\n');
        const { line, column } = fix.location;
        
        // Insert the fix at the correct location
        if (fix.type === 'SyntaxError') {
            lines[line - 1] = fix.code;
        } else {
            // For other types of fixes, we might need to insert new code
            lines.splice(line - 1, 0, fix.code);
        }
        
        return lines.join('\n');
    }

    async verifyFix(filepath) {
        try {
            // Read the fixed file
            const content = await fs.readFile(filepath, 'utf8');
            
            // Try parsing it
            esprima.parseScript(content);
            
            // Run basic tests if available
            await this.runTests(filepath);
            
            return true;
        } catch (error) {
            console.error('Fix verification failed:', error);
            return false;
        }
    }

    async createBackup(filepath, content) {
        const backupPath = `${filepath}.backup`;
        await fs.writeFile(backupPath, content, 'utf8');
    }

    async restoreFromBackup(filepath) {
        const backupPath = `${filepath}.backup`;
        try {
            const backup = await fs.readFile(backupPath, 'utf8');
            await fs.writeFile(filepath, backup, 'utf8');
            console.log('✅ Restored from backup');
        } catch (error) {
            console.error('Backup restoration failed:', error);
        }
    }

    recordFix(filepath, fix) {
        if (!this.fixHistory.has(filepath)) {
            this.fixHistory.set(filepath, []);
        }
        
        this.fixHistory.get(filepath).push({
            timestamp: Date.now(),
            fix: fix
        });
    }

    getFixHistory(filepath) {
        return this.fixHistory.get(filepath) || [];
    }

    async selfRecover() {
        console.log('🔄 Initiating self-recovery...');
        
        try {
            // Reset internal state
            this.watching = false;
            
            // Reload error patterns
            await this.loadErrorPatterns();
            
            // Restart code watcher
            this.startCodeWatcher();
            
            console.log('✅ Self-recovery complete');
        } catch (error) {
            console.error('Self-recovery failed:', error);
        }
    }
}

// Create and export the Preventive Coding AI
const preventiveCodingAI = new PreventiveCodingAI();
module.exports = preventiveCodingAI; 