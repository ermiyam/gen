const fs = require('fs').promises;
const path = require('path');
const detective = require('detective');
const resolve = require('resolve');

class DependencyManager {
    constructor() {
        this.importMap = new Map();
        this.moduleCache = new Map();
        this.initialize();
    }

    async initialize() {
        try {
            await this.scanProjectImports();
            this.watchImports();
            console.log('📦 Dependency Manager initialized');
        } catch (error) {
            console.error('Dependency Manager initialization error:', error);
        }
    }

    async scanProjectImports() {
        const files = await this.getProjectFiles();
        for (const file of files) {
            await this.analyzeFileImports(file);
        }
    }

    async analyzeFileImports(filepath) {
        try {
            const content = await fs.readFile(filepath, 'utf8');
            const imports = detective(content);
            
            // Store imports for this file
            this.importMap.set(filepath, new Set(imports));
            
            // Check for duplicates
            await this.checkDuplicateImports(filepath);
        } catch (error) {
            console.error(`Import analysis error in ${filepath}:`, error);
        }
    }

    async checkDuplicateImports(filepath) {
        const content = await fs.readFile(filepath, 'utf8');
        const lines = content.split('\n');
        const imports = new Map();

        // Find all require/import statements
        lines.forEach((line, index) => {
            const requireMatch = line.match(/const\s+(\w+)\s+=\s+require\(['"]([^'"]+)['"]\)/);
            const importMatch = line.match(/import\s+(\w+)\s+from\s+['"]([^'"]+)['"]/);
            
            if (requireMatch || importMatch) {
                const [, variable, module] = requireMatch || importMatch;
                
                if (imports.has(module)) {
                    this.fixDuplicateImport(filepath, index, line);
                } else {
                    imports.set(module, { line: index, variable });
                }
            }
        });
    }

    async fixDuplicateImport(filepath, lineNumber, duplicateLine) {
        try {
            let content = await fs.readFile(filepath, 'utf8');
            const lines = content.split('\n');
            
            // Remove duplicate import
            lines.splice(lineNumber, 1);
            
            // Write back to file
            await fs.writeFile(filepath, lines.join('\n'));
            
            console.log(`✅ Removed duplicate import in ${filepath} at line ${lineNumber + 1}`);
            
            // Verify the fix
            await this.verifyFix(filepath);
        } catch (error) {
            console.error('Fix application error:', error);
        }
    }

    async verifyFix(filepath) {
        try {
            // Try to require the file
            delete require.cache[require.resolve(filepath)];
            require(filepath);
            return true;
        } catch (error) {
            console.error('Fix verification failed:', error);
            return false;
        }
    }

    watchImports() {
        const watcher = require('chokidar').watch(['**/*.js', '**/*.ts'], {
            ignored: /(node_modules|\.git)/,
            persistent: true
        });

        watcher.on('change', async (filepath) => {
            await this.analyzeFileImports(filepath);
        });
    }
}

// Create and export the Dependency Manager
const dependencyManager = new DependencyManager();
module.exports = dependencyManager; 