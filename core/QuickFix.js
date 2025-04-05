const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class QuickFix {
    static async fix() {
        console.log('🔧 Running Quick Fix...');
        
        try {
            // 1. Install fs-extra specifically
            execSync('npm install fs-extra --save', { stdio: 'inherit' });
            
            // 2. Verify fs-extra installation
            require('fs-extra');
            
            console.log('✅ fs-extra installed successfully');
            
            // 3. Install other critical dependencies
            const dependencies = [
                'express',
                'cors',
                'axios',
                'natural',
                'chokidar',
                'nodemon'
            ];
            
            dependencies.forEach(dep => {
                try {
                    require(dep);
                } catch {
                    console.log(`Installing ${dep}...`);
                    execSync(`npm install ${dep} --save`, { stdio: 'inherit' });
                }
            });

            console.log('✅ All dependencies installed');
            
            return true;
        } catch (error) {
            console.error('❌ Error during fix:', error);
            return false;
        }
    }
}

// Run the fix immediately
QuickFix.fix().then(success => {
    if (success) {
        console.log('🚀 System ready to start');
    } else {
        console.log('⚠️ Please run: npm install fs-extra --save');
    }
});

module.exports = QuickFix; 