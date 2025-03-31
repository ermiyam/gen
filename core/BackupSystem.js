const fs = require('fs-extra');
const path = require('path');
const tar = require('tar');
const crypto = require('crypto');

class BackupSystem {
    constructor() {
        this.backupConfig = {
            interval: 24 * 60 * 60 * 1000, // Daily
            maxBackups: 7,
            encryptBackups: true,
            compressionLevel: 9
        };
    }

    async createBackup() {
        const timestamp = new Date().toISOString();
        const backupPath = path.join(__dirname, '../backups', `backup-${timestamp}`);
        
        try {
            // Backup knowledge base
            const knowledge = await this.exportKnowledge();
            
            // Backup analytics
            const analytics = await this.exportAnalytics();
            
            // Backup configurations
            const configs = await this.exportConfigurations();

            // Create compressed archive
            await this.createArchive(backupPath, {
                knowledge,
                analytics,
                configs
            });

            // Encrypt if configured
            if (this.backupConfig.encryptBackups) {
                await this.encryptBackup(backupPath);
            }

            // Cleanup old backups
            await this.cleanupOldBackups();

            return {
                success: true,
                path: backupPath,
                timestamp
            };
        } catch (error) {
            console.error('Backup creation failed:', error);
            throw error;
        }
    }

    async restoreFromBackup(backupPath) {
        try {
            // Decrypt if needed
            if (this.backupConfig.encryptBackups) {
                await this.decryptBackup(backupPath);
            }

            // Extract archive
            const data = await this.extractArchive(backupPath);

            // Restore knowledge base
            await this.importKnowledge(data.knowledge);
            
            // Restore analytics
            await this.importAnalytics(data.analytics);
            
            // Restore configurations
            await this.importConfigurations(data.configs);

            return {
                success: true,
                timestamp: new Date()
            };
        } catch (error) {
            console.error('Restore failed:', error);
            throw error;
        }
    }
}

module.exports = new BackupSystem(); 