const fs = require('fs-extra');
const path = require('path');

class DatabaseFallback {
    constructor() {
        this.dbPath = path.join(process.cwd(), 'data');
        this.initialize();
    }

    initialize() {
        fs.ensureDirSync(this.dbPath);
    }

    async save(collection, data) {
        const filePath = path.join(this.dbPath, `${collection}.json`);
        await fs.writeJSON(filePath, data);
    }

    async load(collection) {
        const filePath = path.join(this.dbPath, `${collection}.json`);
        try {
            return await fs.readJSON(filePath);
        } catch {
            return [];
        }
    }
}

module.exports = { DatabaseFallback }; 