class VisualResponseFormatter {
    formatResponse(response) {
        return {
            html: this.generateHTML(response),
            markdown: this.generateMarkdown(response),
            plainText: this.generatePlainText(response)
        };
    }

    generateHTML(response) {
        let html = '<div class="ai-response">';
        
        // Add header with confidence indicator
        html += `
            <div class="response-header">
                <div class="confidence-indicator" style="background: ${this.getConfidenceColor(response.confidence)}">
                    ${Math.round(response.confidence * 100)}% confident
                </div>
            </div>
        `;

        // Main content
        html += `<div class="response-content">${response.content}</div>`;

        // Sources if available
        if (response.sources && response.sources.length > 0) {
            html += '<div class="response-sources">';
            html += '<h4>Sources:</h4>';
            html += '<ul>';
            response.sources.forEach(source => {
                html += `<li><a href="${source.url}" target="_blank">${source.title}</a></li>`;
            });
            html += '</ul></div>';
        }

        // Related topics
        if (response.relatedTopics && response.relatedTopics.length > 0) {
            html += '<div class="related-topics">';
            html += '<h4>Related Topics:</h4>';
            html += response.relatedTopics.map(topic => 
                `<span class="topic-tag">${topic}</span>`
            ).join('');
            html += '</div>';
        }

        html += '</div>';
        return html;
    }

    // ... (markdown and plaintext generators) ...
} 