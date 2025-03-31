# Use official Node.js image
FROM node:16-slim

# Create app directory
WORKDIR /app

# Install app dependencies
COPY package*.json ./
RUN npm install --production

# Bundle app source
COPY . .

# Expose port
EXPOSE 8080

# Start command
CMD [ "npm", "start" ] 