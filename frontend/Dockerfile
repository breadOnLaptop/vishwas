FROM node:20

WORKDIR /app

# Copy package files first to leverage cache
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy app source
COPY public/ ./public/
COPY src/ ./src/

# Expose port 3000
EXPOSE 3000

# Default command to start Vite dev server
CMD ["npm", "run", "dev"]
