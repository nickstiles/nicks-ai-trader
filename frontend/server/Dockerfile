FROM node:18-alpine

WORKDIR /app

# Install only what's needed
COPY package*.json ./
RUN npm install

# Copy the rest of the code
COPY . .

# Expose the proxy port
EXPOSE 3001

CMD ["node", "index.js"]