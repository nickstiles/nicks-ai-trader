FROM node:18-slim

WORKDIR /app

COPY package*.json ./

ENV NODE_ENV=development
ENV ROLLUP_DISABLE_NATIVE=1

RUN npm install && ls -l node_modules && echo "✅ npm install complete"

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]