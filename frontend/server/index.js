// frontend/server/index.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const GO_API_URL = process.env.GO_API_URL || 'http://localhost:8080';

app.use(cors());
app.use(express.json());

// Health check
app.get('/api/ping', (req, res) => {
  res.json({ message: 'pong from node' });
});

// Proxy to Python FastAPI
app.post('/api/predict', async (req, res) => {
  console.log("🔁 [Node] Received /api/predict request:", req.body);
  try {
    const response = await axios.post(`${PYTHON_API_URL}/predict`, req.body);
    console.log("✅ [Node] Response from Python:", response.data);
    res.json(response.data);
  } catch (error) {
    console.error("❌ [Node] Prediction proxy error:", error.message);
    res.status(500).json({ error: 'Prediction service unavailable' });
  }
});

// Proxy to Go service
app.get('/api/status', async (req, res) => {
  console.log("🔎 [Node] Calling Go /status...");
  try {
    const response = await axios.get(`${GO_API_URL}/status`);
    console.log("✅ [Node] Response from Go:", response.data);
    res.json(response.data);
  } catch (error) {
    console.error("❌ [Node] Go service error:", error.message);
    res.status(500).json({ error: 'Go service unavailable' });
  }
});

app.listen(PORT, () => {
  console.log(`Node API server listening on http://localhost:${PORT}`);
});