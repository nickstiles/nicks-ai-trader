// frontend/server/index.js
require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
const GO_API_URL = process.env.GO_API_URL || 'http://localhost:8080';
const FMP_API_URL = process.env.FMP_API_URL || 'https://financialmodelingprep.com/api/v3';
const FMP_API_KEY = process.env.FMP_API_KEY;

app.use(cors());
app.use(express.json());

// In-memory ticker cache
let allTickers = [];

// Function to load tickers from external API
async function loadAllTickers() {
  try {
    const response = await axios.get(`${FMP_API_URL}/stock/list`, {
      params: { apikey: FMP_API_KEY }
    });
    allTickers = response.data.map(item => ({
      value: item.symbol,
      label: `${item.symbol} — ${item.name}`
    }));
    console.log(`🔄 [Node] Loaded ${allTickers.length} tickers`);
  } catch (error) {
    console.error('❌ [Node] Failed to load tickers:', error.message);
  }
}

// Initial load
loadAllTickers();
// Refresh tickers every 24 hours
setInterval(loadAllTickers, 24 * 60 * 60 * 1000);

// Health check
app.get('/api/ping', (req, res) => {
  res.json({ message: 'pong from node' });
});

// Proxy to Python FastAPI
app.post('/api/predict', async (req, res) => {
  console.log('🔁 [Node] Received /api/predict request:', req.body);
  try {
    const response = await axios.post(`${PYTHON_API_URL}/predict`, req.body);
    console.log('✅ [Node] Response from Python:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('❌ [Node] Prediction proxy error:', error.message);
    res.status(500).json({ error: 'Prediction service unavailable' });
  }
});

// Proxy to Go service status
app.get('/api/status', async (req, res) => {
  console.log('🔎 [Node] Calling Go /status...');
  try {
    const response = await axios.get(`${GO_API_URL}/status`);
    console.log('✅ [Node] Response from Go:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('❌ [Node] Go service status error:', error.message);
    res.status(500).json({ error: 'Go service unavailable' });
  }
});

// Proxy trade submissions to Go
app.post('/api/submit-trade', async (req, res) => {
  console.log('🔁 [Node] Received /api/submit-trade request:', req.body);
  try {
    const response = await axios.post(`${GO_API_URL}/submit-trade`, req.body);
    console.log('✅ [Node] Response from Go (submit-trade):', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('❌ [Node] Submit trade proxy error:', error.message);
    res.status(500).json({ error: 'Trade service unavailable' });
  }
});

// Proxy backtest requests to Python FastAPI
app.post('/api/backtest', async (req, res) => {
  console.log('🔁 [Node] Received /api/backtest request:', req.body);
  try {
    const response = await axios.post(`${PYTHON_API_URL}/backtest`, req.body);
    console.log('✅ [Node] Response from Python (backtest):', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('❌ [Node] Backtest proxy error:', error.message);
    res.status(500).json({ error: 'Backtest service unavailable' });
  }
});

// Ticker search using in-memory cache
app.get('/api/tickers', (req, res) => {
  const q = (req.query.q || '').toUpperCase();
  const results = allTickers
    .filter(t => t.value.startsWith(q) || t.label.toUpperCase().startsWith(q))
    .slice(0, 100);
  res.json(results);
});

app.listen(PORT, () => {
  console.log(`Node API server listening on http://localhost:${PORT}`);
});
