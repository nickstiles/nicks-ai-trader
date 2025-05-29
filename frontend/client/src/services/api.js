const API_BASE = "http://localhost:3001/api";

/**
 * Fetch health/status from Go service via Node.
 */
export async function getStatus() {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

/**
 * Call /predict (FastAPI) via Node proxy.
 * Pass an empty array for recent_prices to trigger live-data fetch.
 */
export async function postPrediction(ticker, prices = [], timeframe) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, recent_prices: prices, timeframe }),
  });
  if (!res.ok) throw new Error("Prediction request failed");
  return res.json();
}

/**
 * Submit a trade order via Go service through Node proxy.
 */
export async function submitTrade({ ticker, action, quantity }) {
  const res = await fetch(`${API_BASE}/submit-trade`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, action, quantity }),
  });
  if (!res.ok) throw new Error("Trade submission failed");
  return res.json();
}

/**
 * Run a backtest via FastAPI → Node proxy.
 */
export async function postBacktest({ ticker, period = "1mo", timeframe = "1d", short_window = 5, long_window = 20, quantity = 10 }) {
  const res = await fetch(`${API_BASE}/backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, period, timeframe, short_window, long_window, quantity }),
  });
  if (!res.ok) throw new Error("Backtest request failed");
  return res.json();
}