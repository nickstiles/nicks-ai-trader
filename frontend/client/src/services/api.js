const API_BASE = "http://localhost:3001/api";

export async function getStatus() {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

export async function postPrediction(ticker, prices, timeframe) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, recent_prices: prices, timeframe }),
  });
  return res.json();
}