import { useState } from "react";

const PredictForm = ({ onSubmit }) => {
  const [ticker, setTicker] = useState("");
  const [prices, setPrices] = useState("");
  const [timeframe, setTimeframe] = useState("1d");

  const handleSubmit = (e) => {
    e.preventDefault();
    const priceArray = prices.split(",").map(Number);
    onSubmit(ticker, priceArray, timeframe);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input placeholder="Ticker" value={ticker} onChange={e => setTicker(e.target.value)} />
      <input placeholder="Prices (comma-separated)" value={prices} onChange={e => setPrices(e.target.value)} />
      <select value={timeframe} onChange={e => setTimeframe(e.target.value)}>
        <option value="1d">1d</option>
        <option value="1h">1h</option>
        <option value="5m">5m</option>
      </select>
      <button type="submit">Predict</button>
    </form>
  );
};

export default PredictForm;