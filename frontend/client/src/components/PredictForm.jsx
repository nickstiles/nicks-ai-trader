import { useState } from "react";
import AsyncSelect from "react-select/async";
import { loadTickerOptions } from "../services/tickers";

const PredictForm = ({ onSubmit }) => {
  const [ticker, setTicker] = useState(null);
  const [timeframe, setTimeframe] = useState("1d");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!ticker) return;
    setIsSubmitting(true);
    await onSubmit(ticker.value, timeframe);
    setIsSubmitting(false);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white p-6 rounded-xl shadow-md space-y-4 mt-6"
    >
      <h2 className="text-2xl font-semibold text-gray-800">Predict a Trade</h2>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Ticker Symbol</label>
        <AsyncSelect
          cacheOptions
          loadOptions={loadTickerOptions}
          defaultOptions={[]}
          onChange={setTicker}
          placeholder="Start typing a ticker..."
          className="react-select-container"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
        <select
          className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
        >
          <option value="1d">1 Day</option>
          <option value="1h">1 Hour</option>
          <option value="5m">5 Minutes</option>
        </select>
      </div>

      <button
        type="submit"
        disabled={!ticker || isSubmitting}
        className={`w-full bg-blue-600 text-white py-2 px-4 rounded transition duration-300 transform ${
          isSubmitting ? "opacity-50 scale-95 cursor-not-allowed" : "hover:bg-blue-700 hover:scale-105"
        }`}
      >
        {isSubmitting ? "Predicting..." : "Predict"}
      </button>
    </form>
  );
};

export default PredictForm;