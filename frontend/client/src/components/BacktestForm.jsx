import { useState } from 'react';
import AsyncSelect from 'react-select/async';
import { loadTickerOptions } from "../services/tickers";

export default function BacktestForm({ onSubmit }) {
  const [ticker, setTicker] = useState(null);
  const [period, setPeriod] = useState('3mo');
  const [timeframe, setTimeframe] = useState('1d');
  const [shortWindow, setShortWindow] = useState(5);
  const [longWindow, setLongWindow] = useState(20);
  const [quantity, setQuantity] = useState(10);
  const [loading, setLoading] = useState(false);

  const handle = async e => {
    e.preventDefault();
    setLoading(true);
    await onSubmit({ ticker: ticker.value, period, timeframe, short_window: shortWindow, long_window: longWindow, quantity });
    setLoading(false);
  };

  return (
    <form onSubmit={handle} className="bg-white p-6 rounded-xl shadow-md space-y-4 mt-6">
      <h2 className="text-2xl font-semibold text-gray-800">Run Backtest</h2>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Ticker Symbol</label>
        <AsyncSelect
          cacheOptions
          loadOptions={loadTickerOptions}
          defaultOptions={[]}
          onChange={setTicker}            // ticker is now an object { value, label }
          placeholder="Start typing a ticker..."
          className="react-select-container"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Period</label>
          <select className="w-full border rounded px-3 py-2" value={period} onChange={e => setPeriod(e.target.value)}>
            <option value="1mo">1 Month</option>
            <option value="3mo">3 Months</option>
            <option value="6mo">6 Months</option>
            <option value="1y">1 Year</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
          <select className="w-full border rounded px-3 py-2" value={timeframe} onChange={e => setTimeframe(e.target.value)}>
            <option value="1d">1 Day</option>
            <option value="1h">1 Hour</option>
            <option value="5m">5 Minutes</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Short MA</label>
          <input
            type="number" className="w-full border rounded px-3 py-2"
            value={shortWindow} onChange={e => setShortWindow(+e.target.value)}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Long MA</label>
          <input
            type="number" className="w-full border rounded px-3 py-2"
            value={longWindow} onChange={e => setLongWindow(+e.target.value)}
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Quantity</label>
        <input
          type="number" min="1" className="w-full border rounded px-3 py-2"
          value={quantity} onChange={e => setQuantity(+e.target.value)}
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className={`w-full bg-purple-600 text-white py-2 rounded transition ${loading ? 'opacity-50' : 'hover:bg-purple-700'}`}
      >
        {loading ? 'Running...' : 'Run Backtest'}
      </button>
    </form>
  );
}
