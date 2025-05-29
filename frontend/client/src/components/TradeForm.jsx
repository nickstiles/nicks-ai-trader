import { useState } from 'react';

const TradeForm = ({ onSubmit }) => {
  const [ticker, setTicker] = useState('');
  const [action, setAction] = useState('buy');
  const [quantity, setQuantity] = useState(1);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    await onSubmit({ ticker, action, quantity });
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-xl shadow-md space-y-4 mt-6">
      <h2 className="text-2xl font-semibold text-gray-800">Submit a Trade</h2>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Ticker</label>
        <input
          type="text"
          value={ticker}
          onChange={e => setTicker(e.target.value)}
          className="w-full border rounded px-3 py-2 focus:ring-blue-500"
          placeholder="AAPL"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Action</label>
        <select
          value={action}
          onChange={e => setAction(e.target.value)}
          className="w-full border rounded px-3 py-2 focus:ring-blue-500"
        >
          <option value="buy">Buy</option>
          <option value="sell">Sell</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Quantity</label>
        <input
          type="number"
          min="1"
          value={quantity}
          onChange={e => setQuantity(Number(e.target.value))}
          className="w-full border rounded px-3 py-2 focus:ring-blue-500"
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className={`w-full bg-green-600 text-white py-2 rounded transition ${
          loading ? 'opacity-50 scale-95 cursor-not-allowed' : 'hover:bg-green-700'
        }`}
      >
        {loading ? 'Submitting...' : 'Submit Trade'}
      </button>
    </form>
  );
};

export default TradeForm;