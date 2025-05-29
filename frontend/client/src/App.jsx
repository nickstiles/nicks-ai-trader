import React, { useState, useEffect } from "react";
import {
  getStatus,
  postPrediction,
  submitTrade,
  postBacktest
} from "./services/api";
import StatusCard from "./components/StatusCard";
import PredictForm from "./components/PredictForm";
import TradeForm from "./components/TradeForm";
import BacktestForm from "./components/BacktestForm";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from "recharts";

function App() {
  const [isDark, setIsDark] = useState(true);
  const [status, setStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [orderResponse, setOrderResponse] = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);

  useEffect(() => {
    getStatus().then(setStatus);
  }, []);

  const handlePredict = async (ticker, timeframe) => {
    console.log("📤 Sending prediction request:", { ticker, timeframe });
    const result = await postPrediction(ticker, [], timeframe);
    console.log("📥 Received prediction result:", result);
    setPrediction(result);
  };

  const handleTrade = async (order) => {
    console.log("📤 Sending trade:", order);
    const resp = await submitTrade(order);
    console.log("📥 Trade response:", resp);
    setOrderResponse(resp);
  };

  const handleBacktest = async (params) => {
    console.log("📊 Running backtest:", params);
    const res = await postBacktest(params);
    console.log("📈 Backtest result:", res);
    setBacktestResult(res);
  };

  return (
    <div className={isDark ? "dark" : ""}>
      <div className="relative min-h-screen bg-gray-100 dark:bg-gray-900 p-6">
        <button
          onClick={() => setIsDark((d) => !d)}
          className="absolute top-4 right-4 p-2 bg-gray-200 dark:bg-gray-800 text-gray-800 dark:text-gray-200 rounded"
        >
          {isDark ? "Light Mode" : "Dark Mode"}
        </button>
        <div className="max-w-3xl mx-auto space-y-8">
          <h1 className="text-4xl font-bold text-center text-blue-600 dark:text-blue-400 mb-4">
            AI Stock Trading Dashboard
          </h1>

          <StatusCard status={status} />

          <PredictForm onSubmit={handlePredict} />

          {prediction && (
            <div className="p-4 bg-green-50 dark:bg-green-800 border border-green-300 dark:border-green-700 rounded-lg shadow">
              <h3 className="text-xl font-semibold text-green-700 dark:text-green-300">
                Prediction Result
              </h3>
              <p>
                <strong>Signal:</strong> {prediction.signal}
              </p>
              <p>
                <strong>Confidence:</strong> {prediction.confidence}
              </p>
            </div>
          )}

          <TradeForm onSubmit={handleTrade} />

          {orderResponse && (
            <div className="p-4 bg-blue-50 dark:bg-blue-800 border border-blue-300 dark:border-blue-700 rounded-lg shadow">
              <h3 className="text-xl font-semibold text-blue-700 dark:text-blue-300">
                Order Executed
              </h3>
              <p>
                <strong>Order ID:</strong> {orderResponse.order_id}
              </p>
              <p>
                <strong>Status:</strong> {orderResponse.status}
              </p>
            </div>
          )}

          <BacktestForm onSubmit={handleBacktest} />

          {backtestResult && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow-md mt-6">
              <h3 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-200">
                Backtest Equity Curve
              </h3>
              <LineChart
                width={700}
                height={300}
                data={backtestResult.portfolio_values.map((v, i) => ({ index: i, value: v }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#888888" />
                <XAxis dataKey="index" stroke="#888888" />
                <YAxis stroke="#888888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? "#1F2937" : "#fff",
                    borderColor: isDark ? "#374151" : "#ccc",
                    color: isDark ? "#fff" : "#000"
                  }}
                />
                <Line type="monotone" dataKey="value" stroke="#8884d8" dot={false} />
              </LineChart>
              <div className="mt-4 space-y-1 text-gray-700 dark:text-gray-300">
                <p>
                  <strong>Total Return:</strong> {(backtestResult.total_return * 100).toFixed(2)}%
                </p>
                <p>
                  <strong>Trades:</strong> {backtestResult.num_trades}
                </p>
                <p>
                  <strong>Win Rate:</strong> {(backtestResult.win_rate * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
