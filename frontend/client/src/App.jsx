import React, { useState, useEffect } from "react";
import { getStatus, postPrediction, submitTrade } from "./services/api";
import StatusCard from "./components/StatusCard";
import PredictForm from "./components/PredictForm";
import TradeForm from "./components/TradeForm";

function App() {
  const [status, setStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [orderResponse, setOrderResponse] = useState(null);

  useEffect(() => {
    getStatus().then(setStatus);
  }, []);

  const handlePredict = async (ticker, timeframe) => {
    console.log("📤 Sending prediction request:", { ticker, timeframe });
    // send empty recent_prices to trigger live data fetch
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

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold text-center text-blue-600 mb-4">
          AI Stock Trading Dashboard
        </h1>

        <StatusCard status={status} />
        <PredictForm onSubmit={handlePredict} />
        <TradeForm onSubmit={handleTrade} />

        {orderResponse && (
          <div className="p-4 bg-blue-50 border border-blue-300 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-blue-700">Order Executed</h3>
            <p><strong>Order ID:</strong> {orderResponse.order_id}</p>
            <p><strong>Status:</strong> {orderResponse.status}</p>
          </div>
        )}

        {prediction && (
          <div className="p-4 bg-green-50 border border-green-300 rounded-lg shadow">
            <h3 className="text-xl font-semibold text-green-700">Prediction Result</h3>
            <p><strong>Signal:</strong> {prediction.signal}</p>
            <p><strong>Confidence:</strong> {prediction.confidence}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
