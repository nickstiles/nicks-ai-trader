import React, { useState, useEffect } from "react";
import { getStatus, postPrediction } from "./services/api";
import StatusCard from "./components/StatusCard";
import PredictForm from "./components/PredictForm";

function App() {
  const [status, setStatus] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    getStatus().then(setStatus);
  }, []);

  const handlePredict = async (ticker, prices, timeframe) => {
    console.log("\uD83D\uDCE4 Sending prediction request:", { ticker, prices, timeframe });
    const result = await postPrediction(ticker, prices, timeframe);
    console.log("\uD83D\uDCE5 Received prediction result:", result);
    setPrediction(result);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-2xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold text-center text-blue-600 mb-4">
          AI Stock Trading Dashboard
        </h1>

        <StatusCard status={status} />
        <PredictForm onSubmit={handlePredict} />

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
