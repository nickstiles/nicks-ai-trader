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
    console.log("📤 Sending prediction request:", { ticker, prices, timeframe });
    const result = await postPrediction(ticker, prices, timeframe);
    console.log("📥 Received prediction result:", result);
    setPrediction(result);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>AI Stock Trading Dashboard</h1>
      <StatusCard status={status} />
      <PredictForm onSubmit={handlePredict} />
      {prediction && (
        <div>
          <h3>Prediction</h3>
          <p><strong>Signal:</strong> {prediction.signal}</p>
          <p><strong>Confidence:</strong> {prediction.confidence}</p>
        </div>
      )}
    </div>
  );
}

export default App;