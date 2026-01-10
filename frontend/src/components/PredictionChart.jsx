import React from "react";
import Plot from "react-plotly.js";

const PredictionChart = ({ predictionsData, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Predictions vs Actual
        </h3>
        <div className="h-80 bg-gray-100 rounded-lg animate-pulse"></div>
      </div>
    );
  }

  if (!predictionsData) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          📈 Predictions vs Actual
        </h3>
        <p className="text-gray-500 text-center py-8">
          Train the model to see predictions compared to actual values.
        </p>
      </div>
    );
  }

  const { predicted, actual } = predictionsData;

  // Calculate max for diagonal line
  const maxVal = Math.max(...predicted, ...actual);
  const minVal = Math.min(...predicted, ...actual);

  const data = [
    {
      x: actual,
      y: predicted,
      mode: "markers",
      type: "scatter",
      marker: {
        color: "#3b82f6",
        size: 6,
        opacity: 0.6,
      },
      name: "Predictions",
    },
    {
      x: [minVal, maxVal],
      y: [minVal, maxVal],
      mode: "lines",
      type: "scatter",
      line: {
        color: "#ef4444",
        width: 2,
        dash: "dash",
      },
      name: "Perfect Prediction",
    },
  ];

  const layout = {
    title: {
      text: "Predicted vs Actual Bike Rentals",
      font: { size: 16, color: "#374151" },
    },
    xaxis: {
      title: "Actual Count",
      gridcolor: "#e5e7eb",
    },
    yaxis: {
      title: "Predicted Count",
      gridcolor: "#e5e7eb",
    },
    margin: { l: 60, r: 30, t: 50, b: 60 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    height: 400,
    showlegend: true,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: "rgba(255,255,255,0.8)",
    },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  return (
    <div className="card">
      <Plot data={data} layout={layout} config={config} className="w-full" />
    </div>
  );
};

export default PredictionChart;
