import React from "react";
import Plot from "react-plotly.js";

const MetricDisplay = ({ label, value, suffix = "", highlight = false }) => (
  <div
    className={`text-center p-3 rounded-lg ${
      highlight ? "bg-green-50 border border-green-200" : "bg-gray-50"
    }`}
  >
    <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
    <div
      className={`text-lg font-bold ${
        highlight ? "text-green-600" : "text-gray-700"
      }`}
    >
      {typeof value === "number" ? value.toFixed(2) : value}
      {suffix}
    </div>
  </div>
);

const PredictionComparisonChart = ({ comparisonData, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Prediction Comparison
        </h3>
        <div className="h-80 bg-gray-100 rounded-lg animate-pulse"></div>
      </div>
    );
  }

  if (!comparisonData) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Prediction Comparison
        </h3>
        <p className="text-gray-500 text-center py-8">
          Edit shape functions and apply changes to see prediction comparison.
        </p>
      </div>
    );
  }

  const {
    original_predictions,
    interactive_predictions,
    actual_values,
    metrics,
  } = comparisonData;

  // Check if there are any differences
  const hasDifferences = original_predictions.some(
    (orig, i) => Math.abs(orig - interactive_predictions[i]) > 0.001,
  );

  // Calculate improvement
  const rmseImproved = metrics.interactive_rmse < metrics.original_rmse;
  const maeImproved = metrics.interactive_mae < metrics.original_mae;

  // Create sample indices for x-axis
  const indices = actual_values.map((_, i) => i);

  // Scatter plot data
  const data = [
    {
      x: actual_values,
      y: original_predictions,
      type: "scatter",
      mode: "markers",
      name: "IGANN (Original)",
      marker: {
        color: "#3b82f6",
        size: 6,
        opacity: 0.6,
      },
    },
    ...(hasDifferences
      ? [
          {
            x: actual_values,
            y: interactive_predictions,
            type: "scatter",
            mode: "markers",
            name: "IGANN Interactive (Edited)",
            marker: {
              color: "#10b981",
              size: 6,
              opacity: 0.6,
            },
          },
        ]
      : []),
    {
      x: [Math.min(...actual_values), Math.max(...actual_values)],
      y: [Math.min(...actual_values), Math.max(...actual_values)],
      type: "scatter",
      mode: "lines",
      name: "Perfect Prediction",
      line: {
        color: "#ef4444",
        width: 2,
        dash: "dash",
      },
    },
  ];

  const layout = {
    title: {
      text: hasDifferences
        ? "Predictions vs Actual (Original vs Edited)"
        : "Predictions vs Actual",
      font: { size: 14, color: "#374151" },
    },
    xaxis: {
      title: "Actual Bike Rentals",
      gridcolor: "#e5e7eb",
    },
    yaxis: {
      title: "Predicted Bike Rentals",
      gridcolor: "#e5e7eb",
    },
    margin: { l: 60, r: 30, t: 50, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    height: 350,
    legend: {
      x: 0.02,
      y: 0.98,
      bgcolor: "rgba(255,255,255,0.9)",
    },
    hovermode: "closest",
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  return (
    <div className="card">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-700">
          Prediction Comparison
        </h3>
        <p className="text-sm text-gray-500">
          {hasDifferences
            ? "Compare predictions between original IGANN model and your edited version."
            : "Edit shape functions to see how predictions change."}
        </p>
      </div>

      {/* Metrics Comparison */}
      <div className="mb-6">
        <div className="grid grid-cols-2 gap-6">
          {/* Original IGANN Metrics */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-blue-600 mb-3 flex items-center">
              <span className="w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
              IGANN (Original)
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <MetricDisplay label="RMSE" value={metrics.original_rmse} />
              <MetricDisplay label="MAE" value={metrics.original_mae} />
            </div>
          </div>

          {/* Interactive IGANN Metrics */}
          <div
            className={`border rounded-lg p-4 ${
              hasDifferences
                ? "border-green-300 bg-green-50/30"
                : "border-gray-200"
            }`}
          >
            <h4
              className={`text-sm font-semibold mb-3 flex items-center ${
                hasDifferences ? "text-green-600" : "text-gray-600"
              }`}
            >
              <span
                className={`w-3 h-3 rounded-full mr-2 ${
                  hasDifferences ? "bg-green-500" : "bg-gray-400"
                }`}
              ></span>
              IGANN Interactive (Edited)
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <MetricDisplay
                label="RMSE"
                value={metrics.interactive_rmse}
                highlight={rmseImproved && hasDifferences}
              />
              <MetricDisplay
                label="MAE"
                value={metrics.interactive_mae}
                highlight={maeImproved && hasDifferences}
              />
            </div>
          </div>
        </div>

        {/* Improvement Summary */}
        {hasDifferences && (
          <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <div className="text-sm">
              <strong>Summary:</strong>
              <span className="ml-2">
                RMSE{" "}
                <span
                  className={rmseImproved ? "text-green-600" : "text-red-600"}
                >
                  {rmseImproved ? "improved" : "worsened"} by{" "}
                  {Math.abs(
                    metrics.original_rmse - metrics.interactive_rmse,
                  ).toFixed(4)}
                </span>
                {" | "}
                MAE{" "}
                <span
                  className={maeImproved ? "text-green-600" : "text-red-600"}
                >
                  {maeImproved ? "improved" : "worsened"} by{" "}
                  {Math.abs(
                    metrics.original_mae - metrics.interactive_mae,
                  ).toFixed(4)}
                </span>
              </span>
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Format: igann: rmse, mae = ({metrics.original_rmse.toFixed(5)},{" "}
              {metrics.original_mae.toFixed(5)}) | igann_interactive: rmse, mae
              = ({metrics.interactive_rmse.toFixed(5)},{" "}
              {metrics.interactive_mae.toFixed(5)})
            </div>
          </div>
        )}
      </div>

      {/* Scatter Plot */}
      <Plot data={data} layout={layout} config={config} className="w-full" />
    </div>
  );
};

export default PredictionComparisonChart;
