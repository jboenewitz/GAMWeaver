import React, { useState, useEffect } from "react";
import apiService from "../api/apiService";
import Plot from "react-plotly.js";

function CombinedResultsPage({ onBack, onResetDatabase }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [users, setUsers] = useState([]);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [comparison, usersData] = await Promise.all([
        apiService.getCombinedPredictionsComparison(),
        apiService.getUsersWithEdits(),
      ]);
      setComparisonData(comparison);
      setUsers(usersData.users || []);
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Failed to load data"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleResetDatabase = async () => {
    setResetting(true);
    try {
      await onResetDatabase();
      setShowResetConfirm(false);
      await fetchData();
    } catch (err) {
      setError(err.message || "Failed to reset database");
    } finally {
      setResetting(false);
    }
  };

  const renderMetricsComparison = () => {
    if (!comparisonData?.metrics) return null;

    const { metrics } = comparisonData;
    const rmseImprovement = (
      ((metrics.original_rmse - metrics.interactive_rmse) /
        metrics.original_rmse) *
      100
    ).toFixed(1);
    const maeImprovement = (
      ((metrics.original_mae - metrics.interactive_mae) /
        metrics.original_mae) *
      100
    ).toFixed(1);

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Original Model
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">RMSE:</span>
              <span className="font-medium">
                {metrics.original_rmse.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">MAE:</span>
              <span className="font-medium">
                {metrics.original_mae.toFixed(4)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Combined User Edits
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-600">RMSE:</span>
              <span className="font-medium">
                {metrics.interactive_rmse.toFixed(4)}
              </span>
              <span
                className={`text-sm ${
                  parseFloat(rmseImprovement) > 0
                    ? "text-green-600"
                    : "text-red-600"
                }`}
              >
                ({rmseImprovement}%)
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">MAE:</span>
              <span className="font-medium">
                {metrics.interactive_mae.toFixed(4)}
              </span>
              <span
                className={`text-sm ${
                  parseFloat(maeImprovement) > 0
                    ? "text-green-600"
                    : "text-red-600"
                }`}
              >
                ({maeImprovement}%)
              </span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderPredictionChart = () => {
    if (!comparisonData) return null;

    const { original_predictions, interactive_predictions, actual_values } =
      comparisonData;
    const indices = Array.from({ length: actual_values.length }, (_, i) => i);

    return (
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Predictions Over Time (Combined User Edits)
        </h3>
        <Plot
          data={[
            {
              x: indices,
              y: actual_values,
              type: "scatter",
              mode: "lines",
              name: "Actual",
              line: { color: "#6366f1", width: 2 },
            },
            {
              x: indices,
              y: original_predictions,
              type: "scatter",
              mode: "lines",
              name: "Original Model",
              line: { color: "#f59e0b", width: 2, dash: "dot" },
            },
            {
              x: indices,
              y: interactive_predictions,
              type: "scatter",
              mode: "lines",
              name: "Combined Edits",
              line: { color: "#10b981", width: 2 },
            },
          ]}
          layout={{
            autosize: true,
            height: 400,
            margin: { l: 50, r: 30, t: 20, b: 50 },
            xaxis: { title: "Sample Index" },
            yaxis: { title: "Bike Rentals" },
            legend: { orientation: "h", y: -0.15 },
            hovermode: "x unified",
          }}
          config={{ responsive: true }}
          style={{ width: "100%" }}
        />
      </div>
    );
  };

  const renderScatterPlot = () => {
    if (!comparisonData) return null;

    const { original_predictions, interactive_predictions, actual_values } =
      comparisonData;

    // Calculate min/max for the diagonal line
    const allValues = [
      ...actual_values,
      ...original_predictions,
      ...interactive_predictions,
    ];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    return (
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Predicted vs Actual (Scatter Plot)
        </h3>
        <Plot
          data={[
            // Perfect prediction line
            {
              x: [minVal, maxVal],
              y: [minVal, maxVal],
              type: "scatter",
              mode: "lines",
              name: "Perfect Prediction",
              line: { color: "#d1d5db", width: 2, dash: "dash" },
            },
            // Original predictions
            {
              x: actual_values,
              y: original_predictions,
              type: "scatter",
              mode: "markers",
              name: "Original Model",
              marker: { color: "#f59e0b", size: 6, opacity: 0.6 },
            },
            // Combined predictions
            {
              x: actual_values,
              y: interactive_predictions,
              type: "scatter",
              mode: "markers",
              name: "Combined Edits",
              marker: { color: "#10b981", size: 6, opacity: 0.6 },
            },
          ]}
          layout={{
            autosize: true,
            height: 400,
            margin: { l: 60, r: 30, t: 20, b: 60 },
            xaxis: { title: "Actual Bike Rentals" },
            yaxis: { title: "Predicted Bike Rentals" },
            legend: { orientation: "h", y: -0.2 },
            hovermode: "closest",
          }}
          config={{ responsive: true }}
          style={{ width: "100%" }}
        />
        <p className="text-sm text-gray-500 mt-2 text-center">
          Points closer to the diagonal line indicate better predictions.
          Compare how the combined user edits affect prediction accuracy.
        </p>
      </div>
    );
  };

  const renderCombinedShapeFunctions = () => {
    if (!comparisonData?.combined_shape_functions_display?.length) {
      return (
        <div className="bg-white rounded-xl shadow-md p-6 text-center text-gray-500">
          No shape function data available.
        </div>
      );
    }

    // Check if there are any edits
    const hasEdits = comparisonData.combined_shape_functions?.length > 0;

    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Shape Functions: Original vs Combined Edits
        </h3>
        {!hasEdits && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-700 rounded-lg">
            No user edits have been made yet. The charts below show the original
            shape functions.
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {comparisonData.combined_shape_functions_display.map((sf, idx) => {
            const isNumeric = sf.feature_type === "numeric";
            const hasChanges = sf.y_values.some(
              (y, i) => Math.abs(y - sf.original_y_values[i]) > 0.0001
            );

            return (
              <div
                key={idx}
                className={`border rounded-lg p-4 ${
                  hasChanges
                    ? "border-green-300 bg-green-50"
                    : "border-gray-200"
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-700">
                    {sf.feature_name}
                  </h4>
                  {hasChanges && (
                    <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
                      Modified
                    </span>
                  )}
                </div>
                <div style={{ minHeight: 200 }}>
                  <Plot
                    data={
                      isNumeric
                        ? [
                            {
                              x: sf.x_values,
                              y: sf.original_y_values,
                              type: "scatter",
                              mode: "lines",
                              name: "Original",
                              line: { color: "#9ca3af", width: 2, dash: "dot" },
                            },
                            {
                              x: sf.x_values,
                              y: sf.y_values,
                              type: "scatter",
                              mode: "lines",
                              name: "Combined",
                              line: { color: "#10b981", width: 2 },
                            },
                          ]
                        : [
                            {
                              x: sf.x_values,
                              y: sf.original_y_values,
                              type: "bar",
                              name: "Original",
                              marker: { color: "#9ca3af" },
                              opacity: 0.7,
                            },
                            {
                              x: sf.x_values,
                              y: sf.y_values,
                              type: "bar",
                              name: "Combined",
                              marker: { color: "#10b981" },
                            },
                          ]
                    }
                    layout={{
                      autosize: true,
                      height: 200,
                      margin: { l: 40, r: 10, t: 10, b: 40 },
                      xaxis: {
                        title: { text: sf.feature_name, font: { size: 10 } },
                        tickangle: isNumeric ? 0 : -45,
                        tickfont: { size: 9 },
                      },
                      yaxis: {
                        title: { text: "Effect", font: { size: 10 } },
                        tickfont: { size: 9 },
                      },
                      legend: {
                        orientation: "h",
                        y: -0.3,
                        font: { size: 9 },
                      },
                      showlegend: true,
                      barmode: "group",
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: "100%" }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderEditsSummary = () => {
    if (!comparisonData?.combined_shape_functions?.length) {
      return null;
    }

    return (
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Edit Points Summary
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {comparisonData.combined_shape_functions.map((sf, idx) => (
            <div key={idx} className="border rounded-lg p-4 bg-gray-50">
              <h4 className="font-medium text-gray-700 mb-2">
                {sf.feature_name}
                <span className="ml-2 text-sm text-gray-500">
                  ({sf.edited_points.length} points)
                </span>
              </h4>
              <div className="text-sm text-gray-500 space-y-1 max-h-32 overflow-y-auto">
                {sf.edited_points.slice(0, 5).map((point, pidx) => (
                  <div key={pidx} className="flex justify-between text-xs">
                    <span className="font-mono">
                      x:{" "}
                      {typeof point.x_value === "number"
                        ? point.x_value.toFixed(2)
                        : point.x_value}
                    </span>
                    <span
                      className={`font-mono ${
                        point.y_value >= 0 ? "text-green-600" : "text-red-600"
                      }`}
                    >
                      {point.y_value >= 0 ? "+" : ""}
                      {point.y_value.toFixed(3)}
                    </span>
                    <span className="text-gray-400">
                      {point.user_count} user{point.user_count > 1 ? "s" : ""}
                    </span>
                  </div>
                ))}
                {sf.edited_points.length > 5 && (
                  <div className="text-gray-400 italic text-xs">
                    ... and {sf.edited_points.length - 5} more points
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderUsersList = () => {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Participating Users ({users.length})
        </h3>
        {users.length === 0 ? (
          <p className="text-gray-500">No users yet.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {users.map((user) => (
              <span
                key={user.id}
                className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm"
              >
                {user.name}
              </span>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={onBack}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <svg
                  className="w-6 h-6 text-gray-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 19l-7-7 7-7"
                  />
                </svg>
              </button>
              <h1 className="text-xl font-bold text-gray-800">
                Combined Results - All Users
              </h1>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={fetchData}
                disabled={loading}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50"
              >
                Refresh
              </button>
              <button
                onClick={() => setShowResetConfirm(true)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Reset Database
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <svg
                className="animate-spin h-10 w-10 text-primary-600 mx-auto mb-4"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <p className="text-gray-600">Loading combined results...</p>
            </div>
          </div>
        ) : (
          <>
            {/* Summary Card */}
            <div className="bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl shadow-md p-6 mb-6 text-white">
              <h2 className="text-2xl font-bold mb-2">
                Combined Analysis from{" "}
                {comparisonData?.total_users_with_edits || 0} Users
              </h2>
              <p className="opacity-90">
                This page shows the aggregated effect of all user edits on the
                GAM model. Each point's offset is averaged across all users who
                edited it.
              </p>
            </div>

            {/* Users List */}
            {renderUsersList()}

            {/* Metrics Comparison */}
            {renderMetricsComparison()}

            {/* Scatter Plot - Predicted vs Actual */}
            {renderScatterPlot()}

            {/* Prediction Chart - Time Series */}
            {renderPredictionChart()}

            {/* Combined Shape Functions Visualization */}
            {renderCombinedShapeFunctions()}

            {/* Edit Points Summary */}
            {renderEditsSummary()}
          </>
        )}
      </main>

      {/* Reset Confirmation Modal */}
      {showResetConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-bold text-gray-800 mb-4">
              Reset Database?
            </h3>
            <p className="text-gray-600 mb-6">
              This will permanently delete all users and their edits. This
              action cannot be undone.
            </p>
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setShowResetConfirm(false)}
                disabled={resetting}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleResetDatabase}
                disabled={resetting}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50 flex items-center"
              >
                {resetting ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2 h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    Resetting...
                  </>
                ) : (
                  "Yes, Reset"
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default CombinedResultsPage;
