import React, { useState, useEffect } from "react";
import apiService from "../api/apiService";
import Plot from "react-plotly.js";

const CombinedPredictionForm = ({ modelTrained }) => {
  const [formData, setFormData] = useState({
    temperature: 20,
    humidity: 50,
    windspeed: 10,
    time_of_day: 12,
    type_of_day: "Working Day",
    weathersituation: "Clear",
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? parseFloat(value) : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.predictCombined(formData);
      if (result) {
        setPrediction(result.predicted_count);
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Prediction failed",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md p-6 mb-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-2">
        Predict with Combined Edits
      </h3>
      <p className="text-sm text-gray-500 mb-4">
        Make a prediction using the model with all combined user edits applied.
        The result reflects the aggregated shape function modifications from all
        users.
      </p>

      {!modelTrained && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
          Please train the model first before making predictions.
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Temperature (°C)
            </label>
            <input
              type="number"
              name="temperature"
              value={formData.temperature}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              min="-10"
              max="40"
              step="0.5"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Humidity (%)
            </label>
            <input
              type="number"
              name="humidity"
              value={formData.humidity}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              min="0"
              max="100"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Windspeed (km/h)
            </label>
            <input
              type="number"
              name="windspeed"
              value={formData.windspeed}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              min="0"
              max="70"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Time of Day (Hour)
            </label>
            <input
              type="number"
              name="time_of_day"
              value={formData.time_of_day}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              min="0"
              max="23"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Type of Day
            </label>
            <select
              name="type_of_day"
              value={formData.type_of_day}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
            >
              <option value="Working Day">Working Day</option>
              <option value="Weekend">Weekend</option>
              <option value="Holiday">Holiday</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Weather
            </label>
            <select
              name="weathersituation"
              value={formData.weathersituation}
              onChange={handleChange}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
            >
              <option value="Clear">Clear</option>
              <option value="Cloudy">Cloudy</option>
              <option value="Light Rain">Light Rain</option>
              <option value="Heavy Rain">Heavy Rain</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={!modelTrained || loading}
          className="w-full px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          {loading ? "Predicting..." : "Predict Bike Rentals (Combined Model)"}
        </button>
      </form>

      {prediction !== null && (
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600">
              Predicted Bike Rentals (Combined Edits)
            </div>
            <div className="text-4xl font-bold text-primary-600 mt-1">
              {Math.round(prediction)}
            </div>
            <div className="text-sm text-gray-500 mt-1">bikes per hour</div>
          </div>
        </div>
      )}
    </div>
  );
};

function CombinedResultsPage({ onBack, onResetDatabase, currentUser }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [users, setUsers] = useState([]);
  const [editLogs, setEditLogs] = useState(null);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [expandedFeatures, setExpandedFeatures] = useState({});

  // Per-user shape function overlay state
  const [showUserOverlay, setShowUserOverlay] = useState(false);
  const [perUserShapeFunctions, setPerUserShapeFunctions] = useState(null);
  const [loadingOverlay, setLoadingOverlay] = useState(false);
  const [useWeighting, setUseWeighting] = useState(true);
  const [unweightedComparisonData, setUnweightedComparisonData] =
    useState(null);

  const USER_COLORS = [
    "#6366f1",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#f97316",
    "#06b6d4",
    "#ec4899",
    "#84cc16",
    "#14b8a6",
    "#a855f7",
  ];

  // Delete edit modal state
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [editToDelete, setEditToDelete] = useState(null);
  const [deleteReason, setDeleteReason] = useState("");
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState("");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Always fetch weighted data; also re-fetch unweighted if it was already loaded
      const fetches = [
        apiService.getCombinedPredictionsComparison(true),
        apiService.getUsersWithEdits(),
        apiService.getEditLogs(),
      ];
      if (unweightedComparisonData !== null) {
        fetches.push(apiService.getCombinedPredictionsComparison(false));
      }
      const results = await Promise.all(fetches);
      setComparisonData(results[0]);
      setUsers(results[1].users || []);
      setEditLogs(results[2]);
      if (unweightedComparisonData !== null) {
        setUnweightedComparisonData(results[3]);
      }
      // Refresh per-user overlay data if visible
      if (showUserOverlay) {
        try {
          const perUser =
            await apiService.getPerUserShapeFunctions(useWeighting);
          setPerUserShapeFunctions(perUser.users || []);
        } catch (_) {
          // silently fail — overlay will show stale data
        }
      } else {
        setPerUserShapeFunctions(null);
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Failed to load data",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleOverlayToggle = async () => {
    const next = !showUserOverlay;
    setShowUserOverlay(next);
    if (next) {
      setLoadingOverlay(true);
      try {
        const data = await apiService.getPerUserShapeFunctions(useWeighting);
        setPerUserShapeFunctions(data.users || []);
      } catch (_) {
        // silently fail
      } finally {
        setLoadingOverlay(false);
      }
    }
  };

  const handleWeightingToggle = async () => {
    const next = !useWeighting;
    setUseWeighting(next);
    setLoadingOverlay(true);
    try {
      // Always re-fetch per-user overlay with new weighting, and lazily fetch
      // unweighted combined data on first toggle-off
      const fetches = [];
      if (showUserOverlay) {
        fetches.push(
          apiService
            .getPerUserShapeFunctions(next)
            .then((data) => setPerUserShapeFunctions(data.users || [])),
        );
      }
      if (!next && !unweightedComparisonData) {
        fetches.push(
          apiService
            .getCombinedPredictionsComparison(false)
            .then((data) => setUnweightedComparisonData(data)),
        );
      }
      await Promise.all(fetches);
    } catch (_) {
      // silently fail
    } finally {
      setLoadingOverlay(false);
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

  const handleOpenDeleteModal = (edit) => {
    setEditToDelete(edit);
    setDeleteReason("");
    setDeleteError("");
    setShowDeleteModal(true);
  };

  const handleDeleteEdit = async () => {
    if (!deleteReason.trim()) {
      setDeleteError("Please provide a reason for deleting this edit.");
      return;
    }

    setDeleting(true);
    setDeleteError("");
    try {
      await apiService.deleteEdit(
        editToDelete.edit_id,
        currentUser?.id,
        deleteReason.trim(),
      );
      setShowDeleteModal(false);
      setEditToDelete(null);
      setDeleteReason("");
      await fetchData();
    } catch (err) {
      setDeleteError(
        err.response?.data?.detail || err.message || "Failed to delete edit",
      );
    } finally {
      setDeleting(false);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteModal(false);
    setEditToDelete(null);
    setDeleteReason("");
    setDeleteError("");
  };

  const renderMetricsComparison = () => {
    if (!comparisonData?.metrics) return null;

    const activeData =
      (useWeighting ? comparisonData : unweightedComparisonData) ??
      comparisonData;
    const { metrics } = activeData;
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

  const renderScatterPlot = () => {
    if (!comparisonData) return null;

    const activeData =
      (useWeighting ? comparisonData : unweightedComparisonData) ??
      comparisonData;
    const { original_predictions, interactive_predictions, actual_values } =
      activeData;

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
          Predicted vs Original (Scatter Plot)
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

    const activeData =
      (useWeighting ? comparisonData : unweightedComparisonData) ??
      comparisonData;

    // Check if there are any edits
    const hasEdits = activeData.combined_shape_functions?.length > 0;

    // Build per-user traces lookup: { feature_name: { user_name: y_values } }
    // Used only for the overlay mode (always shows raw individual user edits)
    const userTracesMap = {};
    if (perUserShapeFunctions) {
      perUserShapeFunctions.forEach((userSF) => {
        userSF.shape_functions.forEach((sf) => {
          if (!userTracesMap[sf.feature_name]) {
            userTracesMap[sf.feature_name] = {};
          }
          userTracesMap[sf.feature_name][userSF.user_name] = sf.y_values;
        });
      });
    }

    // The server already computed the correct combined y_values for the active
    // weighting mode — sf.y_values from activeData is always ready to use.
    const getCombinedY = (sf) => sf.y_values;

    const userNames = perUserShapeFunctions
      ? perUserShapeFunctions.map((u) => u.user_name)
      : [];

    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        {/* Section header with toggle buttons */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">
            Shape Functions: Original vs Combined Edits
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleWeightingToggle}
              disabled={loadingOverlay}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                useWeighting
                  ? "bg-amber-500 text-white hover:bg-amber-600"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
              title="Toggle whether the combined line uses confidence-weighted averaging or a simple mean"
            >
              {loadingOverlay && !useWeighting
                ? "Loading..."
                : useWeighting
                  ? "Weighting: On"
                  : "Weighting: Off"}
            </button>
            <button
              onClick={handleOverlayToggle}
              disabled={loadingOverlay}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                showUserOverlay
                  ? "bg-indigo-600 text-white hover:bg-indigo-700"
                  : "bg-gray-100 text-gray-700 hover:bg-gray-200"
              }`}
            >
              {loadingOverlay && useWeighting
                ? "Loading..."
                : showUserOverlay
                  ? "Hide User Overlay"
                  : "Show User Overlay"}
            </button>
          </div>
        </div>

        {/* Horizontal color legend — visible when overlay is active */}
        {showUserOverlay && !loadingOverlay && (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mb-4 px-3 py-2.5 bg-gray-50 border border-gray-200 rounded-lg text-xs">
            <span className="font-semibold text-gray-500 shrink-0">
              Legend:
            </span>
            {userNames.map((name, i) => (
              <div key={name} className="flex items-center gap-1.5 shrink-0">
                <span
                  className="inline-block rounded-full"
                  style={{
                    width: 28,
                    height: 3,
                    backgroundColor: USER_COLORS[i % USER_COLORS.length],
                  }}
                />
                <span className="text-gray-700">{name}</span>
              </div>
            ))}
            <div className="flex items-center gap-1.5 shrink-0">
              <svg width="28" height="6" style={{ overflow: "visible" }}>
                <line
                  x1="0"
                  y1="3"
                  x2="28"
                  y2="3"
                  stroke="#9ca3af"
                  strokeWidth="2"
                  strokeDasharray="4 2"
                />
              </svg>
              <span className="text-gray-500">Original</span>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span
                className="inline-block rounded-full"
                style={{ width: 28, height: 3, backgroundColor: "#10b981" }}
              />
              <span className="text-gray-500">Combined</span>
            </div>
          </div>
        )}

        {!hasEdits && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-700 rounded-lg">
            No user edits have been made yet. The charts below show the original
            shape functions.
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {activeData.combined_shape_functions_display.map((sf, idx) => {
            const isNumeric = sf.feature_type === "numeric";
            const hasChanges = sf.y_values.some(
              (y, i) => Math.abs(y - sf.original_y_values[i]) > 0.0001,
            );

            let plotData;
            if (showUserOverlay && !loadingOverlay) {
              // Overlay mode: lines for numeric, grouped bars for categorical
              plotData = [];
              // Per-user traces (drawn first, thinner lines / lighter bars)
              userNames.forEach((name, userIdx) => {
                const userY = userTracesMap[sf.feature_name]?.[name];
                if (!userY) return;
                const color = USER_COLORS[userIdx % USER_COLORS.length];
                if (isNumeric) {
                  plotData.push({
                    x: sf.x_values,
                    y: userY,
                    type: "scatter",
                    mode: "lines",
                    name: name,
                    line: { color, width: 1.5 },
                    showlegend: false,
                  });
                } else {
                  plotData.push({
                    x: sf.x_values,
                    y: userY,
                    type: "bar",
                    name: name,
                    marker: { color },
                    opacity: 0.7,
                    showlegend: false,
                  });
                }
              });
              // Original — reference baseline
              if (isNumeric) {
                plotData.push({
                  x: sf.x_values,
                  y: sf.original_y_values,
                  type: "scatter",
                  mode: "lines",
                  name: "Original",
                  line: { color: "#9ca3af", width: 2, dash: "dot" },
                  showlegend: false,
                });
              } else {
                plotData.push({
                  x: sf.x_values,
                  y: sf.original_y_values,
                  type: "bar",
                  name: "Original",
                  marker: { color: "#9ca3af" },
                  opacity: 0.7,
                  showlegend: false,
                });
              }
              // Combined (green bold) — weighted or unweighted average of all users
              if (isNumeric) {
                plotData.push({
                  x: sf.x_values,
                  y: getCombinedY(sf),
                  type: "scatter",
                  mode: "lines",
                  name: "Combined",
                  line: { color: "#10b981", width: 3 },
                  showlegend: false,
                });
              } else {
                plotData.push({
                  x: sf.x_values,
                  y: getCombinedY(sf),
                  type: "bar",
                  name: "Combined",
                  marker: { color: "#10b981" },
                  showlegend: false,
                });
              }
            } else {
              // Normal mode: original vs combined
              plotData = isNumeric
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
                      y: getCombinedY(sf),
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
                      y: getCombinedY(sf),
                      type: "bar",
                      name: "Combined",
                      marker: { color: "#10b981" },
                    },
                  ];
            }

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
                    data={plotData}
                    layout={{
                      autosize: true,
                      height: 200,
                      margin: { l: 55, r: 70, t: 10, b: 40 },
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
                        orientation: "v",
                        x: 1.02,
                        xanchor: "left",
                        y: 1,
                        yanchor: "top",
                        font: { size: 9 },
                      },
                      showlegend: !showUserOverlay,
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

  const renderEditLogs = () => {
    if (!editLogs?.features?.length) {
      return null;
    }

    const toggleFeature = (featureName) => {
      setExpandedFeatures((prev) => ({
        ...prev,
        [featureName]: !prev[featureName],
      }));
    };

    return (
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Edit Logs</h3>
        <p className="text-sm text-gray-500 mb-4">
          Detailed log of all user edits, grouped by feature. Shows who edited,
          their self-reported confidence rating (1-10), raw input value, and the
          weighted result applied to the combined view.
        </p>
        <div className="space-y-4">
          {editLogs.features.map((feature, idx) => {
            const isExpanded = expandedFeatures[feature.feature_name] ?? false;
            const uniqueUsers = [
              ...new Set(feature.edits.map((e) => e.user_name)),
            ];

            return (
              <div key={idx} className="border rounded-lg overflow-hidden">
                {/* Feature Header - Clickable */}
                <button
                  onClick={() => toggleFeature(feature.feature_name)}
                  className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors text-left"
                >
                  <div>
                    <h4 className="font-medium text-gray-700">
                      {feature.feature_name}
                    </h4>
                    <span className="text-sm text-gray-500">
                      {feature.edits.length} edit
                      {feature.edits.length !== 1 ? "s" : ""} by{" "}
                      {uniqueUsers.length} user
                      {uniqueUsers.length !== 1 ? "s" : ""}
                    </span>
                  </div>
                  <svg
                    className={`w-5 h-5 text-gray-400 transform transition-transform ${
                      isExpanded ? "rotate-180" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>

                {/* Expanded Content */}
                {isExpanded && (
                  <div className="p-4 border-t">
                    {/* Table Header */}
                    <div
                      className="grid gap-2 text-xs font-medium text-gray-500 uppercase mb-2 px-2"
                      style={{
                        gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr 3fr 28px",
                      }}
                    >
                      <span>User</span>
                      <span>X Value</span>
                      <span className="text-center">Confidence</span>
                      <span className="text-right">Raw Input</span>
                      <span className="text-right">Weighted</span>
                      <span>Edit Message</span>
                      <span></span>
                    </div>

                    {/* Edit Rows */}
                    <div className="space-y-1 max-h-64 overflow-y-auto">
                      {feature.edits.map((edit, editIdx) => (
                        <div
                          key={editIdx}
                          className="group grid gap-2 text-sm py-2 px-2 bg-gray-50 rounded hover:bg-gray-100"
                          style={{
                            gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr 3fr 28px",
                          }}
                        >
                          <span className="font-medium text-gray-700 truncate">
                            {edit.user_name}
                          </span>
                          <span className="font-mono text-gray-600">
                            {typeof edit.x_value === "number"
                              ? edit.x_value.toFixed(2)
                              : edit.x_value}
                          </span>
                          <span className="text-center">
                            <span
                              className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                                edit.sureness >= 7
                                  ? "bg-green-100 text-green-700"
                                  : edit.sureness >= 4
                                    ? "bg-yellow-100 text-yellow-700"
                                    : "bg-red-100 text-red-700"
                              }`}
                            >
                              {edit.sureness}/10
                            </span>
                          </span>
                          <span
                            className={`text-right font-mono ${
                              edit.raw_input >= 0
                                ? "text-green-600"
                                : "text-red-600"
                            }`}
                          >
                            {edit.raw_input >= 0 ? "+" : ""}
                            {edit.raw_input.toFixed(3)}
                          </span>
                          <span
                            className={`text-right font-mono font-semibold ${
                              edit.weighted_result >= 0
                                ? "text-green-600"
                                : "text-red-600"
                            }`}
                          >
                            {edit.weighted_result >= 0 ? "+" : ""}
                            {edit.weighted_result.toFixed(3)}
                          </span>
                          <span
                            className="text-gray-600 text-xs break-words"
                            title={edit.message}
                          >
                            {edit.message || "-"}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleOpenDeleteModal({
                                ...edit,
                                feature_name: feature.feature_name,
                              });
                            }}
                            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-gray-400 hover:text-red-600 rounded hover:bg-red-50"
                            title="Delete this edit"
                          >
                            <svg
                              className="w-4 h-4"
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                              />
                            </svg>
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
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

            {/* Prediction with Combined Edits */}
            <CombinedPredictionForm modelTrained={true} />

            {/* Combined Shape Functions Visualization */}
            {renderCombinedShapeFunctions()}

            <div className="mt-12"></div>

            {/* Edit Logs */}
            {renderEditLogs()}
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

      {/* Delete Edit Modal */}
      {showDeleteModal && editToDelete && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-bold text-gray-800 mb-2">
              Delete Edit
            </h3>
            <p className="text-gray-600 mb-1 text-sm">
              You are about to delete an edit by{" "}
              <span className="font-semibold">{editToDelete.user_name}</span>.
            </p>
            <p className="text-gray-500 mb-4 text-xs">
              Feature:{" "}
              <span className="font-mono">
                {editToDelete.feature_name || "—"}
              </span>{" "}
              • X Value:{" "}
              <span className="font-mono">
                {typeof editToDelete.x_value === "number"
                  ? editToDelete.x_value.toFixed(2)
                  : editToDelete.x_value}
              </span>
            </p>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Why is this edit being removed?{" "}
              <span className="text-red-500">*</span>
            </label>
            <textarea
              value={deleteReason}
              onChange={(e) => {
                setDeleteReason(e.target.value);
                if (deleteError) setDeleteError("");
              }}
              placeholder="Provide a reason for removing this edit..."
              rows={3}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-red-500 focus:border-red-500 resize-none"
            />
            {deleteError && (
              <p className="text-red-600 text-xs mt-1">{deleteError}</p>
            )}
            <div className="flex justify-end space-x-3 mt-4">
              <button
                onClick={handleCancelDelete}
                disabled={deleting}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteEdit}
                disabled={deleting || !deleteReason.trim()}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50 flex items-center text-sm"
              >
                {deleting ? (
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
                    Deleting...
                  </>
                ) : (
                  <>
                    <svg
                      className="w-4 h-4 mr-1"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                    Delete Edit
                  </>
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
