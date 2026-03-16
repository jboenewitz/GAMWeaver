import React, { useState, useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { apiService } from "../api/apiService";

const DEFAULT_COLORS = {
  originalColor: "#3b82f6",
  editedColor: "#10b981",
  lineColor: "#ef4444",
};

const MetricDisplay = ({ label, value, suffix = "", tone = "neutral" }) => {
  const toneClasses =
    tone === "improved"
      ? "bg-green-50 border border-green-200"
      : tone === "worsened"
        ? "bg-red-50 border border-red-200"
        : tone === "unchanged"
          ? "bg-amber-50 border border-amber-200"
          : "bg-gray-50";

  const valueClasses =
    tone === "improved"
      ? "text-green-600"
      : tone === "worsened"
        ? "text-red-600"
        : tone === "unchanged"
          ? "text-amber-700"
          : "text-gray-700";

  return (
    <div className={`text-center p-3 rounded-lg ${toneClasses}`}>
      <div className="text-xs text-gray-500 uppercase tracking-wide">
        {label}
      </div>
      <div className={`text-lg font-bold ${valueClasses}`}>
        {typeof value === "number" ? value.toFixed(2) : value}
        {suffix}
      </div>
    </div>
  );
};

const ColorSwatch = ({ color }) => (
  <span
    className="inline-block w-4 h-4 rounded-sm border border-gray-300 align-middle flex-shrink-0"
    style={{ backgroundColor: color }}
  />
);

const PredictionComparisonChart = ({
  comparisonData,
  loading,
  currentUser,
}) => {
  const [colors, setColors] = useState(DEFAULT_COLORS);
  const [showColorPanel, setShowColorPanel] = useState(false);
  const [draftColors, setDraftColors] = useState(DEFAULT_COLORS);
  const [saving, setSaving] = useState(false);
  const [hoveredSeries, setHoveredSeries] = useState(null);
  const panelRef = useRef(null);

  // Load saved preferences when user is known
  useEffect(() => {
    if (!currentUser?.id) return;
    let cancelled = false;
    apiService
      .getUserPreferences(currentUser.id)
      .then((prefs) => {
        if (cancelled) return;
        const merged = {
          originalColor:
            prefs.chart_original_color ?? DEFAULT_COLORS.originalColor,
          editedColor: prefs.chart_edited_color ?? DEFAULT_COLORS.editedColor,
          lineColor: prefs.chart_line_color ?? DEFAULT_COLORS.lineColor,
        };
        setColors(merged);
        setDraftColors(merged);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [currentUser?.id]);

  // Close panel when clicking outside
  useEffect(() => {
    if (!showColorPanel) return;
    const handleOutside = (e) => {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        setShowColorPanel(false);
        setDraftColors(colors); // discard unsaved draft
      }
    };
    document.addEventListener("mousedown", handleOutside);
    return () => document.removeEventListener("mousedown", handleOutside);
  }, [showColorPanel, colors]);

  const handleSave = async () => {
    setColors(draftColors);
    setShowColorPanel(false);
    if (!currentUser?.id) return;
    setSaving(true);
    try {
      await apiService.updateUserPreferences(currentUser.id, {
        chart_original_color: draftColors.originalColor,
        chart_edited_color: draftColors.editedColor,
        chart_line_color: draftColors.lineColor,
      });
    } catch (_) {
      // silently fail — colors are applied locally regardless
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => setDraftColors(DEFAULT_COLORS);
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

  const hasDifferences = original_predictions.some(
    (orig, i) => Math.abs(orig - interactive_predictions[i]) > 0.001,
  );

  const rmseImproved = metrics.interactive_rmse < metrics.original_rmse;
  const maeImproved = metrics.interactive_mae < metrics.original_mae;

  const originalMarker = {
    color: colors.originalColor,
    size: 6,
    opacity: 0.6,
  };

  const editedMarker = {
    color: colors.editedColor,
    size: 6,
    opacity: 0.6,
  };

  const originalTrace = {
    x: actual_values,
    y: original_predictions,
    type: "scatter",
    mode: "markers",
    name: "IGANN (Original)",
    marker: originalMarker,
  };

  const editedTrace = {
    x: actual_values,
    y: interactive_predictions,
    type: "scatter",
    mode: "markers",
    name: "IGANN Interactive (Edited)",
    marker: editedMarker,
  };

  const perfectLineTrace = {
    x: [Math.min(...actual_values), Math.max(...actual_values)],
    y: [Math.min(...actual_values), Math.max(...actual_values)],
    type: "scatter",
    mode: "lines",
    name: "Perfect Prediction",
    line: { color: colors.lineColor, width: 2, dash: "dash" },
  };

  const rmseTone = !hasDifferences
    ? "neutral"
    : Math.abs(metrics.original_rmse - metrics.interactive_rmse) < 0.000001
      ? "unchanged"
      : rmseImproved
        ? "improved"
        : "worsened";

  const maeTone = !hasDifferences
    ? "neutral"
    : Math.abs(metrics.original_mae - metrics.interactive_mae) < 0.000001
      ? "unchanged"
      : maeImproved
        ? "improved"
        : "worsened";

  let seriesTraces = [originalTrace];
  if (hasDifferences) {
    if (hoveredSeries === "original") {
      seriesTraces = [editedTrace, originalTrace];
    } else if (hoveredSeries === "edited") {
      seriesTraces = [originalTrace, editedTrace];
    } else {
      seriesTraces = [originalTrace, editedTrace];
    }
  }

  const data = [perfectLineTrace, ...seriesTraces];

  const layout = {
    title: {
      text: hasDifferences
        ? "Predictions vs Actual (Original vs Edited)"
        : "Predictions vs Actual",
      font: { size: 14, color: "#374151" },
    },
    xaxis: { title: "Actual Bike Rentals", gridcolor: "#e5e7eb" },
    yaxis: { title: "Predicted Bike Rentals", gridcolor: "#e5e7eb" },
    margin: { l: 60, r: 30, t: 50, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    height: 350,
    legend: { x: 0.02, y: 0.98, bgcolor: "rgba(255,255,255,0.9)" },
    hovermode: "closest",
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };

  const handleHover = (eventData) => {
    if (!eventData?.points?.length) return;
    const traceName = eventData.points[0]?.data?.name;
    if (traceName === "IGANN (Original)") {
      setHoveredSeries("original");
    } else if (traceName === "IGANN Interactive (Edited)") {
      setHoveredSeries("edited");
    }
  };

  const handleUnhover = () => {
    setHoveredSeries(null);
  };

  return (
    <div className="card">
      {/* Header row */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-700">
            Prediction Comparison
          </h3>
          <p className="text-sm text-gray-500">
            {hasDifferences
              ? "Compare predictions between original IGANN model and your edited version."
              : "Edit shape functions to see how predictions change."}
          </p>
        </div>

        {/* Colour picker trigger */}
        <div className="relative flex-shrink-0" ref={panelRef}>
          <button
            onClick={() => {
              setDraftColors(colors);
              setShowColorPanel((v) => !v);
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50 transition-colors"
            title="Customise chart colours"
          >
            <span className="flex gap-0.5 items-center">
              <ColorSwatch color={colors.originalColor} />
              <ColorSwatch color={colors.editedColor} />
              <ColorSwatch color={colors.lineColor} />
            </span>
            Colours
          </button>

          {/* Colour picker dropdown */}
          {showColorPanel && (
            <div className="absolute right-0 top-10 z-50 bg-white border border-gray-200 rounded-xl shadow-xl p-4 w-64">
              <p className="text-sm font-semibold text-gray-700 mb-3">
                Chart Colours
              </p>

              {[
                { key: "originalColor", label: "Original model dots" },
                { key: "editedColor", label: "Edited model dots" },
                { key: "lineColor", label: "Perfect prediction line" },
              ].map(({ key, label }) => (
                <label
                  key={key}
                  className="flex items-center justify-between mb-2 cursor-pointer"
                >
                  <span className="text-sm text-gray-600 flex-1">{label}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400 font-mono">
                      {draftColors[key]}
                    </span>
                    <input
                      type="color"
                      value={draftColors[key]}
                      onChange={(e) =>
                        setDraftColors((prev) => ({
                          ...prev,
                          [key]: e.target.value,
                        }))
                      }
                      className="w-8 h-8 rounded cursor-pointer border border-gray-300 p-0.5"
                    />
                  </div>
                </label>
              ))}

              <div className="flex gap-2 mt-3">
                <button
                  onClick={handleReset}
                  className="flex-1 px-3 py-1.5 border border-gray-300 rounded-lg text-sm text-gray-600 hover:bg-gray-50 transition-colors"
                >
                  Reset
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex-1 px-3 py-1.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors disabled:opacity-50"
                >
                  {saving ? "Saving…" : currentUser ? "Save" : "Apply"}
                </button>
              </div>
              {!currentUser && (
                <p className="text-xs text-gray-400 mt-2 text-center">
                  Log in to persist colours across sessions.
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Metrics Comparison */}
      <div className="mb-6">
        <div className="grid grid-cols-2 gap-6">
          {/* Original IGANN Metrics */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-blue-600 mb-3 flex items-center">
              <span
                className="w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: colors.originalColor }}
              ></span>
              IGANN (Original)
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <MetricDisplay label="RMSE" value={metrics.original_rmse} />
              <MetricDisplay label="MAE" value={metrics.original_mae} />
            </div>
          </div>

          {/* Interactive IGANN Metrics */}
          <div className="border border-gray-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
              <span
                className="w-3 h-3 rounded-full mr-2"
                style={{ backgroundColor: colors.editedColor }}
              ></span>
              IGANN Interactive (Edited)
            </h4>
            <div className="grid grid-cols-2 gap-3">
              <MetricDisplay
                label="RMSE"
                value={metrics.interactive_rmse}
                tone={rmseTone}
              />
              <MetricDisplay
                label="MAE"
                value={metrics.interactive_mae}
                tone={maeTone}
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
      <Plot
        data={data}
        layout={layout}
        config={config}
        className="w-full"
        onHover={handleHover}
        onUnhover={handleUnhover}
      />
    </div>
  );
};

export default PredictionComparisonChart;
