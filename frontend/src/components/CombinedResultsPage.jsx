import React, { useState, useEffect } from "react";
import apiService from "../api/apiService";
import Plot from "react-plotly.js";
import { createTranslator, getDateLocale } from "../i18n";

const roundToTwoDecimals = (value) => Math.round(value * 100) / 100;

const getValidNumericValue = (feature) => {
  const min = Number.isFinite(feature.min_value) ? feature.min_value : undefined;
  const max = Number.isFinite(feature.max_value) ? feature.max_value : undefined;
  let value = Number.isFinite(feature.default_value) ? feature.default_value : 0;

  if (min !== undefined && value < min) {
    value = min;
  }
  if (max !== undefined && value > max) {
    value = max;
  }

  return roundToTwoDecimals(value);
};

const getValidCategoricalValue = (feature) => {
  const options = feature.categorical_options || [];
  if (!options.length) {
    return "";
  }
  return options.includes(feature.default_value)
    ? feature.default_value
    : options[0];
};

const buildInitialFormData = (featureSchema = []) => {
  const initial = {};
  featureSchema.forEach((feature) => {
    if (feature.feature_type === "numeric") {
      initial[feature.name] = getValidNumericValue(feature);
    } else {
      initial[feature.name] = getValidCategoricalValue(feature);
    }
  });
  return initial;
};

const GLASS_CARD_CLASS = "glass-surface-strong p-6 mb-6";
const GLASS_INSET_CLASS = "rounded-xl border border-slate-200/80 bg-white/85";

const formatSignedNumber = (value, digits = 3) =>
  `${value >= 0 ? "+" : ""}${Number(value).toFixed(digits)}`;

const formatXValue = (value) =>
  typeof value === "number" ? value.toFixed(2) : value;

const formatSubmissionDate = (value, language) => {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
  return parsed.toLocaleString(getDateLocale(language));
};

const CombinedPredictionForm = ({
  modelTrained,
  featureSchema,
  targetColumn,
  language = "en",
}) => {
  const t = createTranslator(language);
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setFormData(buildInitialFormData(featureSchema));
    setPrediction(null);
  }, [featureSchema]);

  const handleChange = (feature, value) => {
    setFormData((prev) => ({
      ...prev,
      [feature.name]:
        feature.feature_type === "numeric"
          ? value === ""
            ? ""
            : Number(value)
          : value,
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
        err.response?.data?.detail ||
          err.message ||
          t("combined.predictionFailed"),
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={GLASS_CARD_CLASS}>
      <h3 className="mb-2 text-lg font-semibold text-slate-800">
        {t("combined.formTitle")}
      </h3>
      <p className="mb-4 text-sm text-slate-600">
        {t("combined.formDescription")}
      </p>

      {!modelTrained && (
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50/90 p-3 text-sm text-amber-800">
          {t("prediction.trainFirst")}
        </div>
      )}

      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50/90 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {(!featureSchema || featureSchema.length === 0) && (
        <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50/90 p-3 text-sm text-slate-600">
          {t("prediction.noSchema")}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {(featureSchema || []).map((feature) => (
            <div key={feature.name}>
              <label className="mb-1 block text-sm font-medium text-slate-700">
                {feature.name}
              </label>
              {feature.feature_type === "categorical" ? (
                <select
                  value={formData[feature.name] ?? ""}
                  onChange={(e) => handleChange(feature, e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-white/95 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-500"
                >
                  {(feature.categorical_options || []).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={formData[feature.name] ?? ""}
                  onChange={(e) => handleChange(feature, e.target.value)}
                  className="w-full rounded-lg border border-slate-300 bg-white/95 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-primary-500 focus:ring-2 focus:ring-primary-500"
                  min={Number.isFinite(feature.min_value) ? feature.min_value : undefined}
                  max={Number.isFinite(feature.max_value) ? feature.max_value : undefined}
                  step="any"
                />
              )}
            </div>
          ))}
        </div>

        <button
          type="submit"
          disabled={!modelTrained || loading || !featureSchema?.length}
          className="w-full rounded-lg bg-gradient-to-r from-primary-600 to-cyan-600 px-4 py-2 font-medium text-white shadow-md shadow-primary-800/20 transition-all hover:from-primary-500 hover:to-cyan-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading
            ? t("prediction.predicting")
            : t("combined.predictButton", {
                target: targetColumn || t("common.target"),
              })}
        </button>
      </form>

      {prediction !== null && (
        <div className="mt-6 rounded-lg border border-emerald-200 bg-gradient-to-r from-emerald-50/95 to-sky-50/95 p-4">
          <div className="text-center">
            <div className="text-sm text-slate-600">
              {t("combined.predictedResult", {
                target: targetColumn || t("common.target"),
              })}
            </div>
            <div className="text-4xl font-bold text-primary-600 mt-1">
              {Math.round(prediction)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function CombinedResultsPage({ onBack, currentUser, language = "en" }) {
  const t = createTranslator(language);
  const allowDestructiveActions =
    import.meta.env.VITE_ALLOW_DESTRUCTIVE_ACTIONS === "true" ||
    currentUser?.is_superadmin;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [users, setUsers] = useState([]);
  const [editLogs, setEditLogs] = useState(null);
  const [editLogsError, setEditLogsError] = useState(null);
  const [expandedFeatures, setExpandedFeatures] = useState({});
  const [expandedSubmissions, setExpandedSubmissions] = useState({});

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
    setEditLogsError(null);
    try {
      // Keep requests sequential to avoid overloading the single backend worker.
      // The combined comparison endpoint can be CPU-heavy.
      const statusData = await apiService.getModelStatus();
      setModelStatus(statusData);

      const weightedComparison =
        await apiService.getCombinedPredictionsComparison(true);
      setComparisonData(weightedComparison);

      const usersData = await apiService.getUsersWithEdits();
      setUsers(usersData.users || []);

      // Edit logs are useful but non-critical for page rendering.
      try {
        const logsData = await apiService.getEditLogs();
        setEditLogs(logsData);
      } catch (err) {
        setEditLogs(null);
        setEditLogsError(
          err.response?.data?.detail ||
            err.message ||
            t("combined.loadEditLogsFailed"),
        );
      }

      if (unweightedComparisonData !== null) {
        const unweighted = await apiService.getCombinedPredictionsComparison(false);
        setUnweightedComparisonData(unweighted);
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
        err.response?.data?.detail ||
          err.message ||
          t("combined.loadDataFailed"),
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

  const handleOpenDeleteModal = (edit) => {
    setEditToDelete(edit);
    setDeleteReason("");
    setDeleteError("");
    setShowDeleteModal(true);
  };

  const handleDeleteEdit = async () => {
    if (!allowDestructiveActions) {
      setDeleteError(t("combined.deleteDisabled"));
      return;
    }
    if (!deleteReason.trim()) {
      setDeleteError(t("combined.deleteReasonRequired"));
      return;
    }

    setDeleting(true);
    setDeleteError("");
    try {
      if (editToDelete.persisted_submission_id) {
        await apiService.deleteSubmission(
          editToDelete.persisted_submission_id,
          currentUser?.id,
          deleteReason.trim(),
        );
      } else if (editToDelete.legacy_edit_id) {
        await apiService.deleteEdit(
          editToDelete.legacy_edit_id,
          currentUser?.id,
          deleteReason.trim(),
        );
      } else {
        throw new Error(t("combined.noDeletionTarget"));
      }
      setShowDeleteModal(false);
      setEditToDelete(null);
      setDeleteReason("");
      await fetchData();
    } catch (err) {
      setDeleteError(
        err.response?.data?.detail ||
          err.message ||
          t("combined.deleteFailed"),
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

  const toggleSubmission = (submissionId) => {
    setExpandedSubmissions((prev) => ({
      ...prev,
      [submissionId]: !prev[submissionId],
    }));
  };

  const analyticsAvailable = Boolean(modelStatus?.analytics_available);

  const renderMetricsComparison = () => {
    if (!analyticsAvailable || !comparisonData?.metrics) return null;

    const activeData =
      (useWeighting ? comparisonData : unweightedComparisonData) ??
      comparisonData;
    if (
      !Number.isFinite(activeData?.metrics?.original_rmse) ||
      !Number.isFinite(activeData?.metrics?.original_mae) ||
      !Number.isFinite(activeData?.metrics?.interactive_rmse) ||
      !Number.isFinite(activeData?.metrics?.interactive_mae)
    ) {
      return null;
    }
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
      <div className="mb-6 grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className={GLASS_CARD_CLASS}>
          <h3 className="mb-4 text-lg font-semibold text-slate-800">
            {t("combined.originalModel")}
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-600">{t("combined.rmse")}:</span>
              <span className="font-semibold text-slate-800">
                {metrics.original_rmse.toFixed(4)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-600">{t("combined.mae")}:</span>
              <span className="font-semibold text-slate-800">
                {metrics.original_mae.toFixed(4)}
              </span>
            </div>
          </div>
        </div>

        <div className={GLASS_CARD_CLASS}>
          <h3 className="mb-4 text-lg font-semibold text-slate-800">
            {t("combined.combinedUserEdits")}
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-slate-600">{t("combined.rmse")}:</span>
              <span className="font-semibold text-slate-800">
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
              <span className="text-slate-600">{t("combined.mae")}:</span>
              <span className="font-semibold text-slate-800">
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
    if (!analyticsAvailable || !comparisonData) return null;

    const activeData =
      (useWeighting ? comparisonData : unweightedComparisonData) ??
      comparisonData;
    const { original_predictions, interactive_predictions, actual_values } =
      activeData;
    if (
      !Array.isArray(original_predictions) ||
      !Array.isArray(interactive_predictions) ||
      !Array.isArray(actual_values) ||
      !original_predictions.length ||
      !interactive_predictions.length ||
      !actual_values.length
    ) {
      return null;
    }

    // Calculate min/max for the diagonal line
    const allValues = [
      ...actual_values,
      ...original_predictions,
      ...interactive_predictions,
    ];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    return (
      <div className={GLASS_CARD_CLASS}>
        <h3 className="mb-4 text-lg font-semibold text-slate-800">
          {t("combined.scatterTitle")}
        </h3>
        <Plot
          data={[
            // Perfect prediction line
            {
              x: [minVal, maxVal],
              y: [minVal, maxVal],
              type: "scatter",
              mode: "lines",
              name: t("predictionComparison.perfectPrediction"),
              line: { color: "#d1d5db", width: 2, dash: "dash" },
            },
            // Original predictions
            {
              x: actual_values,
              y: original_predictions,
              type: "scatter",
              mode: "markers",
              name: t("combined.originalModel"),
              marker: { color: "#f59e0b", size: 6, opacity: 0.6 },
            },
            // Combined predictions
            {
              x: actual_values,
              y: interactive_predictions,
              type: "scatter",
              mode: "markers",
              name: t("combined.combinedUserEdits"),
              marker: { color: "#10b981", size: 6, opacity: 0.6 },
            },
          ]}
          layout={{
            autosize: true,
            height: 400,
            margin: { l: 60, r: 30, t: 20, b: 60 },
            xaxis: { title: t("predictionComparison.actualBikeRentals") },
            yaxis: { title: t("predictionComparison.predictedBikeRentals") },
            legend: { orientation: "h", y: -0.2 },
            hovermode: "closest",
          }}
          config={{ responsive: true }}
          style={{ width: "100%" }}
        />
        <p className="mt-2 text-center text-sm text-slate-600">
          {t("combined.scatterDescription")}
        </p>
      </div>
    );
  };

  const renderCombinedShapeFunctions = () => {
    if (!comparisonData?.combined_shape_functions_display?.length) {
      return (
        <div className={`${GLASS_CARD_CLASS} text-center text-slate-500`}>
          {t("combined.shapeFunctionsNone")}
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
      <div className={GLASS_CARD_CLASS}>
        {/* Section header with toggle buttons */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-800">
            {t("combined.shapeFunctionsTitle")}
          </h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleWeightingToggle}
              disabled={loadingOverlay}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                useWeighting
                  ? "bg-amber-500 text-white hover:bg-amber-600"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
              title={t("combined.weightingToggleTitle")}
            >
              {loadingOverlay && !useWeighting
                ? t("common.loading")
                : useWeighting
                  ? t("combined.weightingOn")
                  : t("combined.weightingOff")}
            </button>
            <button
              onClick={handleOverlayToggle}
              disabled={loadingOverlay}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${
                showUserOverlay
                  ? "bg-indigo-600 text-white hover:bg-indigo-700"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              {loadingOverlay && useWeighting
                ? t("common.loading")
                : showUserOverlay
                  ? t("combined.hideUserOverlay")
                  : t("combined.showUserOverlay")}
            </button>
          </div>
        </div>

        {/* Horizontal color legend — visible when overlay is active */}
        {showUserOverlay && !loadingOverlay && (
          <div className={`${GLASS_INSET_CLASS} mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 px-3 py-2.5 text-xs`}>
            <span className="shrink-0 font-semibold text-slate-500">
              {t("combined.legend")}
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
                <span className="text-slate-700">{name}</span>
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
              <span className="text-slate-500">{t("combined.original")}</span>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <span
                className="inline-block rounded-full"
                style={{ width: 28, height: 3, backgroundColor: "#10b981" }}
              />
              <span className="text-slate-500">{t("combined.combined")}</span>
            </div>
          </div>
        )}

        {!hasEdits && (
          <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-700 rounded-lg">
            {t("combined.noEdits")}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {activeData.combined_shape_functions_display.map((sf, idx) => {
            const isNumeric = sf.feature_type === "numeric";
            const xTickLabels =
              Array.isArray(sf.x_tick_labels) &&
              sf.x_tick_labels.length === sf.x_values.length
                ? sf.x_tick_labels
                : null;
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
                  name: t("combined.original"),
                  line: { color: "#9ca3af", width: 2, dash: "dot" },
                  showlegend: false,
                });
              } else {
                plotData.push({
                  x: sf.x_values,
                  y: sf.original_y_values,
                  type: "bar",
                  name: t("combined.original"),
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
                  name: t("combined.combined"),
                  line: { color: "#10b981", width: 3 },
                  showlegend: false,
                });
              } else {
                plotData.push({
                  x: sf.x_values,
                  y: getCombinedY(sf),
                  type: "bar",
                  name: t("combined.combined"),
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
                      name: t("combined.original"),
                      line: { color: "#9ca3af", width: 2, dash: "dot" },
                    },
                    {
                      x: sf.x_values,
                      y: getCombinedY(sf),
                      type: "scatter",
                      mode: "lines",
                      name: t("combined.combined"),
                      line: { color: "#10b981", width: 2 },
                    },
                  ]
                : [
                    {
                      x: sf.x_values,
                      y: sf.original_y_values,
                      type: "bar",
                      name: t("combined.original"),
                      marker: { color: "#9ca3af" },
                      opacity: 0.7,
                    },
                    {
                      x: sf.x_values,
                      y: getCombinedY(sf),
                      type: "bar",
                      name: t("combined.combined"),
                      marker: { color: "#10b981" },
                    },
                  ];
            }

            return (
              <div
                key={idx}
                className={`${GLASS_INSET_CLASS} p-4 ${
                  hasChanges
                    ? "border-emerald-300 bg-emerald-50/95"
                    : ""
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-slate-700">
                    {sf.feature_name}
                  </h4>
                  {hasChanges && (
                    <span className="text-xs px-2 py-1 bg-green-100 text-green-700 rounded-full">
                      {t("combined.modified")}
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
                        ...(!isNumeric && xTickLabels
                          ? {
                              tickmode: "array",
                              tickvals: sf.x_values,
                              ticktext: xTickLabels,
                            }
                          : {}),
                      },
                      yaxis: {
                        title: { text: t("combined.effect"), font: { size: 10 } },
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

  const renderNumericLogPreview = (submission) => {
    const preview = submission.line_preview;
    if (!preview?.x_values?.length) {
      return null;
    }

    const xValues = preview.x_values.map((value) => Number(value));
    const originalYValues = (preview.original_y_values || []).map((value) =>
      Number(value),
    );
    const weightedYValues = (preview.weighted_y_values || []).map((value) =>
      Number(value),
    );
    const finiteYValues = [...originalYValues, ...weightedYValues].filter(
      Number.isFinite,
    );
    const finiteXValues = xValues.filter(Number.isFinite);

    const xMin = finiteXValues.length ? Math.min(...finiteXValues) : undefined;
    const xMax = finiteXValues.length ? Math.max(...finiteXValues) : undefined;
    const yMin = finiteYValues.length ? Math.min(...finiteYValues) : undefined;
    const yMax = finiteYValues.length ? Math.max(...finiteYValues) : undefined;
    const yPadding =
      yMin !== undefined && yMax !== undefined
        ? Math.max((yMax - yMin) * 0.15, 0.5)
        : undefined;

    const sharedLayout = {
      autosize: true,
      height: 240,
      margin: { l: 50, r: 20, t: 10, b: 40 },
      xaxis: {
        tickfont: { size: 9 },
        ...(xMin !== undefined && xMax !== undefined
          ? { range: [xMin, xMax] }
          : {}),
      },
      yaxis: {
        title: { text: t("combined.effect"), font: { size: 10 } },
        tickfont: { size: 9 },
        ...(yMin !== undefined && yMax !== undefined && yPadding !== undefined
          ? { range: [yMin - yPadding, yMax + yPadding] }
          : {}),
      },
      showlegend: false,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.9)",
    };

    return (
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            {t("combined.originalLinePreview")}
          </div>
          <Plot
            data={[
              {
                x: xValues,
                y: originalYValues,
                type: "scatter",
                mode: "lines",
                line: { color: "#94a3b8", width: 2.5 },
              },
            ]}
            layout={sharedLayout}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/70 p-3">
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-emerald-700">
            {t("combined.weightedLinePreview")}
          </div>
          <Plot
            data={[
              {
                x: xValues,
                y: originalYValues,
                type: "scatter",
                mode: "lines",
                line: { color: "#94a3b8", width: 2, dash: "dot" },
              },
              {
                x: xValues,
                y: weightedYValues,
                type: "scatter",
                mode: "lines",
                line: { color: "#10b981", width: 3 },
              },
            ]}
            layout={sharedLayout}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>
      </div>
    );
  };

  const renderEditLogs = () => {
    if (editLogsError) {
      return (
        <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50/95 p-4 text-sm text-amber-800">
          {t("combined.editLogsUnavailable", { error: editLogsError })}
        </div>
      );
    }

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
      <div className={GLASS_CARD_CLASS}>
        <h3 className="mb-4 text-lg font-semibold text-slate-800">
          {t("combined.editLogsTitle")}
        </h3>
        <p className="mb-4 text-sm text-slate-600">
          {t("combined.editLogsDescription")}
        </p>
        <div className="space-y-4">
          {editLogs.features.map((feature, idx) => {
            const isExpanded = expandedFeatures[feature.feature_name] ?? false;
            const uniqueUsers = [
              ...new Set((feature.submissions || []).map((s) => s.user_name)),
            ];
            const submissionCount = (feature.submissions || []).length;

            return (
              <div key={idx} className={`${GLASS_INSET_CLASS} overflow-hidden`}>
                {/* Feature Header - Clickable */}
                <button
                  onClick={() => toggleFeature(feature.feature_name)}
                  className="flex w-full items-center justify-between bg-slate-50/80 p-4 text-left transition-colors hover:bg-slate-100/90"
                >
                  <div>
                    <h4 className="font-medium text-slate-700">
                      {feature.feature_name}
                    </h4>
                    <span className="text-sm text-slate-500">
                      {submissionCount}{" "}
                      {submissionCount === 1
                        ? t("combined.submissionSingular")
                        : t("combined.submissionPlural")}{" "}
                      {t("combined.by")} {uniqueUsers.length}{" "}
                      {uniqueUsers.length === 1
                        ? t("combined.userSingular")
                        : t("combined.userPlural")}
                    </span>
                  </div>
                  <svg
                    className={`w-5 h-5 transform text-slate-400 transition-transform ${
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
                  <div className="space-y-3 border-t border-slate-200/70 p-4">
                    {(feature.submissions || []).map((submission, submissionIdx) => {
                      const submissionKey =
                        submission.submission_id || `${feature.feature_name}-${submissionIdx}`;
                      const isSubmissionExpanded =
                        expandedSubmissions[submissionKey] ?? false;
                      const showNumericPreview =
                        submission.feature_type === "numeric" &&
                        Boolean(submission.line_preview);

                      return (
                        <div
                          key={submissionKey}
                          className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50/60"
                        >
                          <div className="flex items-start gap-3 p-4 transition-colors hover:bg-slate-100/90">
                            <button
                              onClick={() => toggleSubmission(submissionKey)}
                              className="min-w-0 flex-1 text-left"
                            >
                              <div className="grid gap-3 md:grid-cols-6">
                                <div>
                                  <div className="text-xs font-medium uppercase text-slate-500">
                                    {t("combined.user")}
                                  </div>
                                  <div className="truncate font-medium text-slate-700">
                                    {submission.user_name}
                                  </div>
                                  {submission.profession ? (
                                    <div className="truncate text-xs text-slate-500">
                                      {submission.profession}
                                    </div>
                                  ) : null}
                                </div>
                                <div>
                                  <div className="text-xs font-medium uppercase text-slate-500">
                                    {t("combined.submitted")}
                                  </div>
                                  <div className="text-sm text-slate-700">
                                    {formatSubmissionDate(
                                      submission.created_at,
                                      language,
                                    ) || t("common.unknown")}
                                  </div>
                                </div>
                                <div>
                                  <div className="text-xs font-medium uppercase text-slate-500">
                                    {t("combined.confidence")}
                                  </div>
                                  <div>
                                    <span
                                      className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${
                                        submission.sureness >= 7
                                          ? "bg-green-100 text-green-700"
                                          : submission.sureness >= 4
                                            ? "bg-yellow-100 text-yellow-700"
                                            : "bg-red-100 text-red-700"
                                      }`}
                                    >
                                      {submission.sureness}/10
                                    </span>
                                  </div>
                                </div>
                                <div>
                                  <div className="text-xs font-medium uppercase text-slate-500">
                                    {t("combined.points")}
                                  </div>
                                  <div className="text-sm text-slate-700">
                                    {submission.point_count}
                                  </div>
                                </div>
                                <div>
                                  <div className="text-xs font-medium uppercase text-slate-500">
                                    {t("combined.rawTotal")}
                                  </div>
                                  <div
                                    className={`font-mono text-sm ${
                                      submission.raw_input_total >= 0
                                        ? "text-green-600"
                                        : "text-red-600"
                                    }`}
                                  >
                                    {formatSignedNumber(submission.raw_input_total)}
                                  </div>
                                </div>
                                <div>
                                  <div className="text-xs font-medium uppercase text-slate-500">
                                    {t("combined.weightedTotal")}
                                  </div>
                                  <div
                                    className={`font-mono text-sm font-semibold ${
                                      submission.weighted_total >= 0
                                        ? "text-green-600"
                                        : "text-red-600"
                                    }`}
                                  >
                                    {formatSignedNumber(submission.weighted_total)}
                                  </div>
                                </div>
                              </div>
                              <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600">
                                <span>
                                  {t("combined.message")}: {submission.message || "-"}
                                </span>
                                <span>
                                  {t("combined.xSummary")}:{" "}
                                  {submission.x_summary || submission.point_count}
                                </span>
                              </div>
                            </button>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => {
                                  if (allowDestructiveActions) {
                                    handleOpenDeleteModal({
                                      ...submission,
                                      feature_name: feature.feature_name,
                                    });
                                  }
                                }}
                                className={`rounded p-1 transition-colors ${
                                  allowDestructiveActions
                                    ? "text-slate-400 hover:bg-red-50 hover:text-red-600"
                                    : "cursor-not-allowed text-slate-300"
                                }`}
                                title={t("combined.deleteSubmission")}
                                disabled={!allowDestructiveActions}
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
                              <button
                                onClick={() => toggleSubmission(submissionKey)}
                                className="rounded p-1 text-slate-400 transition-colors hover:bg-slate-200/70 hover:text-slate-600"
                              >
                                <svg
                                  className={`h-5 w-5 shrink-0 transform transition-transform ${
                                    isSubmissionExpanded ? "rotate-180" : ""
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
                            </div>
                          </div>

                          {isSubmissionExpanded && (
                            <div className="border-t border-slate-200/70 bg-white/90 p-4">
                              {showNumericPreview ? (
                                renderNumericLogPreview(submission)
                              ) : (
                                <>
                                  <div
                                    className="mb-2 grid gap-2 px-2 text-xs font-medium uppercase text-slate-500"
                                    style={{
                                      gridTemplateColumns: "1fr 1fr 1fr",
                                    }}
                                  >
                                    <span>{t("combined.xValue")}</span>
                                    <span className="text-right">{t("combined.rawInput")}</span>
                                    <span className="text-right">{t("combined.weighted")}</span>
                                  </div>
                                  <div className="space-y-1 max-h-64 overflow-y-auto">
                                    {submission.points.map((point, pointIdx) => (
                                      <div
                                        key={`${submissionKey}-${point.edit_id || pointIdx}`}
                                        className="grid rounded bg-slate-50/90 px-2 py-2 text-sm"
                                        style={{
                                          gridTemplateColumns: "1fr 1fr 1fr",
                                        }}
                                      >
                                        <span className="font-mono text-slate-600">
                                          {formatXValue(point.x_value)}
                                        </span>
                                        <span
                                          className={`text-right font-mono ${
                                            point.raw_input >= 0
                                              ? "text-green-600"
                                              : "text-red-600"
                                          }`}
                                        >
                                          {formatSignedNumber(point.raw_input)}
                                        </span>
                                        <span
                                          className={`text-right font-mono font-semibold ${
                                            point.weighted_result >= 0
                                              ? "text-green-600"
                                              : "text-red-600"
                                          }`}
                                        >
                                          {formatSignedNumber(point.weighted_result)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                    {submissionCount === 0 && (
                      <div className="rounded-lg border border-slate-200 bg-white/80 p-4 text-sm text-slate-500">
                        {t("combined.noSubmissions")}
                      </div>
                    )}
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
      <div className={GLASS_CARD_CLASS}>
        <h3 className="mb-4 text-lg font-semibold text-slate-800">
          {t("combined.participatingUsers", { count: users.length })}
        </h3>
        {users.length === 0 ? (
          <p className="text-slate-500">{t("combined.noUsers")}</p>
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
    <div className="app-ambient-bg relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute -left-16 top-20 h-72 w-72 rounded-full bg-primary-200/35 blur-3xl" />
      <div className="pointer-events-none absolute right-0 top-72 h-80 w-80 rounded-full bg-cyan-200/30 blur-3xl" />
      {/* Header */}
      <header className="relative z-10 pt-5">
        <div className="container mx-auto px-4">
          <div className="glass-surface-strong flex items-center justify-between px-4 py-4 sm:px-6">
            <div className="flex items-center space-x-4">
              <button
                onClick={onBack}
                className="rounded-lg p-2 transition-colors hover:bg-slate-100"
              >
                <svg
                  className="h-6 w-6 text-slate-600"
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
              <h1 className="text-xl font-bold text-slate-800">
                {t("combined.title")}
              </h1>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={fetchData}
                disabled={loading}
                className="rounded-lg bg-gradient-to-r from-primary-600 to-cyan-600 px-4 py-2 text-white shadow-md shadow-primary-800/20 transition-all hover:from-primary-500 hover:to-cyan-500 disabled:opacity-50"
              >
                {t("combined.refresh")}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-4 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50/95 p-4 text-red-700">
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
              <p className="text-slate-600">{t("combined.loading")}</p>
            </div>
          </div>
        ) : (
          <>
            {/* Summary Card */}
            <div className="glass-surface relative mb-6 overflow-hidden border-primary-200/60 bg-gradient-to-r from-primary-600 to-cyan-600 p-6 text-white">
              <div className="pointer-events-none absolute -top-14 right-8 h-40 w-40 rounded-full bg-white/20 blur-3xl" />
              <h2 className="text-2xl font-bold mb-2">
                {t("combined.summaryTitle", {
                  count: comparisonData?.total_users_with_edits || 0,
                })}
              </h2>
              <p className="max-w-3xl opacity-95">
                {t("combined.summaryDescription")}
              </p>
            </div>

            {/* Users List */}
            {renderUsersList()}

            {modelStatus?.model_source === "imported" &&
              !modelStatus?.analytics_available && (
                <div className="mb-6 rounded-lg border border-amber-200 bg-amber-50/95 p-4 text-amber-800">
                  {t("training.importedAnalyticsUnavailable")}
                </div>
              )}

            {/* Metrics Comparison */}
            {renderMetricsComparison()}

            {/* Scatter Plot - Predicted vs Actual */}
            {renderScatterPlot()}

            {/* Prediction with Combined Edits */}
            <CombinedPredictionForm
              modelTrained={Boolean(modelStatus?.is_trained)}
              featureSchema={modelStatus?.feature_schema || []}
              targetColumn={modelStatus?.target_column}
              language={language}
            />

            {/* Combined Shape Functions Visualization */}
            {renderCombinedShapeFunctions()}

            <div className="mt-12"></div>

            {/* Edit Logs */}
            {renderEditLogs()}
          </>
        )}
      </main>

      {/* Delete Edit Modal */}
      {showDeleteModal && editToDelete && allowDestructiveActions && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm">
          <div className="glass-surface-strong mx-4 w-full max-w-md p-6">
            <h3 className="mb-2 text-lg font-bold text-slate-800">
              {t("combined.deleteModalTitle")}
            </h3>
            <p className="mb-1 text-sm text-slate-600">
              {t("combined.deleteModalDescription", {
                user: editToDelete.user_name,
              })}
            </p>
            <p className="mb-4 text-xs text-slate-500">
              {t("combined.feature")}:{" "}
              <span className="font-mono">
                {editToDelete.feature_name || "—"}
              </span>{" "}
              • {t("combined.points")}:{" "}
              <span className="font-mono">
                {editToDelete.point_count || 0}
              </span>
              {` • ${t("combined.xSummary")}: `}
              <span className="font-mono">
                {editToDelete.x_summary || "—"}
              </span>
            </p>
            <label className="mb-1 block text-sm font-medium text-slate-700">
              {t("combined.whyRemoved")}{" "}
              <span className="text-red-500">*</span>
            </label>
            <textarea
              value={deleteReason}
              onChange={(e) => {
                setDeleteReason(e.target.value);
                if (deleteError) setDeleteError("");
              }}
              placeholder={t("combined.deleteReasonPlaceholder")}
              rows={3}
              className="w-full resize-none rounded-lg border border-slate-300 bg-white/95 px-3 py-2 text-sm text-slate-800 focus:border-red-500 focus:ring-2 focus:ring-red-500"
            />
            {deleteError && (
              <p className="text-red-600 text-xs mt-1">{deleteError}</p>
            )}
            <div className="flex justify-end space-x-3 mt-4">
              <button
                onClick={handleCancelDelete}
                disabled={deleting}
                className="rounded-lg border border-slate-300 px-4 py-2 text-sm transition-colors hover:bg-slate-100"
              >
                {t("common.cancel")}
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
                    {t("combined.deleting")}
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
                    {t("combined.deleteSubmission")}
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
