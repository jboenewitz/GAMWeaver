import React, { useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import apiService from "../api/apiService";
import { createTranslator, getDateLocale, getNumberLocale } from "../i18n";
import { getShapeFunctionDisplayName } from "../utils/featureDisplay";

const LEFT_COLORS = {
  base: "#2563eb",
  effective: "#1d4ed8",
};

const RIGHT_COLORS = {
  base: "#f59e0b",
  effective: "#d97706",
};

const buildSelectionMap = (artifact) =>
  Object.fromEntries(
    (artifact?.submissions_by_feature || [])
      .flatMap((feature) => feature.submissions || [])
      .map((submission) => [submission.submission_id, true]),
  );

const getSelectedSubmissionIds = (selectionMap = {}) =>
  Object.entries(selectionMap)
    .filter(([, selected]) => Boolean(selected))
    .map(([submissionId]) => submissionId);

const countSelected = (selectionMap = {}) =>
  Object.values(selectionMap).filter(Boolean).length;

const getFeatureSubmissions = (artifact, featureName) =>
  (artifact?.submissions_by_feature || []).find(
    (feature) => feature?.feature_name === featureName,
  )?.submissions || [];

const countSelectedForFeature = (artifact, featureName, selectionMap = {}) =>
  getFeatureSubmissions(artifact, featureName).filter(
    (submission) => selectionMap?.[submission.submission_id],
  ).length;

const formatSubmissionMeta = (submission, t) => {
  const parts = [
    submission?.user_name || t("common.unknown"),
    `${t("modelCompare.confidence")}: ${submission?.sureness ?? 0}/10`,
    `${submission?.point_count ?? 0} ${t("modelCompare.points")}`,
  ];
  return parts.join(" - ");
};

const formatArtifactDate = (value, language) => {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString(getDateLocale(language));
};

const formatMetricValue = (value, language) => {
  if (!Number.isFinite(value)) return "-";
  return new Intl.NumberFormat(getNumberLocale(language), {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
  }).format(value);
};

const getMetricTone = (status) => {
  if (status === "improved") {
    return {
      container: "border-emerald-200 bg-emerald-50",
      label: "text-emerald-700",
      value: "text-emerald-900",
      status: "text-emerald-700",
    };
  }
  if (status === "worsened") {
    return {
      container: "border-rose-200 bg-rose-50",
      label: "text-rose-700",
      value: "text-rose-900",
      status: "text-rose-700",
    };
  }
  return {
    container: "border-slate-200 bg-slate-50",
    label: "text-slate-600",
    value: "text-slate-900",
    status: "text-slate-600",
  };
};

const getMetricStatusTone = (direction, higherIsBetter = false) => {
  if (direction === "unchanged") return "unchanged";
  if (higherIsBetter) {
    return direction === "increased" ? "improved" : "worsened";
  }
  return direction === "decreased" ? "improved" : "worsened";
};

const buildFeatureTitle = (featurePreview) => {
  const shapeLike = {
    feature_name: featurePreview?.feature_name,
    chart_config:
      featurePreview?.left_chart_config || featurePreview?.right_chart_config || {},
  };
  return getShapeFunctionDisplayName(shapeLike);
};

const CompareFeatureChart = ({ featurePreview, showBaseTraces, t }) => {
  const featureTitle = buildFeatureTitle(featurePreview);
  const isCategorical = featurePreview?.feature_type === "categorical";

  const baseTraces = isCategorical
    ? [
        {
          x: featurePreview.left_x_values,
          y: featurePreview.left_base_y_values,
          type: "bar",
          name: t("modelCompare.leftBase"),
          marker: { color: LEFT_COLORS.base, opacity: 0.35 },
          customdata: featurePreview.left_x_tick_labels || featurePreview.left_x_values,
          hovertemplate:
            "<b>%{customdata}</b><br>%{y:.3f}<extra>" +
            t("modelCompare.leftBase") +
            "</extra>",
        },
        {
          x: featurePreview.right_x_values,
          y: featurePreview.right_base_y_values,
          type: "bar",
          name: t("modelCompare.rightBase"),
          marker: { color: RIGHT_COLORS.base, opacity: 0.35 },
          customdata:
            featurePreview.right_x_tick_labels || featurePreview.right_x_values,
          hovertemplate:
            "<b>%{customdata}</b><br>%{y:.3f}<extra>" +
            t("modelCompare.rightBase") +
            "</extra>",
        },
      ]
    : [
        {
          x: featurePreview.left_x_values,
          y: featurePreview.left_base_y_values,
          type: "scatter",
          mode: "lines",
          name: t("modelCompare.leftBase"),
          line: { color: LEFT_COLORS.base, dash: "dash", width: 2 },
        },
        {
          x: featurePreview.right_x_values,
          y: featurePreview.right_base_y_values,
          type: "scatter",
          mode: "lines",
          name: t("modelCompare.rightBase"),
          line: { color: RIGHT_COLORS.base, dash: "dash", width: 2 },
        },
      ];

  const effectiveTraces = isCategorical
    ? [
        {
          x: featurePreview.left_x_values,
          y: featurePreview.left_effective_y_values,
          type: "bar",
          name: t("modelCompare.leftEffective"),
          marker: { color: LEFT_COLORS.effective, opacity: 0.95 },
          customdata: featurePreview.left_x_tick_labels || featurePreview.left_x_values,
          hovertemplate:
            "<b>%{customdata}</b><br>%{y:.3f}<extra>" +
            t("modelCompare.leftEffective") +
            "</extra>",
        },
        {
          x: featurePreview.right_x_values,
          y: featurePreview.right_effective_y_values,
          type: "bar",
          name: t("modelCompare.rightEffective"),
          marker: { color: RIGHT_COLORS.effective, opacity: 0.95 },
          customdata:
            featurePreview.right_x_tick_labels || featurePreview.right_x_values,
          hovertemplate:
            "<b>%{customdata}</b><br>%{y:.3f}<extra>" +
            t("modelCompare.rightEffective") +
            "</extra>",
        },
      ]
    : [
        {
          x: featurePreview.left_x_values,
          y: featurePreview.left_effective_y_values,
          type: "scatter",
          mode: "lines+markers",
          name: t("modelCompare.leftEffective"),
          marker: { color: LEFT_COLORS.effective, size: 5 },
          line: { color: LEFT_COLORS.effective, width: 3 },
        },
        {
          x: featurePreview.right_x_values,
          y: featurePreview.right_effective_y_values,
          type: "scatter",
          mode: "lines+markers",
          name: t("modelCompare.rightEffective"),
          marker: { color: RIGHT_COLORS.effective, size: 5 },
          line: { color: RIGHT_COLORS.effective, width: 3 },
        },
      ];

  const data = showBaseTraces
    ? [...baseTraces, ...effectiveTraces]
    : effectiveTraces;

  const layout = {
    title: {
      text: featureTitle,
      font: { size: 16, color: "#0f172a" },
    },
    barmode: isCategorical ? "group" : undefined,
    xaxis: {
      title: featureTitle,
      gridcolor: "#e2e8f0",
      tickfont: { size: 10 },
    },
    yaxis: {
      title: t("shapeFunctions.effectOnPrediction"),
      gridcolor: "#e2e8f0",
      zeroline: true,
      zerolinecolor: "#94a3b8",
    },
    height: 360,
    margin: { l: 55, r: 24, t: 50, b: 55 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    legend: {
      orientation: "h",
      x: 0,
      y: 1.15,
      bgcolor: "rgba(255,255,255,0.9)",
    },
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <Plot
        data={data}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        className="w-full"
      />
    </div>
  );
};

const FeatureMetricBadge = ({
  label,
  metricLabel,
  value,
  status,
  higherIsBetter = false,
  language,
  t,
}) => {
  const toneStatus = getMetricStatusTone(status, higherIsBetter);
  const tone = getMetricTone(toneStatus);
  const statusLabel =
    toneStatus === "improved"
      ? t("modelCompare.metricImproved")
      : toneStatus === "worsened"
        ? t("modelCompare.metricWorsened")
        : t("modelCompare.metricUnchanged");

  return (
    <div className={`rounded-2xl border px-3 py-2 ${tone.container}`}>
      <div className={`text-[11px] font-semibold uppercase tracking-wide ${tone.label}`}>
        {label}
      </div>
      <div className={`mt-1 text-base font-semibold ${tone.value}`}>
        {metricLabel}: {formatMetricValue(value, language)}
      </div>
      <div className={`text-xs ${tone.status}`}>{statusLabel}</div>
    </div>
  );
};

const ArtifactSummaryCard = ({
  artifact,
  artifactLabel,
  selectionMap,
  language,
  t,
}) => {
  const metadata = artifact?.metadata || {};
  const totalSubmissions = (artifact?.submissions_by_feature || []).reduce(
    (sum, feature) => sum + (feature?.submissions?.length || 0),
    0,
  );

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {artifactLabel}
          </div>
          <h3 className="mt-1 text-lg font-semibold text-slate-900">
            {metadata.filename || t("common.unknown")}
          </h3>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
          {metadata.artifact_version || "1.0"}
        </span>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 text-sm text-slate-600 sm:grid-cols-2">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            {t("modelCompare.exportedAt")}
          </div>
          <div>{formatArtifactDate(metadata.exported_at, language) || "-"}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            {t("modelCompare.dataset")}
          </div>
          <div>{metadata.dataset_name || "-"}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            {t("common.target")}
          </div>
          <div>{metadata.target_column || "-"}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            {t("modelCompare.features")}
          </div>
          <div>{metadata.selected_feature_count || 0}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            {t("modelCompare.editExport")}
          </div>
          <div>{metadata.has_edit_export ? t("common.yes") : t("common.no")}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">
            {t("modelCompare.selectedSubmissions")}
          </div>
          <div>
            {countSelected(selectionMap)} / {totalSubmissions}
          </div>
        </div>
      </div>
    </div>
  );
};

const FeatureSelectionModal = ({
  open,
  artifact,
  artifactKey,
  artifactLabel,
  featureName,
  selectionMap,
  busy,
  t,
  onClose,
  onToggleSubmission,
}) => {
  if (!open || !featureName) return null;
  const submissions = getFeatureSubmissions(artifact, featureName);
  const featureTitle = getShapeFunctionDisplayName({ feature_name: featureName });
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm">
      <div className="max-h-[85vh] w-full max-w-3xl overflow-hidden rounded-3xl bg-white shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-6 py-5">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              {artifactLabel}
            </div>
            <h3 className="mt-1 text-xl font-semibold text-slate-900">
              {featureTitle}
            </h3>
            <p className="mt-1 text-sm text-slate-600">
              {t("modelCompare.selectionHint")}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
          >
            <svg
              className="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
        <div className="border-b border-slate-200 bg-slate-50 px-6 py-3 text-sm text-slate-600">
          {countSelectedForFeature(artifact, featureName, selectionMap)} /{" "}
          {submissions.length} {t("modelCompare.selectedSubmissions")}
        </div>

        <div className="max-h-[60vh] overflow-y-auto px-6 py-5">
          {!submissions.length && (
            <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-500">
              {t("modelCompare.noImportedEditsForFeature")}
            </div>
          )}

          <div className="space-y-3">
            {submissions.map((submission) => (
              <label
                key={`${artifactKey}-${submission.submission_id}`}
                className="flex cursor-pointer items-start gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3"
              >
                <input
                  type="checkbox"
                  checked={Boolean(selectionMap?.[submission.submission_id])}
                  disabled={busy}
                  onChange={() =>
                    onToggleSubmission(
                      artifactKey,
                      submission.submission_id,
                      !selectionMap?.[submission.submission_id],
                    )
                  }
                  className="mt-1 h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                    <span className="font-medium text-slate-900">
                      {submission.user_name}
                    </span>
                    {submission.profession && (
                      <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
                        {submission.profession}
                      </span>
                    )}
                  </div>
                  <div className="mt-1 text-sm text-slate-600">
                    {formatSubmissionMeta(submission, t)}
                  </div>
                  {submission.x_summary && (
                    <div className="mt-1 text-xs text-slate-500">
                      {t("modelCompare.xSummary")}: {submission.x_summary}
                    </div>
                  )}
                  {submission.message && (
                    <div className="mt-2 text-sm text-slate-700">
                      {submission.message}
                    </div>
                  )}
                </div>
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center justify-between gap-4 border-t border-slate-200 px-6 py-4">
          <div className="text-sm text-slate-500">
            {busy ? t("modelCompare.recomputing") : t("modelCompare.previewOnly")}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-800"
          >
            {t("common.close")}
          </button>
        </div>
      </div>
    </div>
  );
};

function ModelComparePage({ onBack, language = "en" }) {
  const t = createTranslator(language);
  const featureRefs = useRef({});
  const [leftFile, setLeftFile] = useState(null);
  const [rightFile, setRightFile] = useState(null);
  const [comparePayload, setComparePayload] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectionState, setSelectionState] = useState({ left: {}, right: {} });
  const [useConfidence, setUseConfidence] = useState(true);
  const [showBaseTraces, setShowBaseTraces] = useState(true);
  const [featureSearch, setFeatureSearch] = useState("");
  const [jumpFeature, setJumpFeature] = useState("");
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectionModal, setSelectionModal] = useState(null);

  const resetPreparedState = () => {
    setComparePayload(null);
    setPreview(null);
    setSelectionState({ left: {}, right: {} });
    setUseConfidence(true);
    setShowBaseTraces(true);
    setJumpFeature("");
    setSelectionModal(null);
    setError(null);
  };

  const requestPreview = async (nextSelectionState, nextUseConfidence) => {
    if (!comparePayload) return;
    setPreviewLoading(true);
    setError(null);
    try {
      const result = await apiService.previewModelComparison({
        left_artifact: comparePayload.left_artifact,
        right_artifact: comparePayload.right_artifact,
        left_selected_submission_ids: getSelectedSubmissionIds(nextSelectionState.left),
        right_selected_submission_ids: getSelectedSubmissionIds(nextSelectionState.right),
        use_confidence: nextUseConfidence,
        feature_names: comparePayload.shared_features,
      });
      setPreview(result);
    } catch (err) {
      setError(
        err?.response?.data?.detail || err?.message || t("modelCompare.previewError"),
      );
    } finally {
      setPreviewLoading(false);
    }
  };

  const handlePrepare = async () => {
    if (!leftFile || !rightFile) {
      setError(t("modelCompare.chooseBothFiles"));
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await apiService.prepareModelComparison(leftFile, rightFile);
      const nextSelectionState = {
        left: buildSelectionMap(result.left_artifact),
        right: buildSelectionMap(result.right_artifact),
      };
      setComparePayload(result);
      setSelectionState(nextSelectionState);
      setPreview(result.preview);
      setUseConfidence(Boolean(result.preview?.use_confidence));
      const firstFeature = result.shared_features?.[0] || "";
      setJumpFeature(firstFeature);
    } catch (err) {
      resetPreparedState();
      setError(
        err?.response?.data?.detail || err?.message || t("modelCompare.prepareError"),
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (side, file) => {
    if (side === "left") {
      setLeftFile(file || null);
    } else {
      setRightFile(file || null);
    }
    resetPreparedState();
  };

  const handleToggleSubmission = async (artifactKey, submissionId, checked) => {
    const nextSelectionState = {
      ...selectionState,
      [artifactKey]: {
        ...selectionState[artifactKey],
        [submissionId]: checked,
      },
    };
    setSelectionState(nextSelectionState);
    await requestPreview(nextSelectionState, useConfidence);
  };

  const handleConfidenceToggle = async (checked) => {
    setUseConfidence(checked);
    await requestPreview(selectionState, checked);
  };

  const visiblePreviews = useMemo(() => {
    const allPreviews = preview?.feature_previews || [];
    const query = featureSearch.trim().toLowerCase();
    if (!query) return allPreviews;
    return allPreviews.filter((featurePreview) => {
      const title = buildFeatureTitle(featurePreview).toLowerCase();
      return (
        title.includes(query) ||
        String(featurePreview?.feature_name || "")
          .toLowerCase()
          .includes(query)
      );
    });
  }, [featureSearch, preview]);

  const handleJumpToFeature = () => {
    if (!jumpFeature) return;
    featureRefs.current[jumpFeature]?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const openSelectionModal = (artifactKey, featureName) => {
    setSelectionModal({ artifactKey, featureName });
  };

  const closeSelectionModal = () => {
    setSelectionModal(null);
  };

  const modalArtifact =
    selectionModal?.artifactKey === "left"
      ? comparePayload?.left_artifact
      : comparePayload?.right_artifact;
  const modalSelectionMap =
    selectionModal?.artifactKey === "left"
      ? selectionState.left
      : selectionState.right;

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="container mx-auto flex items-center justify-between gap-4 px-4 py-4">
          <div className="flex items-center gap-3">
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
            <div>
              <h1 className="text-xl font-bold text-slate-900">
                {t("modelCompare.title")}
              </h1>
              <p className="text-sm text-slate-600">
                {t("modelCompare.subtitle")}
              </p>
            </div>
          </div>
          <div className="rounded-full bg-indigo-50 px-4 py-2 text-sm font-medium text-indigo-700">
            {t("modelCompare.previewOnly")}
          </div>
        </div>
      </header>

      <main className="container mx-auto space-y-6 px-4 py-8">
        {error && (
          <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700">
            {error}
          </div>
        )}

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-col gap-6 lg:flex-row">
            <div className="grid flex-1 grid-cols-1 gap-4 md:grid-cols-2">
              <label className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-5">
                <div className="text-sm font-medium text-slate-700">
                  {t("modelCompare.leftArtifact")}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {leftFile?.name || t("modelCompare.noFileChosen")}
                </div>
                <input
                  type="file"
                  accept=".json,application/json"
                  className="mt-4 block w-full text-sm text-slate-600"
                  onChange={(event) =>
                    handleFileChange("left", event.target.files?.[0] || null)
                  }
                />
              </label>

              <label className="rounded-xl border border-dashed border-slate-300 bg-slate-50 px-4 py-5">
                <div className="text-sm font-medium text-slate-700">
                  {t("modelCompare.rightArtifact")}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {rightFile?.name || t("modelCompare.noFileChosen")}
                </div>
                <input
                  type="file"
                  accept=".json,application/json"
                  className="mt-4 block w-full text-sm text-slate-600"
                  onChange={(event) =>
                    handleFileChange("right", event.target.files?.[0] || null)
                  }
                />
              </label>
            </div>

            <div className="w-full max-w-sm rounded-2xl bg-slate-900 p-5 text-white">
              <h2 className="text-lg font-semibold">
                {t("modelCompare.compareArtifacts")}
              </h2>
              <p className="mt-2 text-sm text-slate-300">
                {t("modelCompare.instructions")}
              </p>
              <button
                onClick={handlePrepare}
                disabled={loading || !leftFile || !rightFile}
                className="mt-5 w-full rounded-xl bg-indigo-500 px-4 py-3 font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? t("modelCompare.preparing") : t("modelCompare.compareArtifacts")}
              </button>
            </div>
          </div>
        </section>

        {comparePayload && (
          <>
            <section className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              <ArtifactSummaryCard
                artifact={comparePayload.left_artifact}
                artifactLabel={t("modelCompare.leftArtifact")}
                selectionMap={selectionState.left}
                language={language}
                t={t}
              />
              <ArtifactSummaryCard
                artifact={comparePayload.right_artifact}
                artifactLabel={t("modelCompare.rightArtifact")}
                selectionMap={selectionState.right}
                language={language}
                t={t}
              />
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">
                    {t("modelCompare.chartControls")}
                  </h2>
                  <p className="text-sm text-slate-600">
                    {t("modelCompare.chartControlsDescription")}
                  </p>
                </div>

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                  <label className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                    <input
                      type="checkbox"
                      checked={useConfidence}
                      onChange={(event) => handleConfidenceToggle(event.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    <div>
                      <div className="text-sm font-medium text-slate-800">
                        {useConfidence
                          ? t("modelCompare.confidenceOn")
                          : t("modelCompare.confidenceOff")}
                      </div>
                      <div className="text-xs text-slate-500">
                        {t("modelCompare.confidenceDescription")}
                      </div>
                    </div>
                  </label>

                  <button
                    type="button"
                    onClick={() => setShowBaseTraces((current) => !current)}
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showBaseTraces
                        ? "border-slate-200 bg-slate-50 text-slate-800 hover:bg-slate-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {showBaseTraces
                        ? t("modelCompare.hideBaseTraces")
                        : t("modelCompare.showBaseTraces")}
                    </div>
                    <div className="text-xs text-slate-500">
                      {t("modelCompare.baseTracesDescription")}
                    </div>
                  </button>

                  <input
                    type="search"
                    value={featureSearch}
                    onChange={(event) => setFeatureSearch(event.target.value)}
                    placeholder={t("modelCompare.searchFeatures")}
                    className="rounded-xl border border-slate-300 px-4 py-3 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                  />

                  <div className="flex items-center gap-2">
                    <select
                      value={jumpFeature}
                      onChange={(event) => setJumpFeature(event.target.value)}
                      className="rounded-xl border border-slate-300 px-4 py-3 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    >
                      {(comparePayload.shared_features || []).map((featureName) => (
                        <option key={featureName} value={featureName}>
                          {featureName}
                        </option>
                      ))}
                    </select>
                    <button
                      onClick={handleJumpToFeature}
                      className="rounded-xl border border-slate-300 px-4 py-3 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
                    >
                      {t("modelCompare.jumpToFeature")}
                    </button>
                  </div>
                </div>
              </div>
            </section>

            <section className="space-y-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">
                    {t("modelCompare.featureCharts")}
                  </h2>
                  <p className="text-sm text-slate-600">
                    {t("modelCompare.featureChartsDescription")}
                  </p>
                </div>
                {previewLoading && (
                  <div className="text-sm text-slate-500">
                    {t("modelCompare.recomputing")}
                  </div>
                )}
              </div>

              {!visiblePreviews.length && (
                <div className="rounded-xl border border-dashed border-slate-300 bg-white px-4 py-8 text-center text-slate-500">
                  {t("modelCompare.noMatchingFeatures")}
                </div>
              )}

              <div className="space-y-6">
                {visiblePreviews.map((featurePreview) => (
                  <div
                    key={featurePreview.feature_name}
                    ref={(node) => {
                      featureRefs.current[featurePreview.feature_name] = node;
                    }}
                    className="space-y-3"
                  >
                    <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
                      <div className="space-y-3">
                        <h3 className="text-base font-semibold text-slate-900">
                          {buildFeatureTitle(featurePreview)}
                        </h3>
                        <p className="text-sm text-slate-500">
                          {featurePreview.feature_name}
                        </p>
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
                          <FeatureMetricBadge
                            label={t("modelCompare.maeVsEditedBase")}
                            metricLabel={t("combined.mae")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_edited_base_mae
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_edited_base_status
                            }
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.maeVsOriginalBase")}
                            metricLabel={t("combined.mae")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_mae
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_status
                            }
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.rmseVsEditedBase")}
                            metricLabel={t("combined.rmse")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_edited_base_rmse
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_edited_base_rmse_status
                            }
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.rmseVsOriginalBase")}
                            metricLabel={t("combined.rmse")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_rmse
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_rmse_status
                            }
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.r2VsEditedBase")}
                            metricLabel={t("combined.r2")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_edited_base_r2
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_edited_base_r2_status
                            }
                            higherIsBetter
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.r2VsOriginalBase")}
                            metricLabel={t("combined.r2")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_r2
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_r2_status
                            }
                            higherIsBetter
                            language={language}
                            t={t}
                          />
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                        <button
                          type="button"
                          onClick={() =>
                            openSelectionModal("left", featurePreview.feature_name)
                          }
                          className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 transition-colors hover:bg-blue-100"
                        >
                          {t("modelCompare.editLeftSelections")} (
                          {countSelectedForFeature(
                            comparePayload.left_artifact,
                            featurePreview.feature_name,
                            selectionState.left,
                          )}
                          /
                          {getFeatureSubmissions(
                            comparePayload.left_artifact,
                            featurePreview.feature_name,
                          ).length}
                          )
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            openSelectionModal("right", featurePreview.feature_name)
                          }
                          className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700 transition-colors hover:bg-amber-100"
                        >
                          {t("modelCompare.editRightSelections")} (
                          {countSelectedForFeature(
                            comparePayload.right_artifact,
                            featurePreview.feature_name,
                            selectionState.right,
                          )}
                          /
                          {getFeatureSubmissions(
                            comparePayload.right_artifact,
                            featurePreview.feature_name,
                          ).length}
                          )
                        </button>
                      </div>
                    </div>
                    <CompareFeatureChart
                      featurePreview={featurePreview}
                      showBaseTraces={showBaseTraces}
                      t={t}
                    />
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </main>

      <FeatureSelectionModal
        open={Boolean(selectionModal)}
        artifact={modalArtifact}
        artifactKey={selectionModal?.artifactKey}
        artifactLabel={
          selectionModal?.artifactKey === "left"
            ? t("modelCompare.leftArtifactSelections")
            : t("modelCompare.rightArtifactSelections")
        }
        featureName={selectionModal?.featureName}
        selectionMap={modalSelectionMap}
        busy={previewLoading}
        t={t}
        onClose={closeSelectionModal}
        onToggleSubmission={handleToggleSubmission}
      />
    </div>
  );
}

export default ModelComparePage;
