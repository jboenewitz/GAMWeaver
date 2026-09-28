import React, { useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import apiService from "../api/apiService";
import { createTranslator, getDateLocale, getNumberLocale } from "../i18n";
import { getShapeFunctionDisplayName } from "../utils/featureDisplay";

const LEFT_COLORS = {
  base: "#2563eb",
};

const RIGHT_COLORS = {
  base: "#f59e0b",
  effective: "#d97706",
};

const FEATURE_EDIT_SHIFT_STEP = 1;

const FOUR_POINT_SCALE_TICKS = [
  { value: 1, label: "Low" },
  { value: 2, label: "Rather Low" },
  { value: 3, label: "Rather High" },
  { value: 4, label: "High" },
];

const BOOKS_AT_HOME_SCALE_TICKS = [
  { value: 1, label: "0-10" },
  { value: 2, label: "11-25" },
  { value: 3, label: "26-100" },
  { value: 4, label: "101-200" },
  { value: 5, label: "200-500" },
  { value: 6, label: "more than 500" },
];

const usesOrdinalScale = (featurePreview, start, end) => {
  const xValues = [
    ...(featurePreview?.left_x_values || []),
    ...(featurePreview?.right_x_values || []),
  ].map(Number);
  if (!xValues.length || xValues.some((value) => !Number.isFinite(value))) {
    return false;
  }

  const min = Math.min(...xValues);
  const max = Math.max(...xValues);
  return Math.abs(min - start) < 1e-9 && Math.abs(max - end) < 1e-9;
};

const getOrdinalScaleTickValues = (featurePreview, isCategorical, ticks) => {
  if (!isCategorical) {
    return ticks.map((tick) => tick.value);
  }

  const categoricalValues = [
    ...(featurePreview?.left_x_values || []),
    ...(featurePreview?.right_x_values || []),
  ];
  return ticks.map(
    (tick) =>
      categoricalValues.find((value) => Number(value) === tick.value) ??
      String(tick.value),
  );
};

const isBooksAtHomeFeature = (featurePreview) => {
  const featureIdentity = [
    featurePreview?.feature_name,
    featurePreview?.left_chart_config?.display_title,
    featurePreview?.right_chart_config?.display_title,
  ]
    .filter(Boolean)
    .join(" ")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();

  return (
    featureIdentity.includes("kbuecher") ||
    featureIdentity.includes("bucher zu hause") ||
    featureIdentity.includes("books at home")
  );
};

// Deliberately excludes blue so individual edits cannot be confused with the
// original artifact's dashed blue base trace.
const INDIVIDUAL_EDIT_COLORS = [
  "#16a34a",
  "#67e8f9",
  "#dc2626",
  "#fdba74",
  "#c4b5fd",
  "#bef264",
  "#5eead4",
  "#f9a8d4",
  "#fde047",
  "#a5b4fc",
  "#f0abfc",
  "#fda4af",
  "#fcd34d",
];

// Fixed expert labels keep their colors when the selected traces change.
const EXPERT_EDIT_COLORS = {
  E1: "#17eb65",
  E2: "#abf5ff",
  E3: "#ff3d3d",
  E4: "#fdeb74",
  E5: "#c4b5fd",
  E6: "#bef264",
  E7: "#f472b6",
};

const getIndividualEditColor = (userName) => {
  const label = String(userName || "").trim().toUpperCase();
  const expertMatch = label.match(/^[EP](\d+)$/);
  const expertId = expertMatch ? `E${expertMatch[1]}` : null;
  if (expertId && EXPERT_EDIT_COLORS[expertId]) {
    return EXPERT_EDIT_COLORS[expertId];
  }

  const hash = Array.from(label).reduce(
    (value, character) => (value * 31 + character.charCodeAt(0)) >>> 0,
    0,
  );
  return INDIVIDUAL_EDIT_COLORS[hash % INDIVIDUAL_EDIT_COLORS.length];
};

const formatChartExpertLabel = (userName) => {
  const label = String(userName || "").trim();
  const expertMatch = label.match(/^[EP](\d+)$/i);
  return expertMatch
    ? `Expert${expertMatch[1]} (E${expertMatch[1]})`
    : label;
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

const TRANSCRIPT_PARTICIPANT_NAMES = {
  P1: "Anica",
  P2: "Laura",
  P3: "Lilly",
  P4: "Fuchsi",
  P5: "Sarah",
  P6: "Anni",
  P7: "Stephanie Hagen",
};

const formatTranscriptParticipantName = (userName, t) => {
  const name = String(userName || "").trim();
  if (!name) return t("common.unknown");

  const transcriptName = TRANSCRIPT_PARTICIPANT_NAMES[name.toUpperCase()];
  return transcriptName ? `${name} – ${transcriptName}` : name;
};

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
    formatTranscriptParticipantName(submission?.user_name, t),
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

const formatSignedMetricValue = (value, language) => {
  if (!Number.isFinite(value)) return "-";
  const prefix = value > 0 ? "+" : "";
  return prefix + formatMetricValue(value, language);
};

const formatRelativePercentageChange = (value, language) => {
  if (!Number.isFinite(value)) return "-";
  const prefix = value > 0 ? "+" : "";
  return (
    prefix +
    new Intl.NumberFormat(getNumberLocale(language), {
      minimumFractionDigits: 1,
      maximumFractionDigits: 1,
    }).format(value) +
    "%"
  );
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
      featurePreview?.left_chart_config ||
      featurePreview?.right_chart_config ||
      {},
  };
  return getShapeFunctionDisplayName(shapeLike);
};

const centerChartTraces = (traces, shouldCenter) => {
  if (!shouldCenter) return traces;

  return traces.map((trace) => {
    const yValues = trace?.y;
    if (
      !Array.isArray(yValues) ||
      !yValues.length ||
      yValues.some((value) => !Number.isFinite(value))
    ) {
      return trace;
    }
    const mean = yValues.reduce((sum, value) => sum + value, 0) / yValues.length;
    return {
      ...trace,
      y: yValues.map((value) => value - mean),
    };
  });
};

const getEffectYAxisScale = (traces) => {
  const yValues = [];
  for (const trace of traces || []) {
    for (const value of trace?.y || []) {
      if (Number.isFinite(value)) yValues.push(value);
    }
  }

  if (!yValues.length) return undefined;

  const dataMinimum = Math.min(...yValues);
  const dataMaximum = Math.max(...yValues);
  const lowerBound = Math.min(dataMinimum, 0);
  const upperBound = Math.max(dataMaximum, 0);
  const span = Math.max(upperBound - lowerBound, 1);
  const rawTickSize = span / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rawTickSize));
  const normalizedTickSize = rawTickSize / magnitude;
  const niceTickSize =
    normalizedTickSize <= 1
      ? 1
      : normalizedTickSize <= 2
        ? 2
        : normalizedTickSize <= 5
          ? 5
          : 10;
  const dtick = niceTickSize * magnitude;
  let rangeStart = Math.floor(lowerBound / dtick) * dtick;
  let rangeEnd = Math.ceil(upperBound / dtick) * dtick;

  if (dataMinimum < 0 && rangeStart === dataMinimum) rangeStart -= dtick;
  if (dataMaximum > 0 && rangeEnd === dataMaximum) rangeEnd += dtick;
  if (rangeStart === rangeEnd) {
    rangeStart -= dtick;
    rangeEnd += dtick;
  }

  return {
    range: [rangeStart, rangeEnd],
    dtick,
    tick0: 0,
  };
};

const CompareFeatureChart = ({
  featurePreview,
  showBaseTraces,
  centerCharts,
  showIndividualEditTraces,
  showConfidenceInLegend,
  legendOnRight,
  showLineMarkers,
  t,
}) => {
  const featureTitle = buildFeatureTitle(featurePreview);
  const isCategorical = featurePreview?.feature_type === "categorical";
  const participantEditCounts = new Map();
  [
    ...(featurePreview.left_individual_edit_traces || []),
    ...(featurePreview.right_individual_edit_traces || []),
  ].forEach((editTrace) => {
    const participantKey = editTrace.user_name || editTrace.submission_id;
    participantEditCounts.set(
      participantKey,
      (participantEditCounts.get(participantKey) || 0) + 1,
    );
  });

  const baseTraces = isCategorical
    ? [
        {
          x: featurePreview.left_x_values,
          y: featurePreview.left_base_y_values,
          type: "bar",
          name: t("modelCompare.leftBase"),
          marker: { color: LEFT_COLORS.base },
          customdata:
            featurePreview.left_x_tick_labels || featurePreview.left_x_values,
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
          marker: { color: RIGHT_COLORS.base },
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

  const buildIndividualEditTraces = (side) => {
    const isLeft = side === "left";
    const xValues = isLeft
      ? featurePreview.left_x_values
      : featurePreview.right_x_values;
    const xTickLabels = isLeft
      ? featurePreview.left_x_tick_labels
      : featurePreview.right_x_tick_labels;
    const editTraces = isLeft
      ? featurePreview.left_individual_edit_traces || []
      : featurePreview.right_individual_edit_traces || [];

    return editTraces.map((editTrace) => {
      const participantKey = editTrace.user_name || editTrace.submission_id;
      const participantLabel = formatChartExpertLabel(participantKey);
      const color = getIndividualEditColor(participantKey);
      const confidence = Number(editTrace.sureness) || 0;
      const showSubmissionId = participantEditCounts.get(participantKey) > 1;
      const name = `${participantLabel}${
        showConfidenceInLegend
          ? ` — ${t("modelCompare.confidence")}: ${confidence}/10`
          : ""
      }${
        showSubmissionId ? ` (${editTrace.submission_id})` : ""
      }`;
      if (isCategorical) {
        return {
          x: xValues,
          y: editTrace.y_values,
          type: "bar",
          name,
          meta: { sureness: confidence },
          marker: { color, opacity: 0.75 },
          customdata: xTickLabels || xValues,
          hovertemplate:
            "<b>%{customdata}</b><br>%{y:.3f}<extra>" + name + "</extra>",
        };
      }
      return {
        x: xValues,
        y: editTrace.y_values,
        type: "scatter",
        mode: showLineMarkers ? "lines+markers" : "lines",
        name,
        meta: { sureness: confidence },
        marker: { color, size: 5 },
        line: { color, width: 3 },
        opacity: 0.95,
      };
    });
  };

  const individualEditTraces = [
    ...buildIndividualEditTraces("left"),
    ...buildIndividualEditTraces("right"),
  ];
  if (showConfidenceInLegend) {
    individualEditTraces.sort(
      (left, right) => right.meta.sureness - left.meta.sureness,
    );
  }

  const effectiveTraces = showIndividualEditTraces
    ? individualEditTraces
    : isCategorical
    ? [
        {
          x: featurePreview.right_x_values,
          y: featurePreview.right_effective_y_values,
          type: "bar",
          name: t("modelCompare.aggregatedParticipantEdits"),
          marker: { color: RIGHT_COLORS.effective, opacity: 0.95 },
          customdata:
            featurePreview.right_x_tick_labels || featurePreview.right_x_values,
          hovertemplate:
            "<b>%{customdata}</b><br>%{y:.3f}<extra>" +
            t("modelCompare.aggregatedParticipantEdits") +
            "</extra>",
        },
      ]
    : [
        {
          x: featurePreview.right_x_values,
          y: featurePreview.right_effective_y_values,
          type: "scatter",
          mode: showLineMarkers ? "lines+markers" : "lines",
          name: t("modelCompare.aggregatedParticipantEdits"),
          marker: { color: RIGHT_COLORS.effective, size: 5 },
          line: { color: RIGHT_COLORS.effective, width: 3 },
        },
      ];

  const data = showBaseTraces
    ? [...baseTraces, ...effectiveTraces]
    : effectiveTraces;
  const chartData = centerChartTraces(data, centerCharts);
  const booksAtHome = isBooksAtHomeFeature(featurePreview);
  const scaleTicks = booksAtHome
    ? BOOKS_AT_HOME_SCALE_TICKS
    : usesOrdinalScale(featurePreview, 1, 4)
      ? FOUR_POINT_SCALE_TICKS
      : null;
  const hasFixedOrdinalScale =
    scaleTicks &&
    (booksAtHome ||
      usesOrdinalScale(
        featurePreview,
        scaleTicks[0].value,
        scaleTicks[scaleTicks.length - 1].value,
      ));
  const ordinalScaleTickValues = getOrdinalScaleTickValues(
    featurePreview,
    isCategorical,
    scaleTicks || [],
  );

  const layout = {
    autosize: true,
    title: {
      text: featureTitle,
      font: { size: 16, color: "#0f172a" },
    },
    barmode: isCategorical ? "group" : undefined,
    xaxis: {
      title: featureTitle,
      gridcolor: "#e2e8f0",
      tickfont: { size: 10 },
      ...(hasFixedOrdinalScale
        ? {
            ...(!isCategorical
              ? {
                  range: [
                    scaleTicks[0].value,
                    scaleTicks[scaleTicks.length - 1].value,
                  ],
                }
              : {}),
            tickmode: "array",
            tickvals: ordinalScaleTickValues,
            ticktext: scaleTicks.map((tick) => tick.label),
          }
        : {}),
    },
    yaxis: {
      title: t("shapeFunctions.effectOnPrediction"),
      gridcolor: "#e2e8f0",
      zeroline: true,
      zerolinecolor: "#94a3b8",
      ...getEffectYAxisScale(chartData),
    },
    height: 360,
    margin: legendOnRight
      ? { l: 55, r: 220, t: 80, b: 55 }
      : { l: 55, r: 24, t: 80, b: 55 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    legend: legendOnRight
      ? {
          orientation: "v",
          x: 1.02,
          xanchor: "left",
          y: 1,
          yanchor: "top",
          bgcolor: "rgba(255,255,255,0.9)",
        }
      : {
          orientation: "h",
          x: 0,
          y: 1.15,
          bgcolor: "rgba(255,255,255,0.9)",
        },
  };

  return (
    <div className="w-full rounded-2xl border border-slate-200 bg-white p-4 shadow-sm xl:w-3/5">
      <Plot
        data={chartData}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        style={{ width: "100%" }}
        className="w-full"
      />
    </div>
  );
};

const FeatureMetricBadge = ({
  label,
  metricLabel,
  value,
  baselineValue,
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
  const hasBaselineComparison =
    Number.isFinite(value) && Number.isFinite(baselineValue);
  const metricDelta = hasBaselineComparison ? value - baselineValue : null;
  const relativeMetricChange =
    hasBaselineComparison && Math.abs(baselineValue) > 1e-12
      ? (metricDelta / Math.abs(baselineValue)) * 100
      : null;

  return (
    <div className={`rounded-2xl border px-3 py-2 ${tone.container}`}>
      <div
        className={`text-[11px] font-semibold uppercase tracking-wide ${tone.label}`}
      >
        {label}
      </div>
      <div className={`mt-1 text-base font-semibold ${tone.value}`}>
        {metricLabel}: {formatMetricValue(value, language)}
      </div>
      {hasBaselineComparison && (
        <div className="mt-1 text-xs text-slate-600">
          <div>
            {t("modelCompare.reducedVsFullData")}: {" "}
            {formatMetricValue(baselineValue, language)}
          </div>
          <div>
            {t("modelCompare.deltaFromBaseline")}: {" "}
            {formatSignedMetricValue(metricDelta, language)}
          </div>
          <div>
            {t("modelCompare.relativeChange")}: {" "}
            {formatRelativePercentageChange(relativeMetricChange, language)}
          </div>
        </div>
      )}
      {status && <div className={`text-xs ${tone.status}`}>{statusLabel}</div>}
    </div>
  );
};

const getOverallMaeFeatureTone = (status) => {
  const toneStatus = getMetricStatusTone(status);
  if (toneStatus === "improved") {
    return "border-emerald-200 bg-emerald-50 text-emerald-950";
  }
  if (toneStatus === "worsened") {
    return "border-rose-200 bg-rose-50 text-rose-950";
  }
  return "border-slate-200 bg-slate-50 text-slate-700";
};

const OverallShapeFunctionMaeSummary = ({
  metrics,
  featureNames,
  featurePreviews,
  featureSelection,
  onFeatureSelectionChange,
  previewLoading,
  language,
  t,
}) => {
  if (!metrics) return null;
  const selectedFeatureCount = (featureNames || []).filter(
    (featureName) => featureSelection?.[featureName] !== false,
  ).length;
  const featurePreviewByName = new Map(
    (featurePreviews || []).map((featurePreview) => [
      featurePreview.feature_name,
      featurePreview,
    ]),
  );

  return (
    <section className="rounded-2xl border border-emerald-200 bg-emerald-50/60 p-5 shadow-sm">
      <div className="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            {t("modelCompare.overallShapeFunctionMae")}
          </h2>
          <p className="text-sm text-slate-600">
            {t("modelCompare.overallShapeFunctionMaeDescription")}
          </p>
        </div>
        <div className="text-xs text-slate-500">
          {t("modelCompare.overallShapeFunctionMaeCoverage", {
            features: metrics.feature_count || 0,
            points: metrics.point_count || 0,
          })}
        </div>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-2">
        <FeatureMetricBadge
          label={t("modelCompare.pooledShapeFunctionMae")}
          metricLabel={t("combined.mae")}
          value={metrics.pooled?.edited}
          baselineValue={metrics.pooled?.baseline}
          status={metrics.pooled?.status}
          language={language}
          t={t}
        />
        <FeatureMetricBadge
          label={t("modelCompare.meanPerFeatureMae")}
          metricLabel={t("combined.mae")}
          value={metrics.mean_per_feature?.edited}
          baselineValue={metrics.mean_per_feature?.baseline}
          status={metrics.mean_per_feature?.status}
          language={language}
          t={t}
        />
      </div>
      <details className="mt-4 rounded-xl border border-emerald-200 bg-white px-4 py-3">
        <summary className="cursor-pointer text-sm font-semibold text-emerald-950">
          {t("modelCompare.overallMaeFeatureSelection", {
            selected: selectedFeatureCount,
            total: (featureNames || []).length,
          })}
        </summary>
        <p className="mt-2 text-sm text-slate-600">
          {t("modelCompare.overallMaeFeatureSelectionDescription")}
        </p>
        <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {(featureNames || []).map((featureName) => {
            const featureMetrics = featurePreviewByName.get(featureName)
              ?.comparison_metrics;
            const baselineMae =
              featureMetrics?.baseline_edited_vs_original_base_mae;
            const editedMae = featureMetrics?.edited_vs_original_base_mae;
            const maeDelta =
              Number.isFinite(baselineMae) && Number.isFinite(editedMae)
              ? editedMae - baselineMae
              : null;
            const isSelected = featureSelection?.[featureName] !== false;
            return (
              <button
                type="button"
                key={featureName}
                aria-pressed={isSelected}
                disabled={previewLoading || (selectedFeatureCount === 1 && isSelected)}
                onClick={() => onFeatureSelectionChange(featureName, !isSelected)}
                className={`flex min-w-0 items-center rounded-lg border px-3 py-2 text-left text-sm transition-opacity focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:cursor-not-allowed ${
                  isSelected
                    ? getOverallMaeFeatureTone(
                        featureMetrics?.edited_vs_original_base_status,
                      )
                    : "border-slate-200 bg-slate-100 text-slate-500 opacity-50 grayscale"
                }`}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">{featureName}</span>
                  <span className="block text-xs opacity-80">
                    {t("modelCompare.featureMaeDelta", {
                      delta: formatSignedMetricValue(maeDelta, language),
                    })}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </details>
    </section>
  );
};

const IndividualParticipantMaeList = ({ featurePreview, language, t }) => {
  const participantTraces = [
    ...(featurePreview.left_individual_edit_traces || []).map((trace) => ({
      trace,
      artifactLabel: t("modelCompare.leftArtifact"),
    })),
    ...(featurePreview.right_individual_edit_traces || []).map((trace) => ({
      trace,
      artifactLabel: t("modelCompare.rightArtifact"),
    })),
  ];

  if (!participantTraces.length) return null;

  return (
    <div className="rounded-2xl border border-indigo-200 bg-indigo-50/60 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-indigo-800">
        {t("modelCompare.individualMaeErrors")}
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {participantTraces.map(({ trace, artifactLabel }) => (
          <div
            key={`${artifactLabel}-${trace.submission_id}`}
            className="rounded-xl border border-indigo-100 bg-white px-3 py-2 text-sm text-slate-800"
          >
            <span className="font-medium">
              {formatChartExpertLabel(trace.user_name)}
            </span>{" "}
            <span className="text-slate-500">({artifactLabel})</span>
            <span className="mx-1 text-slate-300">·</span>
            <span className="font-semibold">
              {t("combined.mae")}: {formatMetricValue(
                trace.original_base_mae,
                language,
              )}
            </span>
          </div>
        ))}
      </div>
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
          <div>
            {metadata.has_edit_export ? t("common.yes") : t("common.no")}
          </div>
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
  const featureTitle = getShapeFunctionDisplayName({
    feature_name: featureName,
  });
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
                      {formatTranscriptParticipantName(submission.user_name, t)}
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
            {busy
              ? t("modelCompare.recomputing")
              : t("modelCompare.previewOnly")}
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

const MeanShiftFeatureSelectionModal = ({
  open,
  featureRows,
  selectionMap,
  busy,
  t,
  onClose,
  onToggleFeature,
  onApply,
}) => {
  if (!open) return null;
  const selectedCount = featureRows.filter(
    (row) => selectionMap[row.feature_name] === true,
  ).length;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm">
      <div className="max-h-[85vh] w-full max-w-2xl overflow-hidden rounded-3xl bg-white shadow-2xl">
        <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-6 py-5">
          <div>
            <h3 className="text-xl font-semibold text-slate-900">
              {t("modelCompare.meanShiftFeatureModalTitle")}
            </h3>
            <p className="mt-1 text-sm text-slate-600">
              {t("modelCompare.meanShiftFeatureModalDescription")}
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
          {t("modelCompare.meanShiftFeatureModalSelection", {
            selected: selectedCount,
            total: featureRows.length,
          })}
        </div>
        <div className="max-h-[55vh] overflow-y-auto px-6 py-5">
          <div className="space-y-2">
            {featureRows.map((row) => {
              const isSelected = selectionMap[row.feature_name] === true;
              return (
                <label
                  key={row.feature_name}
                  className={`flex cursor-pointer items-center justify-between gap-3 rounded-xl border px-4 py-3 transition-colors ${
                    isSelected
                      ? "border-indigo-300 bg-indigo-50"
                      : "border-slate-200 bg-white hover:bg-slate-50"
                  }`}
                >
                  <span className="font-medium text-slate-900">
                    {row.feature_name}
                  </span>
                  <input
                    type="checkbox"
                    checked={isSelected}
                    disabled={busy}
                    onChange={() => onToggleFeature(row.feature_name)}
                    className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                  />
                </label>
              );
            })}
          </div>
        </div>
        <div className="flex items-center justify-end gap-3 border-t border-slate-200 px-6 py-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50"
          >
            {t("common.cancel")}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={onApply}
            className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {t("modelCompare.applyMeanShiftFeatureLimit")}
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
  const [overallMaeFeatureSelection, setOverallMaeFeatureSelection] =
    useState({});
  const [contributionFeatureSelection, setContributionFeatureSelection] =
    useState({});
  const [meanShiftFeatureFactorSelection, setMeanShiftFeatureFactorSelection] =
    useState({});
  const [meanShiftFeatureFactorDraft, setMeanShiftFeatureFactorDraft] =
    useState({});
  const [meanShiftFeatureModalOpen, setMeanShiftFeatureModalOpen] =
    useState(false);
  const [manualFeatureShifts, setManualFeatureShifts] = useState({});
  const [manualFeatureShiftInputValues, setManualFeatureShiftInputValues] =
    useState({});
  const [useConfidence, setUseConfidence] = useState(true);
  const [averageScoreMethod, setAverageScoreMethod] = useState("uniform");
  const [showBaseTraces, setShowBaseTraces] = useState(true);
  const [centerCharts, setCenterCharts] = useState(false);
  const [showIndividualEditTraces, setShowIndividualEditTraces] = useState(false);
  const [showIndividualParticipantMae, setShowIndividualParticipantMae] =
    useState(false);
  const [showConfidenceInLegend, setShowConfidenceInLegend] = useState(false);
  const [legendOnRight, setLegendOnRight] = useState(false);
  const [showLineMarkers, setShowLineMarkers] = useState(true);
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
    setOverallMaeFeatureSelection({});
    setContributionFeatureSelection({});
    setMeanShiftFeatureFactorSelection({});
    setMeanShiftFeatureFactorDraft({});
    setMeanShiftFeatureModalOpen(false);
    setManualFeatureShifts({});
    setManualFeatureShiftInputValues({});
    setUseConfidence(true);
    setAverageScoreMethod("uniform");
    setShowBaseTraces(true);
    setCenterCharts(false);
    setShowIndividualEditTraces(false);
    setShowIndividualParticipantMae(false);
    setShowConfidenceInLegend(false);
    setLegendOnRight(false);
    setShowLineMarkers(true);
    setJumpFeature("");
    setSelectionModal(null);
    setError(null);
  };

  const requestPreview = async (
    nextSelectionState,
    nextUseConfidence,
    nextLimitEditedMeanDeviation = false,
    nextOverallMaeFeatureSelection = overallMaeFeatureSelection,
    nextContributionFeatureSelection = contributionFeatureSelection,
    nextMeanShiftFeatureFactorSelection = meanShiftFeatureFactorSelection,
    nextManualFeatureShifts = manualFeatureShifts,
  ) => {
    if (!comparePayload) return;
    setPreviewLoading(true);
    setError(null);
    try {
      const result = await apiService.previewModelComparison({
        left_artifact: comparePayload.left_artifact,
        right_artifact: comparePayload.right_artifact,
        left_selected_submission_ids: getSelectedSubmissionIds(
          nextSelectionState.left,
        ),
        right_selected_submission_ids: getSelectedSubmissionIds(
          nextSelectionState.right,
        ),
        use_confidence: nextUseConfidence,
        limit_edited_mean_deviation: nextLimitEditedMeanDeviation,
        feature_names: comparePayload.shared_features,
        overall_mae_feature_names: (comparePayload.shared_features || []).filter(
          (featureName) => nextOverallMaeFeatureSelection[featureName] !== false,
        ),
        mean_deviation_feature_names: (comparePayload.shared_features || []).filter(
          (featureName) =>
            nextContributionFeatureSelection[featureName] !== false,
        ),
        mean_deviation_scaled_feature_names: (
          comparePayload.shared_features || []
        ).filter(
          (featureName) =>
            nextMeanShiftFeatureFactorSelection[featureName] === true &&
            nextContributionFeatureSelection[featureName] !== false,
        ),
        manual_feature_shifts: nextManualFeatureShifts,
      });
      setPreview(result);
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          t("modelCompare.previewError"),
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
      const result = await apiService.prepareModelComparison(
        leftFile,
        rightFile,
      );
      const nextSelectionState = {
        left: buildSelectionMap(result.left_artifact),
        right: buildSelectionMap(result.right_artifact),
      };
      setComparePayload(result);
      setSelectionState(nextSelectionState);
      setOverallMaeFeatureSelection(
        Object.fromEntries(
          (result.shared_features || []).map((featureName) => [featureName, true]),
        ),
      );
      setMeanShiftFeatureFactorSelection({});
      setMeanShiftFeatureFactorDraft({});
      setPreview(result.preview);
      setUseConfidence(Boolean(result.preview?.use_confidence));
      const firstFeature = result.shared_features?.[0] || "";
      setJumpFeature(firstFeature);
    } catch (err) {
      resetPreparedState();
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          t("modelCompare.prepareError"),
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

  const handleOverallMaeFeatureToggle = async (featureName, checked) => {
    const nextOverallMaeFeatureSelection = {
      ...overallMaeFeatureSelection,
      [featureName]: checked,
    };
    setOverallMaeFeatureSelection(nextOverallMaeFeatureSelection);
    await requestPreview(
      selectionState,
      useConfidence,
      false,
      nextOverallMaeFeatureSelection,
    );
  };

  const handleContributionFeatureToggle = async (featureName) => {
    const nextContributionFeatureSelection = {
      ...contributionFeatureSelection,
      [featureName]: contributionFeatureSelection[featureName] === false,
    };
    const nextMeanShiftFeatureFactorSelection = {
      ...meanShiftFeatureFactorSelection,
      [featureName]:
        contributionFeatureSelection[featureName] === false
          ? meanShiftFeatureFactorSelection[featureName]
          : false,
    };
    setContributionFeatureSelection(nextContributionFeatureSelection);
    setMeanShiftFeatureFactorSelection(nextMeanShiftFeatureFactorSelection);
    if (
      Object.values(nextMeanShiftFeatureFactorSelection).some(Boolean)
    ) {
      await requestPreview(
        selectionState,
        useConfidence,
        false,
        overallMaeFeatureSelection,
        nextContributionFeatureSelection,
        nextMeanShiftFeatureFactorSelection,
      );
    }
  };

  const handleMeanShiftFeatureFactorDraftToggle = (featureName) => {
    setMeanShiftFeatureFactorDraft((currentSelection) => ({
      ...currentSelection,
      [featureName]: currentSelection[featureName] !== true,
    }));
  };

  const handleOpenMeanShiftFeatureModal = () => {
    setMeanShiftFeatureFactorDraft(meanShiftFeatureFactorSelection);
    setMeanShiftFeatureModalOpen(true);
  };

  const handleApplyMeanShiftFeatureFactors = async () => {
    const nextMeanShiftFeatureFactorSelection = {
      ...meanShiftFeatureFactorDraft,
    };
    setMeanShiftFeatureFactorSelection(nextMeanShiftFeatureFactorSelection);
    setMeanShiftFeatureModalOpen(false);
    await requestPreview(
      selectionState,
      useConfidence,
      false,
      overallMaeFeatureSelection,
      contributionFeatureSelection,
      nextMeanShiftFeatureFactorSelection,
    );
  };

  const applyManualFeatureShift = async (featureName, shift) => {
    const nextShift = Math.abs(shift) < Number.EPSILON ? 0 : shift;
    const nextManualFeatureShifts = {
      ...manualFeatureShifts,
      [featureName]: nextShift,
    };
    setManualFeatureShifts(nextManualFeatureShifts);
    await requestPreview(
      selectionState,
      useConfidence,
      false,
      overallMaeFeatureSelection,
      contributionFeatureSelection,
      meanShiftFeatureFactorSelection,
      nextManualFeatureShifts,
    );
  };

  const handleManualFeatureShift = async (featureName, direction) => {
    const currentShift = Number(manualFeatureShifts[featureName] || 0);
    const nextShift = currentShift + direction * FEATURE_EDIT_SHIFT_STEP;
    setManualFeatureShiftInputValues((currentValues) => ({
      ...currentValues,
      [featureName]: String(nextShift),
    }));
    await applyManualFeatureShift(featureName, nextShift);
  };

  const handleManualFeatureShiftInputChange = (featureName, value) => {
    setManualFeatureShiftInputValues((currentValues) => ({
      ...currentValues,
      [featureName]: value,
    }));
  };

  const handleManualFeatureShiftInputCommit = async (featureName) => {
    const value = manualFeatureShiftInputValues[featureName];
    const parsedShift = Number(value);
    if (!Number.isFinite(parsedShift)) {
      setManualFeatureShiftInputValues((currentValues) => ({
        ...currentValues,
        [featureName]: String(manualFeatureShifts[featureName] || 0),
      }));
      return;
    }
    const nextShift = Math.abs(parsedShift) < Number.EPSILON ? 0 : parsedShift;
    setManualFeatureShiftInputValues((currentValues) => ({
      ...currentValues,
      [featureName]: String(nextShift),
    }));
    if (nextShift === Number(manualFeatureShifts[featureName] || 0)) return;
    await applyManualFeatureShift(featureName, nextShift);
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

  const averagePredictionScores =
    averageScoreMethod === "data_weighted"
      ? preview?.data_weighted_average_prediction_scores
      : preview?.average_prediction_scores;
  const selectedContributionRows =
    [...(preview?.average_prediction_feature_contributions?.[averageScoreMethod] || [])]
      .sort((left, right) => {
        if (left.is_base_offset !== right.is_base_offset) {
          return left.is_base_offset ? -1 : 1;
        }
        return Math.abs(right.difference) - Math.abs(left.difference);
      });
  const contributionFeatureRows = selectedContributionRows.filter(
    (row) => !row.is_base_offset,
  );
  const meanShiftModalFeatureRows = contributionFeatureRows.filter(
    (row) => contributionFeatureSelection[row.feature_name] !== false,
  );
  const selectedContributionFeatureCount = contributionFeatureRows.filter(
    (row) => contributionFeatureSelection[row.feature_name] !== false,
  ).length;
  const selectedContributionScoreRows = selectedContributionRows.filter(
    (row) =>
      row.is_base_offset || contributionFeatureSelection[row.feature_name] !== false,
  );
  const displayedContributionRows = selectedContributionRows;
  const selectedContributionAverageScores = selectedContributionScoreRows.length
    ? selectedContributionScoreRows.reduce(
        (scores, row) => ({
          original: scores.original + row.original,
          edited: scores.edited + row.edited,
          difference: scores.difference + row.difference,
        }),
        { original: 0, edited: 0, difference: 0 },
      )
    : averagePredictionScores;
  const averagePredictionRelativeChange =
    Number.isFinite(selectedContributionAverageScores?.original) &&
    Number.isFinite(selectedContributionAverageScores?.edited) &&
    Math.abs(selectedContributionAverageScores.original) > Number.EPSILON
      ? ((selectedContributionAverageScores.edited -
          selectedContributionAverageScores.original) /
          Math.abs(selectedContributionAverageScores.original)) *
        100
      : null;
  const contributionSelectionIsActive =
    contributionFeatureRows.length > 0 &&
    selectedContributionFeatureCount < contributionFeatureRows.length;
  const selectedContributionDifference = contributionFeatureRows
    .filter((row) => contributionFeatureSelection[row.feature_name] !== false)
    .reduce((sum, row) => sum + row.difference, 0);

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
                {loading
                  ? t("modelCompare.preparing")
                  : t("modelCompare.compareArtifacts")}
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

            <OverallShapeFunctionMaeSummary
              metrics={preview?.overall_shape_function_mae}
              featureNames={comparePayload.shared_features}
              featurePreviews={preview?.feature_previews}
              featureSelection={overallMaeFeatureSelection}
              onFeatureSelectionChange={handleOverallMaeFeatureToggle}
              previewLoading={previewLoading}
              language={language}
              t={t}
            />

            <section className="rounded-2xl border border-indigo-200 bg-indigo-50/60 p-5 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">
                    {t("modelCompare.averagePredictedScore")}
                  </h2>
                  <p className="text-sm text-slate-600">
                    {averageScoreMethod === "uniform"
                      ? t("modelCompare.uniformAverageDescription")
                      : t("modelCompare.dataWeightedAverageDescription")}
                  </p>
                </div>
                {previewLoading && (
                  <div className="text-sm text-slate-500">
                    {t("modelCompare.recomputing")}
                  </div>
                )}
              </div>
              <div className="mt-4 inline-flex rounded-xl border border-indigo-200 bg-white p-1">
                <button
                  type="button"
                  onClick={() => setAverageScoreMethod("uniform")}
                  className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    averageScoreMethod === "uniform"
                      ? "bg-indigo-600 text-white"
                      : "text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  {t("modelCompare.uniformAverage")}
                </button>
                <button
                  type="button"
                  onClick={() => setAverageScoreMethod("data_weighted")}
                  disabled={!preview?.data_weighted_average_prediction_scores}
                  title={
                    preview?.data_weighted_average_prediction_scores
                      ? ""
                      : t("modelCompare.dataWeightedUnavailable")
                  }
                  className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    averageScoreMethod === "data_weighted"
                      ? "bg-indigo-600 text-white"
                      : "text-slate-700 hover:bg-slate-100"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  {t("modelCompare.dataWeightedAverage")}
                </button>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3 rounded-xl border border-indigo-200 bg-white px-4 py-3">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-slate-900">
                    {t("modelCompare.featureMeanShiftLimit")}
                  </div>
                  <div className="text-xs text-slate-600">
                    {t("modelCompare.meanShiftFeatureLimitDescription")}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleOpenMeanShiftFeatureModal}
                  disabled={!meanShiftModalFeatureRows.length}
                  className="rounded-xl border border-indigo-300 bg-indigo-50 px-3 py-2 text-sm font-medium text-indigo-800 transition-colors hover:bg-indigo-100 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {t("modelCompare.chooseMeanShiftFeatures")}
                </button>
              </div>
              {Object.keys(
                preview?.mean_deviation_constraint?.feature_scale_factors || {},
              ).length > 0 && (
                <p className="mt-2 text-xs text-slate-600">
                  {preview.mean_deviation_constraint.applied
                    ? (
                        <>
                          {t("modelCompare.meanDeviationScaled", {
                            percent: Math.round(
                              preview.mean_deviation_constraint.scale_factor * 100,
                            ),
                          })}{" "}
                          {t("modelCompare.editScaleFactor", {
                            factor: formatMetricValue(
                              preview.mean_deviation_constraint.scale_factor,
                              language,
                            ),
                          })}
                        </>
                      )
                    : preview.mean_deviation_constraint.satisfied
                      ? t("modelCompare.meanDeviationWithinLimit")
                      : t("modelCompare.meanDeviationCannotMeetLimit")}
                </p>
              )}
              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-blue-200 bg-white px-4 py-3">
                  <div className="text-xs font-semibold uppercase tracking-wide text-blue-700">
                    {t("modelCompare.originalAveragePredictedScore")}
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-blue-950">
                    {formatMetricValue(
                      selectedContributionAverageScores?.original,
                      language,
                    )}
                  </div>
                </div>
                <div className="rounded-xl border border-amber-200 bg-white px-4 py-3">
                  <div className="text-xs font-semibold uppercase tracking-wide text-amber-700">
                    {t("modelCompare.editedAveragePredictedScore")}
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-amber-950">
                    {formatMetricValue(
                      selectedContributionAverageScores?.edited,
                      language,
                    )}
                  </div>
                </div>
                <div className="rounded-xl border border-indigo-200 bg-white px-4 py-3">
                  <div className="text-xs font-semibold uppercase tracking-wide text-indigo-700">
                    {t("modelCompare.averagePredictedScoreDifference")}
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-indigo-950">
                    {formatSignedMetricValue(
                      selectedContributionAverageScores?.difference,
                      language,
                    )}
                  </div>
                  <div className="mt-1 text-xs font-medium text-indigo-700">
                    {t("modelCompare.averagePredictionRelativeChange", {
                      change: formatRelativePercentageChange(
                        averagePredictionRelativeChange,
                        language,
                      ),
                    })}
                  </div>
                </div>
              </div>
              {contributionSelectionIsActive && (
                <p className="mt-2 text-xs text-indigo-700">
                  {t("modelCompare.selectedContributionAverageView")}
                </p>
              )}
              <details className="mt-4 rounded-xl border border-indigo-200 bg-white px-4 py-3">
                <summary className="cursor-pointer text-sm font-semibold text-indigo-950">
                  {t("modelCompare.showAverageContributions")}
                </summary>
                <p className="mt-2 text-sm text-slate-600">
                  {t("modelCompare.averageContributionsDescription")}
                </p>
                <div className="mt-3 rounded-xl border border-indigo-100 bg-indigo-50/50 p-3">
                  <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                    <div className="text-sm font-semibold text-indigo-950">
                      {t("modelCompare.contributionFeatureSelection", {
                        selected: selectedContributionFeatureCount,
                        total: contributionFeatureRows.length,
                      })}
                    </div>
                    <div className="text-sm text-indigo-900">
                      {t("modelCompare.selectedContributionDifference")}: {" "}
                      <span className="font-semibold tabular-nums">
                        {formatSignedMetricValue(
                          selectedContributionDifference,
                          language,
                        )}
                      </span>
                    </div>
                  </div>
                  <p className="mt-1 text-xs text-slate-600">
                    {t("modelCompare.contributionFeatureSelectionDescription")}
                  </p>
                </div>
                <div className="mt-3 overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                      <tr>
                        <th className="px-2 py-2 font-semibold">
                          {t("modelCompare.contributionFeature")}
                        </th>
                        <th className="px-2 py-2 text-right font-semibold">
                          {t("modelCompare.originalAveragePredictedScore")}
                        </th>
                        <th className="px-2 py-2 text-right font-semibold">
                          {t("modelCompare.editedAveragePredictedScore")}
                        </th>
                        <th className="px-2 py-2 text-right font-semibold">
                          {t("modelCompare.averagePredictedScoreDifference")}
                        </th>
                        <th className="px-2 py-2 text-right font-semibold">
                          {t("modelCompare.featureEditShift")}
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 text-slate-700">
                      {displayedContributionRows.map((row) => {
                        const isSelected =
                          row.is_base_offset ||
                          contributionFeatureSelection[row.feature_name] !== false;
                        const rowClassName = row.is_base_offset
                          ? "bg-slate-50"
                          : !isSelected
                            ? "cursor-pointer bg-slate-50 text-slate-400 grayscale"
                            : "cursor-pointer hover:bg-indigo-50";
                        return (
                          <tr
                            key={row.feature_name}
                            onClick={
                              row.is_base_offset
                                ? undefined
                                : () =>
                                    handleContributionFeatureToggle(
                                      row.feature_name,
                                    )
                            }
                            className={rowClassName}
                          >
                            <td className="px-2 py-2 font-medium">
                              {row.is_base_offset
                                ? t("modelCompare.basePredictionOffset")
                                : row.feature_name}
                            </td>
                            <td className="px-2 py-2 text-right tabular-nums">
                              {formatMetricValue(row.original, language)}
                            </td>
                            <td className="px-2 py-2 text-right tabular-nums">
                              {formatMetricValue(row.edited, language)}
                            </td>
                            <td className="px-2 py-2 text-right tabular-nums">
                              {formatSignedMetricValue(row.difference, language)}
                            </td>
                            <td className="px-2 py-2 text-right">
                              {!row.is_base_offset && (
                                <div className="inline-flex items-center gap-1">
                                  <button
                                    type="button"
                                    disabled={!isSelected || previewLoading}
                                    aria-label={t(
                                      "modelCompare.decreaseFeatureEditShift",
                                    )}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleManualFeatureShift(
                                        row.feature_name,
                                        -1,
                                      );
                                    }}
                                    className="rounded border border-slate-300 px-2 py-1 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
                                  >
                                    −
                                  </button>
                                  <input
                                    type="number"
                                    step="any"
                                    disabled={!isSelected || previewLoading}
                                    aria-label={t("modelCompare.featureEditShift")}
                                    value={
                                      manualFeatureShiftInputValues[
                                        row.feature_name
                                      ] ??
                                      String(
                                        manualFeatureShifts[row.feature_name] || 0,
                                      )
                                    }
                                    onClick={(event) => event.stopPropagation()}
                                    onChange={(event) =>
                                      handleManualFeatureShiftInputChange(
                                        row.feature_name,
                                        event.target.value,
                                      )
                                    }
                                    onBlur={() =>
                                      handleManualFeatureShiftInputCommit(
                                        row.feature_name,
                                      )
                                    }
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter") {
                                        event.currentTarget.blur();
                                      }
                                    }}
                                    className="w-20 rounded border border-slate-300 bg-white px-1 py-1 text-center text-xs tabular-nums text-slate-700 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
                                  />
                                  <button
                                    type="button"
                                    disabled={!isSelected || previewLoading}
                                    aria-label={t(
                                      "modelCompare.increaseFeatureEditShift",
                                    )}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleManualFeatureShift(
                                        row.feature_name,
                                        1,
                                      );
                                    }}
                                    className="rounded border border-slate-300 px-2 py-1 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40"
                                  >
                                    +
                                  </button>
                                </div>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </details>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  {t("modelCompare.chartControls")}
                </h2>
                <p className="text-sm text-slate-600">
                  {t("modelCompare.chartControlsDescription")}
                </p>
              </div>

              <div className="mt-4 flex flex-col gap-3">
                  <div className="flex flex-wrap items-center gap-3">
                  <label className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                    <input
                      type="checkbox"
                      checked={useConfidence}
                      onChange={(event) =>
                        handleConfidenceToggle(event.target.checked)
                      }
                      className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    <div>
                      <div className="text-sm font-medium text-slate-800">
                        {useConfidence
                          ? t("modelCompare.confidenceOn")
                          : t("modelCompare.confidenceOff")}
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
                  </button>

                  <button
                    type="button"
                    onClick={() => setCenterCharts((current) => !current)}
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      centerCharts
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {centerCharts
                        ? t("modelCompare.restoreChartPositions")
                        : t("modelCompare.centerCharts")}
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setShowLineMarkers((current) => !current)}
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showLineMarkers
                        ? "border-slate-200 bg-slate-50 text-slate-800 hover:bg-slate-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {showLineMarkers
                        ? t("modelCompare.hideLineDots")
                        : t("modelCompare.showLineDots")}
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setShowIndividualEditTraces((current) => !current)
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showIndividualEditTraces
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {showIndividualEditTraces
                        ? t("modelCompare.showAggregatedEdits")
                        : t("modelCompare.showIndividualEdits")}
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setShowIndividualParticipantMae((current) => !current)
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showIndividualParticipantMae
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {showIndividualParticipantMae
                        ? t("modelCompare.hideIndividualMae")
                        : t("modelCompare.showIndividualMae")}
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      setShowConfidenceInLegend((current) => !current)
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showConfidenceInLegend
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {showConfidenceInLegend
                        ? t("modelCompare.hideConfidenceInLegend")
                        : t("modelCompare.showConfidenceInLegend")}
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() => setLegendOnRight((current) => !current)}
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      legendOnRight
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {legendOnRight
                        ? t("modelCompare.legendAboveChart")
                        : t("modelCompare.legendRightOfChart")}
                    </div>
                  </button>
                  </div>

                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
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
                        {(comparePayload.shared_features || []).map(
                          (featureName) => (
                            <option key={featureName} value={featureName}>
                              {featureName}
                            </option>
                          ),
                        )}
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
                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                          <FeatureMetricBadge
                            label={t("modelCompare.maeOriginalVsEdited")}
                            metricLabel={t("combined.mae")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_mae
                            }
                            baselineValue={
                              featurePreview?.comparison_metrics
                                ?.baseline_edited_vs_original_base_mae
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_status
                            }
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.rmseOriginalVsEdited")}
                            metricLabel={t("combined.rmse")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_rmse
                            }
                            baselineValue={
                              featurePreview?.comparison_metrics
                                ?.baseline_edited_vs_original_base_rmse
                            }
                            status={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_rmse_status
                            }
                            language={language}
                            t={t}
                          />
                          <FeatureMetricBadge
                            label={t("modelCompare.r2OriginalVsEdited")}
                            metricLabel={t("combined.r2")}
                            value={
                              featurePreview?.comparison_metrics
                                ?.edited_vs_original_base_r2
                            }
                            baselineValue={
                              featurePreview?.comparison_metrics
                                ?.baseline_edited_vs_original_base_r2
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
                        {showIndividualParticipantMae && (
                          <IndividualParticipantMaeList
                            featurePreview={featurePreview}
                            language={language}
                            t={t}
                          />
                        )}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 xl:justify-end">
                        <button
                          type="button"
                          onClick={() =>
                            openSelectionModal(
                              "left",
                              featurePreview.feature_name,
                            )
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
                          {
                            getFeatureSubmissions(
                              comparePayload.left_artifact,
                              featurePreview.feature_name,
                            ).length
                          }
                          )
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            openSelectionModal(
                              "right",
                              featurePreview.feature_name,
                            )
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
                          {
                            getFeatureSubmissions(
                              comparePayload.right_artifact,
                              featurePreview.feature_name,
                            ).length
                          }
                          )
                        </button>
                      </div>
                    </div>
                    <CompareFeatureChart
                      featurePreview={featurePreview}
                      showBaseTraces={showBaseTraces}
                      centerCharts={centerCharts}
                      showIndividualEditTraces={showIndividualEditTraces}
                      showConfidenceInLegend={showConfidenceInLegend}
                      legendOnRight={legendOnRight}
                      showLineMarkers={showLineMarkers}
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
      <MeanShiftFeatureSelectionModal
        open={meanShiftFeatureModalOpen}
        featureRows={meanShiftModalFeatureRows}
        selectionMap={meanShiftFeatureFactorDraft}
        busy={previewLoading}
        t={t}
        onClose={() => setMeanShiftFeatureModalOpen(false)}
        onToggleFeature={handleMeanShiftFeatureFactorDraftToggle}
        onApply={handleApplyMeanShiftFeatureFactors}
      />
    </div>
  );
}

export default ModelComparePage;
