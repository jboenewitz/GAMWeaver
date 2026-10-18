import React, { useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import apiService from "../api/apiService";
import { createTranslator, getDateLocale, getNumberLocale } from "../i18n";
import { getShapeFunctionDisplayName } from "../utils/featureDisplay";
import { getReadingSpellingDifficultyAxisTicks } from "../utils/featureAxisTicks";
import {
  getFeatureChartExperts,
  isChartExpertVisible,
} from "../utils/chartExpertVisibility";
import { getAveragePredictionScores } from "../utils/averagePredictionAdjustment";
import { buildExpertConfidenceBars } from "../utils/expertConfidence";
import { HELD_OUT_COMPARISONS, getHeldOutMetricComparison } from "../utils/heldOutComparison";
import { buildSharedHeldOutArtifacts, needsIqbSharedHeldOut } from "../utils/sharedHeldOutEvaluation";

const LEFT_COLORS = {
  base: "#2563eb",
};

const RIGHT_COLORS = {
  base: "#f59e0b",
  effective: "#d97706",
};

const REDUCED_SCATTER_COLOR = "#16a34a";
const getChartHeight = (height, tallCharts) =>
  tallCharts ? Math.round(height * 1.5) : height;

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
  { value: 5, label: "201-500" },
  { value: 6, label: "more than 500" },
];

const GERMAN_SPOKEN_AT_HOME_SCALE_TICKS = [
  { value: 1, label: "Never" },
  { value: 2, label: "Sometimes<br>(bilingual)" },
  { value: 3, label: "Always" },
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

const getNormalizedFeatureIdentity = (featurePreview) =>
  [
    featurePreview?.feature_name,
    featurePreview?.left_chart_config?.display_title,
    featurePreview?.right_chart_config?.display_title,
  ]
    .filter(Boolean)
    .join(" ")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();

const isBooksAtHomeFeature = (featurePreview) => {
  const featureIdentity = getNormalizedFeatureIdentity(featurePreview);
  return (
    featureIdentity.includes("kbuecher") ||
    featureIdentity.includes("bucher zu hause") ||
    featureIdentity.includes("books at home")
  );
};

const isGermanSpokenAtHomeFeature = (featurePreview) => {
  const featureIdentity = getNormalizedFeatureIdentity(featurePreview);
  return (
    featureIdentity.includes("german spoken at home") ||
    featureIdentity.includes("zu hause gesprochenes deutsch")
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
  const label = String(userName || "")
    .trim()
    .toUpperCase();
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
  return expertMatch ? `Expert${expertMatch[1]} (E${expertMatch[1]})` : label;
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
    const mean =
      yValues.reduce((sum, value) => sum + value, 0) / yValues.length;
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
  hiddenExpertKeys,
  showBaseTraces,
  centerCharts,
  showIndividualEditTraces,
  showAlternateConfidenceAggregation,
  useConfidence,
  showConfidenceInLegend,
  legendOnRight,
  showLineMarkers,
  tallCharts,
  t,
}) => {
  const featureTitle = buildFeatureTitle(featurePreview);
  const isCategorical = featurePreview?.feature_type === "categorical";
  const participantEditCounts = new Map();
  [
    ...(featurePreview.left_individual_edit_traces || []),
    ...(featurePreview.right_individual_edit_traces || []),
  ]
    .filter((trace) => isChartExpertVisible(trace, hiddenExpertKeys))
    .forEach((editTrace) => {
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

    return editTraces
      .filter((trace) => isChartExpertVisible(trace, hiddenExpertKeys))
      .map((editTrace) => {
        const participantKey = editTrace.user_name || editTrace.submission_id;
        const participantLabel = formatChartExpertLabel(participantKey);
        const color = getIndividualEditColor(participantKey);
        const confidence = Number(editTrace.sureness) || 0;
        const showSubmissionId = participantEditCounts.get(participantKey) > 1;
        const name = `${participantLabel}${
          showConfidenceInLegend
            ? ` — ${t("modelCompare.confidence")}: ${confidence}/10`
            : ""
        }${showSubmissionId ? ` (${editTrace.submission_id})` : ""}`;
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

  const buildAggregatedTrace = (yValues, name, alternate = false) =>
    isCategorical
      ? {
          x: featurePreview.right_x_values,
          y: yValues,
          type: "bar",
          name,
          marker: {
            color: alternate ? "#7c3aed" : RIGHT_COLORS.effective,
            opacity: 0.95,
          },
          customdata:
            featurePreview.right_x_tick_labels || featurePreview.right_x_values,
          hovertemplate: "<b>%{customdata}</b><br>%{y:.3f}<extra>" + name + "</extra>",
        }
      : {
          x: featurePreview.right_x_values,
          y: yValues,
          type: "scatter",
          mode: showLineMarkers ? "lines+markers" : "lines",
          name,
          marker: { color: alternate ? "#7c3aed" : RIGHT_COLORS.effective, size: 5 },
          line: {
            color: alternate ? "#7c3aed" : RIGHT_COLORS.effective,
            width: 3,
            ...(alternate ? { dash: "dot" } : {}),
          },
        };
  const aggregateName = t("modelCompare.aggregatedParticipantEdits");
  const confidenceModeLabel = (confidence) =>
    t(
      confidence
        ? "modelCompare.allFeatureMaeChangeWithConfidence"
        : "modelCompare.allFeatureMaeChangeWithoutConfidence",
    );
  const effectiveTraces = showIndividualEditTraces
    ? individualEditTraces
    : [
        buildAggregatedTrace(
          featurePreview.right_effective_y_values,
          showAlternateConfidenceAggregation
            ? `${aggregateName} — ${confidenceModeLabel(useConfidence)}`
            : aggregateName,
        ),
        ...(showAlternateConfidenceAggregation &&
        featurePreview.right_alternate_confidence_y_values
          ? [
              buildAggregatedTrace(
                featurePreview.right_alternate_confidence_y_values,
                `${aggregateName} — ${confidenceModeLabel(!useConfidence)}`,
                true,
              ),
            ]
          : []),
      ];

  const data = showBaseTraces
    ? [...baseTraces, ...effectiveTraces]
    : effectiveTraces;
  const chartData = centerChartTraces(data, centerCharts);
  const booksAtHome = isBooksAtHomeFeature(featurePreview);
  const germanSpokenAtHome = isGermanSpokenAtHomeFeature(featurePreview);
  const scaleTicks = booksAtHome
    ? BOOKS_AT_HOME_SCALE_TICKS
    : germanSpokenAtHome
      ? GERMAN_SPOKEN_AT_HOME_SCALE_TICKS
      : usesOrdinalScale(featurePreview, 1, 4)
        ? FOUR_POINT_SCALE_TICKS
        : null;
  const hasFixedOrdinalScale =
    scaleTicks &&
    (booksAtHome ||
      germanSpokenAtHome ||
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
  const readingSpellingTicks = getReadingSpellingDifficultyAxisTicks(
    featurePreview?.feature_name,
    [
      ...(featurePreview?.left_x_values || []),
      ...(featurePreview?.right_x_values || []),
    ],
    isCategorical,
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
      ...(readingSpellingTicks || {}),
    },
    yaxis: {
      title: t("shapeFunctions.effectOnPrediction"),
      gridcolor: "#e2e8f0",
      zeroline: true,
      zerolinecolor: "#94a3b8",
      ...getEffectYAxisScale(chartData),
    },
    height: getChartHeight(360, tallCharts),
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
            {t("modelCompare.reducedVsFullData")}:{" "}
            {formatMetricValue(baselineValue, language)}
          </div>
          <div>
            {t("modelCompare.deltaFromBaseline")}:{" "}
            {formatSignedMetricValue(metricDelta, language)}
          </div>
          <div>
            {t("modelCompare.relativeChange")}:{" "}
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

const ExpertCompleteModelMaeList = ({
  experts,
  useConfidence,
  language,
  t,
}) => {
  if (!experts?.length) return null;

  return (
    <details className="mt-4 rounded-xl border border-emerald-200 bg-white px-4 py-3">
      <summary className="cursor-pointer text-sm font-semibold text-emerald-950">
        {t("modelCompare.expertCompleteModelMae")}
      </summary>
      <p className="mt-2 text-sm text-slate-600">
        {t("modelCompare.expertCompleteModelMaeDescription")}
      </p>
      <p className="mt-1 text-xs text-slate-500">
        {t(
          useConfidence
            ? "modelCompare.expertCompleteModelMaeConfidenceWeighted"
            : "modelCompare.expertCompleteModelMaeEqualWeighted",
        )}
      </p>
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {experts.map((expert) => (
          <div
            key={expert.user_name}
            className="rounded-lg border border-emerald-100 bg-emerald-50/40 px-3 py-2 text-sm text-slate-800"
          >
            <div className="font-medium">
              {formatChartExpertLabel(expert.user_name)}
            </div>
            <div className="mt-2">
              <FeatureMetricBadge
                label={t("modelCompare.expertCompleteModelMae")}
                metricLabel={t("combined.mae")}
                value={expert.mae}
                baselineValue={expert.baseline_mae}
                status={expert.status}
                language={language}
                t={t}
              />
            </div>
            <div className="mt-1 text-xs text-slate-600">
              {t("modelCompare.expertCompleteModelMaeCoverage", {
                features: expert.feature_count || 0,
                points: expert.point_count || 0,
                submissions: expert.submission_count || 0,
              })}
            </div>
          </div>
        ))}
      </div>
    </details>
  );
};

const OverallShapeFunctionMaeSummary = ({
  metrics,
  featureNames,
  featurePreviews,
  featureSelection,
  onFeatureSelectionChange,
  previewLoading,
  useConfidence,
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
            const featureMetrics =
              featurePreviewByName.get(featureName)?.comparison_metrics;
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
                disabled={
                  previewLoading || (selectedFeatureCount === 1 && isSelected)
                }
                onClick={() =>
                  onFeatureSelectionChange(featureName, !isSelected)
                }
                className={`flex min-w-0 items-center rounded-lg border px-3 py-2 text-left text-sm transition-opacity focus:outline-none focus:ring-2 focus:ring-indigo-400 disabled:cursor-not-allowed ${
                  isSelected
                    ? getOverallMaeFeatureTone(
                        featureMetrics?.edited_vs_original_base_status,
                      )
                    : "border-slate-200 bg-slate-100 text-slate-500 opacity-50 grayscale"
                }`}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">
                    {featureName}
                  </span>
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
      <ExpertCompleteModelMaeList
        experts={metrics.expert_complete_model_mae}
        useConfidence={useConfidence}
        language={language}
        t={t}
      />
    </section>
  );
};

const getActualVsPredictedRange = (values) => {
  const finiteValues = (values || []).filter((value) => Number.isFinite(value));
  if (!finiteValues.length) return null;

  const minimum = Math.min(...finiteValues);
  const maximum = Math.max(...finiteValues);
  const span = Math.max(maximum - minimum, 1);
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
  let rangeStart = Math.floor(minimum / dtick) * dtick;
  let rangeEnd = Math.ceil(maximum / dtick) * dtick;

  if (rangeStart === rangeEnd) {
    rangeStart -= dtick;
    rangeEnd += dtick;
  }

  return { range: [rangeStart, rangeEnd], dtick, tick0: 0 };
};

const getInitialHeldOutTestSource = (leftArtifact, hasMainTrainingData) => {
  const leftTestData = leftArtifact?.held_out_test_data;
  const hasLeftTestData =
    Array.isArray(leftTestData?.feature_rows) &&
    leftTestData.feature_rows.length > 0 &&
    Array.isArray(leftTestData?.target_values) &&
    leftTestData.target_values.length === leftTestData.feature_rows.length;
  return hasLeftTestData || !hasMainTrainingData ? "left" : "main_training";
};

const HeldOutActualVsPredictedChart = ({ performance, visibleModels, tallCharts, t }) => {
  const buildPairs = (predictions) =>
    (performance?.actual_values || []).reduce((pairs, actual, index) => {
      const predicted = predictions?.[index];
      if (Number.isFinite(actual) && Number.isFinite(predicted)) {
        pairs.push({ actual, predicted });
      }
      return pairs;
    }, []);
  const baselinePairs = buildPairs(performance?.baseline_predictions);
  const reducedPairs = buildPairs(performance?.reduced_predictions);
  const editedPairs = buildPairs(performance?.edited_predictions);
  const allValues = [...baselinePairs, ...reducedPairs, ...editedPairs].flatMap((pair) => [
    pair.actual,
    pair.predicted,
  ]);
  const yAxisScale = getActualVsPredictedRange(allValues);
  if (!yAxisScale) return null;
  const xAxisRange = [0, 800];
  const identityTrace = {
    x: xAxisRange,
    y: xAxisRange,
    type: "scatter",
    mode: "lines",
    name: t("modelCompare.actualVsPredictedIdentityLine"),
    line: { color: "#64748b", dash: "dash", width: 2 },
    hoverinfo: "skip",
  };
  const baselineTrace = {
    x: baselinePairs.map((pair) => pair.actual),
    y: baselinePairs.map((pair) => pair.predicted),
    type: "scatter",
    mode: "markers",
    name: t("modelCompare.fullData"),
    marker: { color: LEFT_COLORS.base, size: 8, opacity: 0.6 },
  };
  const reducedTrace = {
    x: reducedPairs.map((pair) => pair.actual),
    y: reducedPairs.map((pair) => pair.predicted),
    type: "scatter",
    mode: "markers",
    name: t("modelCompare.reducedData"),
    marker: { color: REDUCED_SCATTER_COLOR, size: 8, opacity: 0.6 },
  };
  const editedTrace = {
    x: editedPairs.map((pair) => pair.actual),
    y: editedPairs.map((pair) => pair.predicted),
    type: "scatter",
    mode: "markers",
    name: t("modelCompare.editedReducedData"),
    marker: { color: RIGHT_COLORS.effective, size: 8, opacity: 0.6 },
  };
  const modelTraces = {
    full: baselineTrace,
    reduced: reducedTrace,
    edited: editedTrace,
  };
  const modelLabelKeys = {
    full: "fullData",
    reduced: "reducedData",
    edited: "editedReducedData",
  };
  const renderChart = (traces, title) => (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        title: {
          text: title,
          font: { size: 16, color: "#0f172a" },
        },
        xaxis: {
          title: t("modelCompare.actualValues"),
          gridcolor: "#e2e8f0",
          range: xAxisRange,
        },
        yaxis: {
          title: t("modelCompare.predictedValues"),
          gridcolor: "#e2e8f0",
          ...yAxisScale,
        },
        height: getChartHeight(380, tallCharts),
        margin: { l: 60, r: 24, t: 60, b: 60 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        legend: { orientation: "h", x: 0, y: 1.13 },
      }}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
      style={{ width: "100%" }}
      className="w-full"
    />
  );

  return (
    <>
      <div className="mt-4 rounded-2xl border border-violet-200 bg-white p-3">
        {renderChart(
          [identityTrace, ...visibleModels.map((modelKey) => modelTraces[modelKey])],
          t("modelCompare.actualVsPredicted"),
        )}
      </div>
      <div className={`mt-4 grid gap-4 lg:grid-cols-2 ${visibleModels.length === 3 ? "xl:grid-cols-3" : ""}`}>
        {visibleModels.map((modelKey) => (
          <div key={modelKey} className="min-w-0 rounded-2xl border border-violet-200 bg-white p-3">
            {renderChart(
              [identityTrace, modelTraces[modelKey]],
              `${t("modelCompare.actualVsPredicted")} — ${t(`modelCompare.${modelLabelKeys[modelKey]}`)}`,
            )}
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-2xl border border-violet-200 bg-white p-3">
        <Plot data={[
          { x: xAxisRange, y: [0, 0], type: "scatter", mode: "lines", name: t("modelCompare.zeroResidual"), line: { color: "#64748b", dash: "dash" }, hoverinfo: "skip" },
          ...[["reduced", reducedPairs, REDUCED_SCATTER_COLOR, "reducedData"], ["edited", editedPairs, RIGHT_COLORS.effective, "editedReducedData"]]
            .map(([, pairs, color, label]) => ({ x: pairs.map((pair) => pair.actual), y: pairs.map((pair) => pair.predicted - pair.actual), type: "scatter", mode: "markers", name: t(`modelCompare.${label}`), marker: { color, size: 8, opacity: 0.6 } })),
        ]} layout={{ autosize: true, title: t("modelCompare.residualComparison"), xaxis: { title: t("modelCompare.actualValues"), range: xAxisRange }, yaxis: { title: t("modelCompare.residualAxis") }, height: getChartHeight(340, tallCharts), margin: { l: 60, r: 24, t: 60, b: 60 }, paper_bgcolor: "white", plot_bgcolor: "white" }} config={{ responsive: true, displayModeBar: false }} useResizeHandler style={{ width: "100%" }} />
      </div>
    </>
  );
};

const HeldOutTestPerformanceSummary = ({
  performance,
  hasOriginalTestData,
  hasMainTrainingData,
  testDataSource,
  onTestDataSourceChange,
  loading,
  onEvaluate,
  useConfidence,
  requiresSharedLineage,
  sharedLineage,
  tallCharts,
  language,
  t,
}) => {
  const [selectedComparison, setSelectedComparison] = useState("reducedEdited");
  const comparison = HELD_OUT_COMPARISONS[selectedComparison];
  const hasHeldOutTestData =
    testDataSource === "left" ? hasOriginalTestData : hasMainTrainingData;
  const isMainTrainingData = testDataSource === "main_training";
  const needsLineageCheck = !isMainTrainingData && requiresSharedLineage && !sharedLineage;

  return (
    <section className="rounded-2xl border border-violet-200 bg-violet-50/60 p-5 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            {t(isMainTrainingData
              ? "modelCompare.mainTrainingPerformance"
              : "modelCompare.heldOutTestPerformance")}
          </h2>
          <p className="text-sm text-slate-600">
            {t(isMainTrainingData
              ? "modelCompare.mainTrainingPerformanceDescription"
              : "modelCompare.heldOutTestPerformanceDescription")}
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:items-end">
          <label className="text-xs font-medium text-slate-700">
            <span className="sr-only">
              {t("modelCompare.heldOutTestDataSource")}
            </span>
            <select
              value={testDataSource}
              disabled={loading}
              onChange={(event) => onTestDataSourceChange(event.target.value)}
              className="w-full rounded-xl border border-violet-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-200"
            >
              <option value="left" disabled={!hasOriginalTestData}>
                {t("modelCompare.useOriginalArtifactTestSet")}
              </option>
              <option value="main_training" disabled={!hasMainTrainingData}>
                {t("modelCompare.useMainTrainingData")}
              </option>
            </select>
          </label>
          <button
            type="button"
            disabled={!hasHeldOutTestData || loading || needsLineageCheck}
            onClick={onEvaluate}
            className="rounded-xl bg-violet-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-violet-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading
              ? t("modelCompare.recomputing")
              : t("modelCompare.evaluateHeldOutTest")}
          </button>
        </div>
      </div>
      {needsLineageCheck && (
        <p className="mt-3 text-sm text-amber-800">
          {t("modelCompare.sharedHeldOutNeedsLineage")}
        </p>
      )}
      {sharedLineage && !isMainTrainingData && (
        <p className="mt-3 text-sm text-emerald-800">
          {t("modelCompare.sharedHeldOutReady", { samples: sharedLineage.shared_test_row_count })}
        </p>
      )}
      {!hasHeldOutTestData ? (
        <p className="mt-4 rounded-xl border border-dashed border-violet-300 bg-white px-4 py-3 text-sm text-slate-600">
          {t("modelCompare.heldOutTestUnavailable")}
        </p>
      ) : loading ? (
        <p className="mt-4 text-sm text-slate-600">
          {t("modelCompare.recomputing")}
        </p>
      ) : performance ? (
        <>
          <p className="mt-4 text-xs text-slate-600">
            {t(isMainTrainingData
              ? "modelCompare.mainTrainingPerformanceCoverage"
              : "modelCompare.heldOutTestCoverage", {
              samples: performance.test_size || 0,
            })}
          </p>
          <p className="mt-2 text-xs text-slate-600">{t("modelCompare.meanErrorConvention")}</p>
          <p className="mt-2 text-xs text-slate-600">
            {t(useConfidence
              ? "modelCompare.heldOutConfidenceWeighted"
              : "modelCompare.heldOutUnweighted")}
          </p>
          <div className="mt-3 overflow-x-auto rounded-xl bg-white p-3">
            <table className="w-full min-w-[650px] text-left text-sm">
              <thead><tr className="border-b border-slate-200">
                <th className="p-2">{t("modelCompare.model")}</th>
                {["mae", "rmse", "r2", "mean_error", "std_dev"].map((key) => <th key={key} className="p-2">{t(`modelCompare.metric.${key}`)}</th>)}
              </tr></thead>
              <tbody>{[["fullData", performance.baseline], ["reducedData", performance.reduced], ["editedReducedData", performance.edited]].map(([label, metrics]) => (
                <tr key={label} className="border-b border-slate-100 last:border-0">
                  <th className="p-2 font-medium">{t(`modelCompare.${label}`)}</th>
                  {["mae", "rmse", "r2", "mean_error", "std_dev"].map((key) => <td key={key} className="p-2 tabular-nums">{key === "mean_error" ? formatSignedMetricValue(metrics?.[key], language) : formatMetricValue(metrics?.[key], language)}</td>)}
                </tr>
              ))}</tbody>
            </table>
          </div>
          <div className="mt-4 flex flex-wrap gap-2" role="group" aria-label={t("modelCompare.selectedModelComparison")}>
            {Object.keys(HELD_OUT_COMPARISONS).map((option) => (
              <button key={option} type="button"
                aria-pressed={selectedComparison === option}
                onClick={() => setSelectedComparison(option)}
                className={`rounded-lg px-3 py-2 text-sm ${selectedComparison === option ? "bg-violet-700 text-white" : "bg-white text-violet-900"}`}
              >{t(`modelCompare.${option}`)}</button>
            ))}
          </div>
          <div className="mt-4 rounded-xl border-2 border-violet-400 bg-white p-4">
            <h3 className="font-semibold text-violet-950">{t(`modelCompare.${comparison.headingKey}`)}</h3>
            <p className="text-xs text-slate-600">{t(`modelCompare.${comparison.descriptionKey}`)}</p>
            <div className="mt-3 grid gap-3 md:grid-cols-2 xl:w-[calc(50%_-_0.375rem)]">
              {["mae", "rmse", "r2", "mean_error", "std_dev"].map((key) => {
                const { models, differences } = getHeldOutMetricComparison(performance, selectedComparison, key);
                const onlyStatus = differences.length === 1 ? differences[0].status : null;
                const cardTone = onlyStatus === "improved"
                  ? "border-emerald-300 bg-emerald-50"
                  : onlyStatus === "worsened"
                    ? "border-rose-300 bg-rose-50"
                    : "border-slate-200 bg-slate-50";
                return <div key={key} className={`rounded-xl border p-3 text-sm ${cardTone}`}>
                  <div className="font-semibold">{t(key === "mean_error" ? "modelCompare.metric.bias_magnitude" : `modelCompare.metric.${key}`)}</div>
                  {models.map((model) => <div key={model.key}>{t(`modelCompare.${model.labelKey}`)}: {formatMetricValue(model.value, language)}</div>)}
                  {differences.map((difference) => <div key={`${difference.from}-${difference.to}`} className="mt-1">
                    <div>Δ {t(`modelCompare.${difference.labelKey}`)}: {formatSignedMetricValue(difference.change, language)} {difference.relativeChange === null ? "" : `(${formatRelativePercentageChange(difference.relativeChange, language)})`}</div>
                    {difference.status && <div className={difference.status === "worsened" ? "text-rose-700" : difference.status === "improved" ? "text-emerald-700" : "text-slate-600"}>{t(`modelCompare.${difference.status}`)}</div>}
                  </div>)}
                </div>;
              })}
            </div>
          </div>
          <p className="mt-3 text-xs text-slate-600">{t("modelCompare.benchmarkRelationships")}</p>
          <details className="mt-2 rounded-xl border border-violet-200 bg-white p-3 text-sm">
            <summary className="cursor-pointer font-medium">{t("modelCompare.benchmarkGaps")}</summary>
            <div className="mt-2 overflow-x-auto"><table className="w-full min-w-[520px] text-left">
              <thead><tr><th className="p-2">{t("modelCompare.model")}</th><th className="p-2">{t("modelCompare.metric.mae")}</th><th className="p-2">{t("modelCompare.metric.rmse")}</th><th className="p-2">{t("modelCompare.metric.r2")}</th></tr></thead>
              <tbody>{[["reducedData", performance.reduced], ["editedReducedData", performance.edited]].map(([label, metrics]) => <tr key={label} className="border-t border-slate-100"><th className="p-2 font-medium">{t(`modelCompare.${label}`)} − {t("modelCompare.fullData")}</th>{["mae", "rmse", "r2"].map((key) => <td key={key} className="p-2 tabular-nums">{Number.isFinite(metrics?.[key]) && Number.isFinite(performance.baseline?.[key]) ? formatSignedMetricValue(metrics[key] - performance.baseline[key], language) : "—"}</td>)}</tr>)}</tbody>
            </table></div>
          </details>
          <p className="mt-2 text-xs text-slate-600">{t("modelCompare.correctionMetadata", { enabled: performance.intercept_correction?.enabled ? t("common.yes") : t("common.no"), shift: formatSignedMetricValue(performance.intercept_correction?.shift, language), source: performance.intercept_correction?.calculated_on || "manual" })}</p>
          {performance.edited_before_intercept_correction && <details className="mt-2 rounded-xl border border-violet-200 bg-white p-3 text-sm">
            <summary className="cursor-pointer font-medium">{t("modelCompare.beforeAfterCorrection")}</summary>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {[["beforeCorrection", performance.edited_before_intercept_correction], ["afterCorrection", performance.edited]].map(([label, metrics]) => <div key={label}>
                <strong>{t(`modelCompare.${label}`)}</strong>: {t("modelCompare.metric.mae")} {formatMetricValue(metrics.mae, language)} · {t("modelCompare.metric.mean_error")} {formatSignedMetricValue(metrics.mean_error, language)}
              </div>)}
            </div>
          </details>}
          {!isMainTrainingData && (performance.validation?.common_holdout_verified
            ? <p className="mt-2 text-xs text-emerald-800">{t("modelCompare.splitVerified", { samples: performance.test_size })}</p>
            : <p className="mt-2 text-xs text-amber-800">{t("modelCompare.splitValidation", { full: performance.validation?.full_training_rows_available ? performance.validation.full_training_feature_matches : t("common.unknown"), reduced: performance.validation?.reduced_training_rows_available ? performance.validation.reduced_training_feature_matches : t("common.unknown"), shared: performance.validation?.same_exported_test_rows == null ? t("common.unknown") : performance.validation.same_exported_test_rows ? t("common.yes") : t("common.no") })}</p>)}
          {!isMainTrainingData && Object.values(performance.validation?.preprocessing_scope || {}).includes("whole_dataset_before_split") && <p className="mt-2 text-xs text-amber-800">{t("modelCompare.preprocessingLeakage")}</p>}
          <details className="mt-3 rounded-xl border border-violet-200 bg-white p-3 text-sm">
            <summary className="cursor-pointer font-medium">{t("modelCompare.observationEffects")}</summary>
            <p className="mt-2">{t("modelCompare.improved")}: {formatMetricValue(performance.edit_effect?.improved_percent, language)}% · {t("modelCompare.worsened")}: {formatMetricValue(performance.edit_effect?.worsened_percent, language)}% · {t("modelCompare.medianErrorChange")}: {formatSignedMetricValue(performance.edit_effect?.median_absolute_error_change, language)}</p>
          </details>
          <HeldOutActualVsPredictedChart performance={performance} visibleModels={comparison.modelKeys} tallCharts={tallCharts} t={t} />
        </>
      ) : (
        <p className="mt-4 text-sm text-slate-600">
          {t("modelCompare.heldOutTestReady")}
        </p>
      )}
    </section>
  );
};

const TrainingMeanDetails = ({
  summary,
  preview,
  originalArtifact,
  reducedArtifact,
  language,
  t,
}) => {
  const originalIntercept = originalArtifact?.base_prediction_offset;
  const reducedIntercept = reducedArtifact?.base_prediction_offset;
  const appliedAdjustment = preview?.intercept_adjustment;
  const editedIntercept = preview?.adjusted_edited_base_prediction_offset;
  const meanRows = summary ? [
    ["reducedTrainingMean", summary.reduced_model_mean],
    ["editedTrainingMeanBeforeCorrection", summary.edited_before_correction_mean],
    ["trainingMeanDifferenceBeforeCorrection", summary.difference_before_correction, true],
    ["editedTrainingMeanAfterCorrection", summary.edited_after_correction_mean],
  ] : [];
  const interceptRows = [
    ["originalFullDataIntercept", originalIntercept],
    ["reducedModelIntercept", reducedIntercept],
    ["editedInterceptBeforeCorrection", reducedIntercept],
    ["appliedInterceptCorrection", appliedAdjustment, true],
    ["editedInterceptAfterCorrection", editedIntercept],
    ...(summary ? [
      ["suggestedMatchingCorrection", summary.suggested_matching_adjustment, true],
      ["suggestedMatchingIntercept", summary.suggested_matching_intercept],
    ] : []),
  ];
  const renderRows = (rows) => rows.map(([key, value, signed]) => (
    <div
      key={key}
      className="grid grid-cols-1 gap-1 border-b border-slate-100 py-2 last:border-b-0 sm:grid-cols-[minmax(0,1fr)_auto] sm:gap-6"
    >
      <dt className="text-slate-700">{t(`modelCompare.${key}`)}</dt>
      <dd className="font-semibold tabular-nums text-slate-950">
        {signed
          ? formatSignedMetricValue(value, language)
          : formatMetricValue(value, language)}
      </dd>
    </div>
  ));

  return (
    <details className="mt-4 w-full rounded-xl border border-indigo-200 bg-white px-4 py-3 lg:w-1/2">
      <summary className="cursor-pointer text-sm font-semibold text-indigo-950">
        {t("modelCompare.trainingMeanDetailsTitle")}
      </summary>
      <div className="mt-3 text-sm">
        {summary ? (
          <>
            <p className="mb-2 text-slate-600">
              {t("modelCompare.trainingMeanDetailsDescription", {
                count: summary.training_row_count,
              })}
            </p>
            <dl>{renderRows(meanRows)}</dl>
          </>
        ) : (
          <p className="mb-2 text-slate-600">
            {t("modelCompare.trainingMeanUnavailable")}
          </p>
        )}
        <dl className="mt-3 border-t border-indigo-200 pt-2">
          {renderRows(interceptRows)}
        </dl>
      </div>
    </details>
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
              {t("combined.mae")}:{" "}
              {formatMetricValue(trace.original_base_mae, language)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const buildIndividualMaeChangeBars = (
  featurePreview,
  formatExpertLabel = formatChartExpertLabel,
  isVisible = () => true,
  confidenceRated = false,
) => {
  const baselineMae = Number(
    featurePreview?.comparison_metrics?.baseline_edited_vs_original_base_mae,
  );
  if (!Number.isFinite(baselineMae)) return [];

  return (featurePreview?.right_individual_edit_traces || [])
    .filter(isVisible)
    .map((trace) => {
      const originalBaseMae = Number(
        confidenceRated
          ? trace?.original_base_mae_with_confidence
          : trace?.original_base_mae_without_confidence ?? trace?.original_base_mae,
      );
      if (!Number.isFinite(originalBaseMae)) return null;

      const userName = trace?.user_name || trace?.submission_id || "";
      return {
        expertLabel: formatExpertLabel(userName),
        maeChange: Number((originalBaseMae - baselineMae).toPrecision(12)),
        userName,
      };
    })
    .filter(Boolean);
};

const formatMaeDeltaLabel = (value) => {
  const delta = Number(value);
  if (!Number.isFinite(delta)) return "";

  const roundedDelta = Math.round(delta * 1000) / 1000;
  const sign = roundedDelta < 0 ? "-" : "+";
  const magnitude = Math.abs(roundedDelta)
    .toFixed(3)
    .replace(/\.?0+$/, "");
  return `${sign}${magnitude}`;
};

const getDetailedMaeDeltaXAxis = (bars) => {
  const largestAbsoluteDelta = Math.max(
    ...bars
      .map((bar) => Math.abs(Number(bar?.maeChange)))
      .filter(Number.isFinite),
    0.01,
  );
  const rawTickSize = largestAbsoluteDelta / 6;
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
  const halfRange = (Math.ceil(largestAbsoluteDelta / dtick) + 2) * dtick;

  return {
    range: [-halfRange, halfRange],
    dtick,
    tick0: 0,
    tickformat: "+.3~f",
  };
};

const ExpertMaeChangeBarChart = ({ bars, confidenceBars, title, tallCharts, t, className }) => {
  const [hiddenModes, setHiddenModes] = useState([]);
  const makeTrace = (traceBars, confidenceRated) => ({
    type: "bar",
    orientation: "h",
    visible: !hiddenModes.includes(confidenceRated ? "rated" : "plain"),
    name: t(
      confidenceRated
        ? "modelCompare.allFeatureMaeChangeWithConfidence"
        : "modelCompare.allFeatureMaeChangeWithoutConfidence",
    ),
    x: traceBars.map((bar) => bar.maeChange),
    y: traceBars.map((bar) => bar.expertLabel),
    marker: {
      color: traceBars.map((bar) => getIndividualEditColor(bar.userName)),
      ...(confidenceRated
        ? {
            pattern: {
              shape: "/",
              fgcolor: "#fff7ed",
              bgcolor: traceBars.map((bar) => getIndividualEditColor(bar.userName)),
              size: 7,
              solidity: 0.25,
            },
          }
        : {}),
    },
    text: traceBars.map((bar) => formatMaeDeltaLabel(bar.maeChange)),
    texttemplate: "%{text}",
    textposition: "outside",
    cliponaxis: false,
    hovertemplate:
      "%{y}<br>" +
      t("modelCompare.expertMaeChange") +
      ": %{x:+.3~f}<extra>" +
      t(
        confidenceRated
          ? "modelCompare.allFeatureMaeChangeWithConfidence"
          : "modelCompare.allFeatureMaeChangeWithoutConfidence",
      ) +
      "</extra>",
  });
  const data = [makeTrace(bars, false)];
  if (confidenceBars.length) data.push(makeTrace(confidenceBars, true));
  const chartHeight = Math.max(220, bars.length * 48 + 70);

  return (
    <div className={className}>
      <h3 className="mb-2 text-center text-base font-semibold text-slate-900">
        {title}
      </h3>
      {confidenceBars.length > 0 && (
        <div className="mb-2 flex flex-wrap justify-center gap-x-4 gap-y-1">
          {[
            { mode: "plain", label: data[0].name, striped: false },
            { mode: "rated", label: data[1].name, striped: true },
          ].map(({ mode, label, striped }) => (
            <button
              key={mode}
              type="button"
              aria-pressed={!hiddenModes.includes(mode)}
              onClick={() =>
                setHiddenModes((current) =>
                  current.includes(mode)
                    ? current.filter((item) => item !== mode)
                    : [...current, mode],
                )
              }
              className={`inline-flex items-center gap-2 rounded px-1 py-0.5 text-xs text-slate-700 hover:bg-slate-100 ${
                hiddenModes.includes(mode) ? "opacity-50" : ""
              }`}
            >
              <span
                aria-hidden="true"
                className="h-3 w-4 rounded-sm border border-slate-500 bg-slate-400"
                style={
                  striped
                    ? {
                        backgroundImage:
                          "repeating-linear-gradient(135deg, #94a3b8 0 4px, #f8fafc 4px 7px)",
                      }
                    : undefined
                }
              />
              {label}
            </button>
          ))}
        </div>
      )}
      <Plot
        data={data}
        layout={{
          autosize: true,
          barmode: "group",
          xaxis: {
            title: t("modelCompare.expertMaeChangeAxis"),
            gridcolor: "#e2e8f0",
            zeroline: true,
            zerolinecolor: "#475569",
            zerolinewidth: 2,
            ...getDetailedMaeDeltaXAxis([...bars, ...confidenceBars]),
          },
          yaxis: {
            automargin: true,
            autorange: "reversed",
            tickfont: { size: 11 },
          },
          height: getChartHeight(chartHeight, tallCharts),
          margin: { l: 105, r: 64, t: 24, b: 60 },
          paper_bgcolor: "white",
          plot_bgcolor: "white",
          showlegend: false,
        }}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        style={{ width: "100%" }}
        className="w-full"
      />
    </div>
  );
};

const IndividualParticipantMaeChangeChart = ({
  featurePreview,
  hiddenExpertKeys,
  showConfidenceRatedMae,
  tallCharts,
  t,
}) => {
  const isVisible = (trace) => isChartExpertVisible(trace, hiddenExpertKeys);
  const bars = buildIndividualMaeChangeBars(
    featurePreview,
    formatChartExpertLabel,
    isVisible,
  );
  const confidenceBars = showConfidenceRatedMae
    ? buildIndividualMaeChangeBars(
        featurePreview,
        formatChartExpertLabel,
        isVisible,
        true,
      )
    : [];
  if (!bars.length) return null;

  return (
    <ExpertMaeChangeBarChart
      bars={bars}
      confidenceBars={confidenceBars}
      title={t("modelCompare.expertMaeChangeTitle")}
      tallCharts={tallCharts}
      t={t}
      className="w-full rounded-2xl border border-indigo-200 bg-white p-4 shadow-sm xl:w-2/5"
    />
  );
};

const buildAllFeatureMaeChangeBars = (features) =>
  (features || [])
    .map((feature) => {
      const mae = Number(feature.mae);
      const baselineMae = Number(feature.baseline_mae);
      if (!Number.isFinite(mae) || !Number.isFinite(baselineMae)) return null;
      return {
        featureName: feature.feature_name,
        maeChange: Number((mae - baselineMae).toPrecision(12)),
      };
    })
    .filter(Boolean);

const formatMaeFeatureLabel = (featureName) =>
  featureName.replace(/^.*?\s+[-–—]\s+/, "");

const FeatureMaeChangeChart = ({ series, title, combined, tallCharts, t }) => {
  const data = series.map((item) => ({
    type: "bar",
    orientation: "h",
    name: item.label,
    x: item.bars.map((bar) => bar.maeChange),
    y: item.bars.map((bar) => formatMaeFeatureLabel(bar.featureName)),
    customdata: item.bars.map((bar) => bar.featureName),
    marker: {
      color: item.color,
      ...(combined && item.confidenceRated
        ? {
            pattern: {
              shape: "/",
              fgcolor: "#fff7ed",
              bgcolor: RIGHT_COLORS.effective,
              size: 7,
              solidity: 0.25,
            },
          }
        : {}),
    },
    text: item.bars.map((bar) => formatMaeDeltaLabel(bar.maeChange)),
    texttemplate: "%{text}",
    textfont: { size: 16 },
    textposition: "outside",
    cliponaxis: false,
    hovertemplate:
      "%{customdata}<br>" + item.label + ": %{x:+.3~f}<extra></extra>",
  }));
  const featureCount = Math.max(...series.map((item) => item.bars.length), 0);
  const chartHeight = Math.min(
    680,
    Math.max(300, featureCount * (combined ? 48 : 38) + 110),
  );

  return (
    <div className="min-w-0 rounded-2xl border border-amber-200 bg-white p-3 shadow-sm">
      <Plot
        data={data}
        layout={{
          autosize: true,
          title: { text: title, font: { size: 20, color: "#0f172a" } },
          barmode: "group",
          bargap: 0.06,
          xaxis: {
            title: t("modelCompare.expertMaeChangeAxis"),
            gridcolor: "#e2e8f0",
            zeroline: true,
            zerolinecolor: "#475569",
            zerolinewidth: 2,
            range: [-10, 25],
            tickvals: [-10, 0, 10, 20, 25],
            tickformat: "+.3~f",
            tickfont: { size: 15 },
            fixedrange: true,
          },
          yaxis: {
            automargin: true,
            autorange: "reversed",
            tickfont: { size: 15 },
          },
          height: getChartHeight(chartHeight, tallCharts),
          margin: { l: 90, r: 52, t: combined ? 96 : 48, b: 54 },
          paper_bgcolor: "white",
          plot_bgcolor: "white",
          showlegend: combined,
          legend: {
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: 1.02,
            yanchor: "bottom",
            font: { size: 16 },
          },
        }}
        config={{ responsive: true, displayModeBar: false }}
        useResizeHandler
        style={{ width: "100%" }}
        className="w-full"
      />
    </div>
  );
};

const AllFeatureMaeChangeCharts = ({ metrics, tallCharts, t }) => {
  const [combinedView, setCombinedView] = useState(false);
  const withoutConfidenceBars = buildAllFeatureMaeChangeBars(
    metrics?.without_confidence,
  );
  const withConfidenceBars = buildAllFeatureMaeChangeBars(
    metrics?.with_confidence,
  );
  const withoutConfidenceSeries = {
    bars: withoutConfidenceBars,
    label: t("modelCompare.allFeatureMaeChangeWithoutConfidence"),
    color: RIGHT_COLORS.effective,
    confidenceRated: false,
  };
  const withConfidenceSeries = {
    bars: withConfidenceBars,
    label: t("modelCompare.allFeatureMaeChangeWithConfidence"),
    color: RIGHT_COLORS.effective,
    confidenceRated: true,
  };

  return (
    <section className="rounded-2xl border border-amber-200 bg-amber-50/60 p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">
            {t("modelCompare.allFeatureMaeChange")}
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            {t("modelCompare.allFeatureMaeChangeDescription")}
          </p>
        </div>
        <button
          type="button"
          aria-pressed={combinedView}
          onClick={() => setCombinedView((current) => !current)}
          className="rounded-xl border border-amber-300 bg-white px-4 py-2 text-sm font-medium text-amber-900 hover:bg-amber-50"
        >
          {combinedView
            ? t("modelCompare.showSeparateMaeCharts")
            : t("modelCompare.showCombinedMaeChart")}
        </button>
      </div>
      {combinedView ? (
        <div className="mt-4 w-full lg:mx-auto lg:w-[80%]">
          <FeatureMaeChangeChart
            series={[withoutConfidenceSeries, withConfidenceSeries]}
            title={t("modelCompare.allFeatureMaeChange")}
            combined
            tallCharts={tallCharts}
            t={t}
          />
        </div>
      ) : (
        <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
          <FeatureMaeChangeChart
            series={[withoutConfidenceSeries]}
            title={withoutConfidenceSeries.label}
            combined={false}
            tallCharts={tallCharts}
            t={t}
          />
          <FeatureMaeChangeChart
            series={[withConfidenceSeries]}
            title={withConfidenceSeries.label}
            combined={false}
            tallCharts={tallCharts}
            t={t}
          />
        </div>
      )}
    </section>
  );
};

const ExpertAverageConfidenceChart = ({ artifact, language, tallCharts, t }) => {
  const bars = buildExpertConfidenceBars(artifact);
  const confidenceFormatter = new Intl.NumberFormat(getNumberLocale(language), {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

  return (
    <section className="rounded-2xl border border-amber-200 bg-amber-50/60 p-5 shadow-sm">
      <h2 className="text-xl font-semibold text-slate-900">
        {t("modelCompare.expertAverageConfidence")}
      </h2>
      <p className="mt-1 text-sm text-slate-600">
        {t("modelCompare.expertAverageConfidenceDescription")}
      </p>
      {bars.length ? (
        <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
          <div className="min-w-0 rounded-2xl border border-amber-200 bg-white p-3 shadow-sm">
            <Plot
              data={[{
                type: "bar",
                orientation: "h",
                x: bars.map((bar) => bar.averageConfidence),
                y: bars.map((bar) => formatChartExpertLabel(bar.userName)),
                marker: {
                  color: bars.map((bar) => getIndividualEditColor(bar.userName)),
                },
                text: bars.map((bar) =>
                  confidenceFormatter.format(bar.averageConfidence),
                ),
                texttemplate: "%{text}",
                textposition: "outside",
                cliponaxis: false,
                customdata: bars.map((bar) => bar.ratingCount),
                hovertemplate:
                  "%{y}<br>" +
                  `${t("modelCompare.expertAverageConfidenceAxis")}: %{x:.2f}/10<br>` +
                  `${t("modelCompare.expertAverageConfidenceRatings")}: %{customdata}<extra></extra>`,
              }]}
              layout={{
                autosize: true,
                xaxis: {
                  title: t("modelCompare.expertAverageConfidenceAxis"),
                  range: [0, 11],
                  tickvals: [0, 2, 4, 6, 8, 10],
                  gridcolor: "#e2e8f0",
                  zeroline: true,
                },
                yaxis: {
                  automargin: true,
                  autorange: "reversed",
                  tickfont: { size: 12 },
                },
                height: getChartHeight(Math.max(280, bars.length * 48 + 100), tallCharts),
                margin: { l: 115, r: 65, t: 24, b: 65 },
                paper_bgcolor: "white",
                plot_bgcolor: "white",
                showlegend: false,
              }}
              config={{ responsive: true, displayModeBar: false }}
              useResizeHandler
              style={{ width: "100%" }}
              className="w-full"
            />
          </div>
        </div>
      ) : (
        <p className="mt-4 text-sm text-slate-500">
          {t("modelCompare.noExpertConfidenceRatings")}
        </p>
      )}
    </section>
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
  const overlapCheckId = useRef(0);
  const previewRequestId = useRef(0);
  const [leftFile, setLeftFile] = useState(null);
  const [rightFile, setRightFile] = useState(null);
  const [overlapCsvFile, setOverlapCsvFile] = useState(null);
  const [overlapSourceCsvFile, setOverlapSourceCsvFile] = useState(null);
  const [overlapResult, setOverlapResult] = useState(null);
  const [overlapError, setOverlapError] = useState(null);
  const [overlapLoading, setOverlapLoading] = useState(false);
  const [comparePayload, setComparePayload] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectionState, setSelectionState] = useState({ left: {}, right: {} });
  const [overallMaeFeatureSelection, setOverallMaeFeatureSelection] = useState(
    {},
  );
  const [contributionFeatureSelection, setContributionFeatureSelection] =
    useState({});
  const [meanShiftFeatureFactorSelection, setMeanShiftFeatureFactorSelection] =
    useState({});
  const [meanShiftFeatureFactorDraft, setMeanShiftFeatureFactorDraft] =
    useState({});
  const [meanShiftFeatureModalOpen, setMeanShiftFeatureModalOpen] =
    useState(false);
  const [interceptAdjustment, setInterceptAdjustment] = useState(0);
  const [interceptAdjustmentInput, setInterceptAdjustmentInput] = useState("0");
  const [useConfidence, setUseConfidence] = useState(true);
  const [featureMetricsOnOriginalTestSet, setFeatureMetricsOnOriginalTestSet] =
    useState(false);
  const [averageScoreMethod, setAverageScoreMethod] = useState("main_dataset");
  const [showBaseTraces, setShowBaseTraces] = useState(true);
  const [centerCharts, setCenterCharts] = useState(false);
  const [showIndividualEditTraces, setShowIndividualEditTraces] =
    useState(false);
  const [chartExpertSelectionEnabled, setChartExpertSelectionEnabled] =
    useState(false);
  const [hiddenChartExpertsByFeature, setHiddenChartExpertsByFeature] = useState(
    {},
  );
  const [showIndividualParticipantMae, setShowIndividualParticipantMae] =
    useState(false);
  const [
    showIndividualParticipantMaeChangeChart,
    setShowIndividualParticipantMaeChangeChart,
  ] = useState(false);
  const [showConfidenceRatedMae, setShowConfidenceRatedMae] = useState(false);
  const [showHeldOutTestPerformance, setShowHeldOutTestPerformance] =
    useState(false);
  const [heldOutTestSource, setHeldOutTestSource] = useState("left");
  const sharedHeldOutLineage = overlapResult?.training_overlap?.shared_evaluation;
  const [showConfidenceInLegend, setShowConfidenceInLegend] = useState(false);
  const [legendOnRight, setLegendOnRight] = useState(false);
  const [showLineMarkers, setShowLineMarkers] = useState(true);
  const [tallCharts, setTallCharts] = useState(false);
  const [showAlternateConfidenceAggregation, setShowAlternateConfidenceAggregation] =
    useState(false);
  const [featureSearch, setFeatureSearch] = useState("");
  const [jumpFeature, setJumpFeature] = useState("");
  const [loading, setLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectionModal, setSelectionModal] = useState(null);

  const resetPreparedState = () => {
    overlapCheckId.current += 1;
    previewRequestId.current += 1;
    setPreviewLoading(false);
    setComparePayload(null);
    setOverlapResult(null);
    setOverlapError(null);
    setOverlapLoading(false);
    setPreview(null);
    setSelectionState({ left: {}, right: {} });
    setOverallMaeFeatureSelection({});
    setContributionFeatureSelection({});
    setMeanShiftFeatureFactorSelection({});
    setMeanShiftFeatureFactorDraft({});
    setMeanShiftFeatureModalOpen(false);
    setInterceptAdjustment(0);
    setInterceptAdjustmentInput("0");
    setUseConfidence(true);
    setFeatureMetricsOnOriginalTestSet(false);
    setAverageScoreMethod("main_dataset");
    setShowBaseTraces(true);
    setCenterCharts(false);
    setShowIndividualEditTraces(false);
    setChartExpertSelectionEnabled(false);
    setHiddenChartExpertsByFeature({});
    setShowIndividualParticipantMae(false);
    setShowIndividualParticipantMaeChangeChart(false);
    setShowConfidenceRatedMae(false);
    setShowHeldOutTestPerformance(false);
    setHeldOutTestSource("left");
    setShowConfidenceInLegend(false);
    setLegendOnRight(false);
    setShowLineMarkers(true);
    setTallCharts(false);
    setShowAlternateConfidenceAggregation(false);
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
    nextHeldOutTestSource = showHeldOutTestPerformance
      ? heldOutTestSource
      : null,
    nextInterceptAdjustment = interceptAdjustment,
    nextFeatureMetricsOnOriginalTestSet = featureMetricsOnOriginalTestSet,
  ) => {
    if (!comparePayload) return null;
    const requestId = ++previewRequestId.current;
    setPreviewLoading(true);
    setError(null);
    try {
      const sharedArtifacts = nextHeldOutTestSource === "left" && sharedHeldOutLineage
        ? buildSharedHeldOutArtifacts(
            comparePayload.left_artifact,
            comparePayload.right_artifact,
            sharedHeldOutLineage,
          )
        : null;
      if (nextHeldOutTestSource === "left" &&
          needsIqbSharedHeldOut(comparePayload.left_artifact) && !sharedArtifacts) {
        throw new Error(t("modelCompare.sharedHeldOutNeedsLineage"));
      }
      const result = await apiService.previewModelComparison({
        left_artifact: sharedArtifacts?.left || comparePayload.left_artifact,
        right_artifact: sharedArtifacts?.right || comparePayload.right_artifact,
        left_selected_submission_ids: getSelectedSubmissionIds(
          nextSelectionState.left,
        ),
        right_selected_submission_ids: getSelectedSubmissionIds(
          nextSelectionState.right,
        ),
        use_confidence: nextUseConfidence,
        limit_edited_mean_deviation: nextLimitEditedMeanDeviation,
        feature_names: comparePayload.shared_features,
        overall_mae_feature_names: (
          comparePayload.shared_features || []
        ).filter(
          (featureName) =>
            nextOverallMaeFeatureSelection[featureName] !== false,
        ),
        mean_deviation_feature_names: (
          comparePayload.shared_features || []
        ).filter(
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
        intercept_adjustment: nextInterceptAdjustment,
        apply_training_intercept_correction: false,
        feature_metrics_on_original_test_set:
          nextFeatureMetricsOnOriginalTestSet,
        ...(nextHeldOutTestSource
          ? { held_out_test_source: nextHeldOutTestSource }
          : {}),
      });
      if (requestId !== previewRequestId.current) return null;
      setPreview(result);
      return result;
    } catch (err) {
      if (requestId === previewRequestId.current) {
        setShowHeldOutTestPerformance(false);
        setError(
          err?.response?.data?.detail ||
            err?.message ||
            t("modelCompare.previewError"),
        );
      }
      return null;
    } finally {
      if (requestId === previewRequestId.current) setPreviewLoading(false);
    }
  };

  const handlePrepare = async () => {
    if (!leftFile || !rightFile) {
      setError(t("modelCompare.chooseBothFiles"));
      return;
    }
    overlapCheckId.current += 1;
    previewRequestId.current += 1;
    setPreviewLoading(false);
    setOverlapResult(null);
    setOverlapError(null);
    setOverlapLoading(false);
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
      setHiddenChartExpertsByFeature({});
      setOverallMaeFeatureSelection(
        Object.fromEntries(
          (result.shared_features || []).map((featureName) => [
            featureName,
            true,
          ]),
        ),
      );
      setMeanShiftFeatureFactorSelection({});
      setMeanShiftFeatureFactorDraft({});
      setInterceptAdjustment(0);
      setInterceptAdjustmentInput("0");
      setPreview(result.preview);
      setUseConfidence(Boolean(result.preview?.use_confidence));
      setFeatureMetricsOnOriginalTestSet(false);
      setHeldOutTestSource(
        getInitialHeldOutTestSource(
          result.left_artifact,
          result.preview?.main_training_performance_available,
        ),
      );
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

  const handleCheckHeldOutOverlap = async () => {
    if (!leftFile || !rightFile || !overlapCsvFile) return;
    const checkId = ++overlapCheckId.current;
    previewRequestId.current += 1;
    setPreviewLoading(false);
    setShowHeldOutTestPerformance(false);
    setOverlapLoading(true);
    setOverlapResult(null);
    setOverlapError(null);
    try {
      const result = await apiService.checkModelCompareHeldOutOverlap(
        leftFile, rightFile, overlapCsvFile, overlapSourceCsvFile,
      );
      if (checkId === overlapCheckId.current) {
        setOverlapResult(result);
        setShowHeldOutTestPerformance(false);
      }
    } catch (err) {
      if (checkId === overlapCheckId.current) {
        setOverlapError(
          err?.response?.data?.detail || err?.message || t("modelCompare.overlapError"),
        );
      }
    } finally {
      if (checkId === overlapCheckId.current) setOverlapLoading(false);
    }
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

  const handleFeatureMetricsTestSetToggle = async (checked) => {
    setFeatureMetricsOnOriginalTestSet(checked);
    await requestPreview(
      selectionState,
      useConfidence,
      false,
      overallMaeFeatureSelection,
      contributionFeatureSelection,
      meanShiftFeatureFactorSelection,
      showHeldOutTestPerformance ? heldOutTestSource : null,
      interceptAdjustment,
      checked,
    );
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

  const handleEvaluateHeldOutTest = async () => {
    const rawValue = interceptAdjustmentInput.trim();
    const nextAdjustment = Number(rawValue);
    if (!rawValue || !Number.isFinite(nextAdjustment)) {
      setError(t("modelCompare.interceptAdjustmentInvalid"));
      return;
    }
    setInterceptAdjustment(nextAdjustment);
    setInterceptAdjustmentInput(String(nextAdjustment));
    setShowHeldOutTestPerformance(false);
    const result = await requestPreview(
      selectionState,
      useConfidence,
      false,
      overallMaeFeatureSelection,
      contributionFeatureSelection,
      meanShiftFeatureFactorSelection,
      heldOutTestSource,
      nextAdjustment,
    );
    if (result) setShowHeldOutTestPerformance(true);
  };

  const handleHeldOutTestSourceChange = (source) => {
    setHeldOutTestSource(source);
    setShowHeldOutTestPerformance(false);
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
    if (Object.values(nextMeanShiftFeatureFactorSelection).some(Boolean)) {
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

  const handleInterceptAdjustmentSubmit = async (event) => {
    event.preventDefault();
    const rawValue = interceptAdjustmentInput.trim();
    const nextAdjustment = Number(rawValue);
    if (!rawValue || !Number.isFinite(nextAdjustment)) {
      setError(t("modelCompare.interceptAdjustmentInvalid"));
      return;
    }
    setInterceptAdjustment(nextAdjustment);
    setInterceptAdjustmentInput(String(nextAdjustment));
    if (nextAdjustment === interceptAdjustment) return;
    await requestPreview(
      selectionState,
      useConfidence,
      false,
      overallMaeFeatureSelection,
      contributionFeatureSelection,
      meanShiftFeatureFactorSelection,
      showHeldOutTestPerformance ? heldOutTestSource : null,
      nextAdjustment,
    );
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

  const averagePredictionScores = getAveragePredictionScores(
    preview,
    averageScoreMethod,
  );
  const selectedContributionRows = [
    ...(preview?.average_prediction_feature_contributions?.[
      averageScoreMethod
    ] || []),
  ].sort((left, right) => {
    if (left.is_base_offset !== right.is_base_offset) {
      return left.is_base_offset ? -1 : 1;
    }
    return Math.abs(right.difference) - Math.abs(left.difference);
  });
  const contributionFeatureRows = selectedContributionRows.filter(
    (row) => !row.is_base_offset && !row.is_internal_feature,
  );
  const meanShiftModalFeatureRows = contributionFeatureRows.filter(
    (row) => contributionFeatureSelection[row.feature_name] !== false,
  );
  const selectedContributionScoreRows = selectedContributionRows.filter(
    (row) =>
      row.is_base_offset ||
      row.is_internal_feature ||
      contributionFeatureSelection[row.feature_name] !== false,
  );
  const displayedContributionRows = selectedContributionRows.filter(
    (row) => !row.is_base_offset,
  );
  const displayedReducedEditedContributionRows = [
    ...displayedContributionRows,
  ].sort(
    (left, right) =>
      Math.abs(right.edited - right.reduced) -
      Math.abs(left.edited - left.reduced),
  );
  const selectedContributionAverageScores = selectedContributionScoreRows.length
    ? selectedContributionScoreRows.reduce(
        (scores, row) => ({
          original: scores.original + row.original,
          reduced: scores.reduced + row.reduced,
          edited: scores.edited + row.edited,
          difference: scores.difference + row.difference,
          edited_minus_reduced:
            scores.edited_minus_reduced + row.edited - row.reduced,
        }),
        { original: 0, reduced: 0, edited: 0, difference: 0, edited_minus_reduced: 0 },
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
  const contributionSelectionIsActive = contributionFeatureRows.some(
    (row) => contributionFeatureSelection[row.feature_name] === false,
  );

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

            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold text-slate-900">
                {t("modelCompare.overlapTitle")}
              </h2>
              <p className="mt-2 text-sm text-slate-600">
                {t("modelCompare.overlapDescription")}
              </p>
              {comparePayload.left_artifact?.held_out_test_data?.feature_rows?.length ? (
                <>
                  {!comparePayload.right_artifact?.held_out_test_data?.feature_rows?.length && (
                    <p className="mt-3 text-sm text-amber-800">
                      {t("modelCompare.overlapEditedUnavailable")}
                    </p>
                  )}
                  <div className="mt-4 flex flex-wrap items-end gap-4">
                    <label className="text-sm font-medium text-slate-700">
                      {t("modelCompare.overlapCsvLabel")}
                      <input
                        type="file"
                        accept=".csv,text/csv"
                        className="mt-2 block w-full text-sm text-slate-600"
                        onChange={(event) => {
                          overlapCheckId.current += 1;
                          setOverlapLoading(false);
                          setOverlapCsvFile(event.target.files?.[0] || null);
                          setOverlapResult(null);
                          setOverlapError(null);
                          setShowHeldOutTestPerformance(false);
                        }}
                      />
                    </label>
                    <label className="text-sm font-medium text-slate-700">
                      {t("modelCompare.overlapSourceCsvLabel")}
                      <input
                        type="file"
                        accept=".csv,text/csv"
                        className="mt-2 block w-full text-sm text-slate-600"
                        onChange={(event) => {
                          overlapCheckId.current += 1;
                          setOverlapLoading(false);
                          setOverlapSourceCsvFile(event.target.files?.[0] || null);
                          setOverlapResult(null);
                          setOverlapError(null);
                          setShowHeldOutTestPerformance(false);
                        }}
                      />
                    </label>
                    <button
                      type="button"
                      onClick={handleCheckHeldOutOverlap}
                      disabled={!overlapCsvFile || overlapLoading}
                      className="rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {overlapLoading ? t("modelCompare.overlapChecking") : t("modelCompare.overlapCheck")}
                    </button>
                  </div>
                  {overlapError && <p role="alert" className="mt-4 text-sm text-rose-700">{overlapError}</p>}
                  {overlapResult && (
                    <div className="mt-4 rounded-xl bg-slate-50 p-4 text-sm text-slate-700" role="status">
                      <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-4">
                        <h3 className="font-semibold text-indigo-950">
                          {t("modelCompare.trainingOverlapTitle")}
                        </h3>
                        {overlapResult.training_overlap?.status === "exact" ? (
                          <>
                            <p className="mt-2 text-base font-semibold text-indigo-950">
                              {t("modelCompare.trainingOverlapExact", {
                                matches: overlapResult.training_overlap.overlap_count,
                                training: overlapResult.training_overlap.training_row_count,
                                test: overlapResult.training_overlap.original_test_row_count,
                              })}
                            </p>
                            <p className="mt-1 text-xs text-indigo-900">
                              {t(overlapResult.training_overlap.method === "source_rows"
                                ? "modelCompare.trainingOverlapSourceVerified"
                                : "modelCompare.trainingOverlapIdsVerified")}
                            </p>
                          </>
                        ) : overlapResult.training_overlap?.status === "target_only" ? (
                          <>
                            <p className="mt-2 text-base font-semibold text-indigo-950">
                              {t("modelCompare.trainingOverlapTargetOnly", {
                                matches: overlapResult.training_overlap.overlap_count,
                                training: overlapResult.training_overlap.training_row_count,
                                test: overlapResult.training_overlap.original_test_row_count,
                              })}
                            </p>
                            <p className="mt-1 text-xs text-amber-900">
                              {t("modelCompare.trainingOverlapNeedsSource")}
                            </p>
                          </>
                        ) : (
                          <p className="mt-2 text-sm text-amber-900">
                            {t("modelCompare.trainingOverlapUnavailable", {
                              reason: overlapResult.training_overlap?.reason || t("common.unknown"),
                            })}
                          </p>
                        )}
                        {Number.isFinite(overlapResult.training_overlap?.usable_reduced_csv_rows) && (
                          <p className="mt-1 text-xs text-indigo-900">
                            {t("modelCompare.trainingOverlapUsableRows", {
                              count: overlapResult.training_overlap.usable_reduced_csv_rows,
                            })}
                          </p>
                        )}
                      </div>
                      <details className="mt-4">
                        <summary className="cursor-pointer font-medium text-slate-900">
                          {t("modelCompare.overlapDescriptiveTitle")}
                        </summary>
                        <div className="mt-3">
                      <p className="font-semibold text-slate-900">
                        {t(overlapResult.subtraction_applied
                          ? "modelCompare.overlapResult"
                          : "modelCompare.overlapDirectResult", {
                            matches: overlapResult.matched_csv_row_count,
                            remaining: overlapResult.remaining_csv_row_count,
                            testRows: overlapResult.held_out_row_count,
                          })}
                      </p>
                      {overlapResult.subtraction_applied && (
                        <>
                          <p className="mt-1">
                            {t("modelCompare.overlapSubtraction", {
                              csvRows: overlapResult.csv_row_count,
                              removed: overlapResult.removed_csv_row_count,
                              remaining: overlapResult.remaining_csv_row_count,
                              editedTestRows: overlapResult.edited_held_out_row_count,
                            })}
                          </p>
                          <p className="mt-1">
                            {t("modelCompare.overlapBeforeSubtraction", {
                              matches: overlapResult.original_matches_before_subtraction,
                            })}
                          </p>
                        </>
                      )}
                      <p className="mt-1">
                        {t(overlapResult.compared_target
                          ? "modelCompare.overlapWithTarget"
                          : "modelCompare.overlapFeaturesOnly")}
                      </p>
                      {overlapResult.removed_rows?.length > 0 && (
                        <details className="mt-3">
                          <summary className="cursor-pointer font-medium text-slate-900">
                            {t("modelCompare.overlapRemovedRows", {
                              count: overlapResult.removed_csv_row_count,
                            })}
                          </summary>
                          <ul className="mt-1 max-h-48 list-disc overflow-auto pl-5">
                            {overlapResult.removed_rows.slice(0, 100).map((match) => (
                              <li key={match.csv_row}>
                                {t("modelCompare.overlapRowDetail", {
                                  csvRow: match.csv_row,
                                  testRows: match.held_out_rows.join(", "),
                                })}
                              </li>
                            ))}
                          </ul>
                          {overlapResult.removed_rows.length > 100 && (
                            <p className="mt-1">{t("modelCompare.overlapShowingFirst", { count: 100 })}</p>
                          )}
                        </details>
                      )}
                      {overlapResult.matched_rows?.length > 0 && (
                        <>
                          <p className="mt-3 font-medium text-slate-900">
                            {t(overlapResult.subtraction_applied
                              ? "modelCompare.overlapMatchedRows"
                              : "modelCompare.overlapDirectMatchedRows")}
                          </p>
                          <ul className="mt-1 max-h-48 list-disc overflow-auto pl-5">
                            {overlapResult.matched_rows.slice(0, 100).map((match) => (
                              <li key={match.csv_row}>
                                {t("modelCompare.overlapRowDetail", {
                                  csvRow: match.csv_row,
                                  testRows: match.held_out_rows.join(", "),
                                })}
                                {match.sample_ids?.length
                                  ? ` (${t("modelCompare.overlapSampleIds", { ids: match.sample_ids.join(", ") })})`
                                  : ""}
                              </li>
                            ))}
                          </ul>
                          {overlapResult.matched_rows.length > 100 && (
                            <p className="mt-1">{t("modelCompare.overlapShowingFirst", { count: 100 })}</p>
                          )}
                        </>
                      )}
                        </div>
                      </details>
                    </div>
                  )}
                </>
              ) : (
                <p className="mt-4 text-sm text-amber-800">
                  {t("modelCompare.overlapUnavailable")}
                </p>
              )}
            </section>

            <OverallShapeFunctionMaeSummary
              metrics={preview?.overall_shape_function_mae}
              featureNames={comparePayload.shared_features}
              featurePreviews={preview?.feature_previews}
              featureSelection={overallMaeFeatureSelection}
              onFeatureSelectionChange={handleOverallMaeFeatureToggle}
              previewLoading={previewLoading}
              useConfidence={useConfidence}
              language={language}
              t={t}
            />

            <HeldOutTestPerformanceSummary
              performance={
                showHeldOutTestPerformance && !previewLoading
                  ? preview?.held_out_test_performance
                  : null
              }
              hasOriginalTestData={Boolean(
                comparePayload?.left_artifact?.held_out_test_data,
              )}
              hasMainTrainingData={Boolean(
                preview?.main_training_performance_available,
              )}
              testDataSource={heldOutTestSource}
              onTestDataSourceChange={handleHeldOutTestSourceChange}
              loading={previewLoading}
              onEvaluate={handleEvaluateHeldOutTest}
              useConfidence={preview?.use_confidence ?? useConfidence}
              requiresSharedLineage={needsIqbSharedHeldOut(comparePayload.left_artifact)}
              sharedLineage={sharedHeldOutLineage}
              tallCharts={tallCharts}
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
                    {averageScoreMethod === "test_set"
                        ? t("modelCompare.testSetAverageDescription")
                        : t("modelCompare.mainDatasetAverageDescription", {
                            count: preview?.main_dataset_training_row_count || 0,
                          })}
                  </p>
                </div>
                {previewLoading && (
                  <div className="text-sm text-slate-500">
                    {t("modelCompare.recomputing")}
                  </div>
                )}
              </div>
              <form
                onSubmit={handleInterceptAdjustmentSubmit}
                className="mt-4 rounded-xl border border-indigo-200 bg-white px-4 py-3"
              >
                <div className="flex flex-wrap items-end gap-3">
                  <label className="min-w-56 flex-1 text-sm font-medium text-slate-900">
                    {t("modelCompare.interceptAdjustmentLabel")}
                    <input
                      type="number"
                      step="any"
                      required
                      value={interceptAdjustmentInput}
                      onChange={(event) => setInterceptAdjustmentInput(event.target.value)}
                      className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-900 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </label>

                  <button
                    type="submit"
                    disabled={previewLoading}
                    className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {t("modelCompare.applyInterceptAdjustment")}
                  </button>
                </div>
                <p className="mt-2 text-xs text-slate-600">
                  {t("modelCompare.interceptAdjustmentDescription")}
                </p>
                <p className="mt-2 text-xs text-slate-600">
                  {t("modelCompare.originalEditedBaseOffset")}: {formatMetricValue(
                    preview?.original_edited_base_prediction_offset ??
                      comparePayload?.right_artifact?.base_prediction_offset,
                    language,
                  )} {" · "}
                  {t("modelCompare.activeInterceptAdjustment")}: {formatSignedMetricValue(
                    preview?.intercept_adjustment ?? interceptAdjustment,
                    language,
                  )} {" · "}
                  {t("modelCompare.adjustedEditedBaseOffset")}: {formatMetricValue(
                    preview?.adjusted_edited_base_prediction_offset ??
                      Number(comparePayload?.right_artifact?.base_prediction_offset || 0) +
                        interceptAdjustment,
                    language,
                  )}
                </p>
              </form>
              <div className="mt-4 inline-flex flex-wrap rounded-xl border border-indigo-200 bg-white p-1">
                <button
                  type="button"
                  onClick={() => setAverageScoreMethod("test_set")}
                  disabled={!preview?.test_set_average_prediction_scores}
                  aria-pressed={averageScoreMethod === "test_set"}
                  title={
                    preview?.test_set_average_prediction_scores
                      ? ""
                      : t("modelCompare.testSetAverageUnavailable")
                  }
                  className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    averageScoreMethod === "test_set"
                      ? "bg-indigo-600 text-white"
                      : "text-slate-700 hover:bg-slate-100"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  {t("modelCompare.testSetAverage")}
                </button>
                <button
                  type="button"
                  onClick={() => setAverageScoreMethod("main_dataset")}
                  disabled={!preview?.main_dataset_average_prediction_scores}
                  aria-pressed={averageScoreMethod === "main_dataset"}
                  title={
                    preview?.main_dataset_average_prediction_scores
                      ? ""
                      : t("modelCompare.mainDatasetAverageUnavailable")
                  }
                  className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    averageScoreMethod === "main_dataset"
                      ? "bg-indigo-600 text-white"
                      : "text-slate-700 hover:bg-slate-100"
                  } disabled:cursor-not-allowed disabled:opacity-50`}
                >
                  {t("modelCompare.mainDatasetAverage")}
                </button>
              </div>
              <p className="mt-2 text-xs text-slate-600">
                {t("modelCompare.missingIndicatorsExcludedFromAverage")}
              </p>
              {!preview?.main_dataset_average_prediction_scores && (
                <p className="mt-2 text-xs text-slate-600">
                  {t("modelCompare.mainDatasetAverageUnavailable")}
                </p>
              )}
              {averageScoreMethod === "main_dataset" &&
                Number.isFinite(preview?.main_dataset_average_prediction_shift) && (
                  <p className="mt-2 text-sm font-medium text-indigo-900">
                    {t("modelCompare.developmentMeanEditShift")}: {formatSignedMetricValue(
                      preview.main_dataset_average_prediction_shift,
                      language,
                    )}{" · "}
                    {t("modelCompare.developmentSuggestedAdjustment")}: {formatSignedMetricValue(
                      -preview.main_dataset_average_prediction_shift,
                      language,
                    )}
                  </p>
                )}
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
                  {preview.mean_deviation_constraint.applied ? (
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
                  ) : preview.mean_deviation_constraint.satisfied ? (
                    t("modelCompare.meanDeviationWithinLimit")
                  ) : (
                    t("modelCompare.meanDeviationCannotMeetLimit")
                  )}
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
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm">
                <span className="font-medium text-amber-950">
                  {t("modelCompare.editedMinusReducedAveragePrediction")}
                </span>
                <span className="font-semibold tabular-nums text-amber-950">
                  {formatSignedMetricValue(
                    selectedContributionAverageScores?.edited_minus_reduced,
                    language,
                  )}
                </span>
              </div>
              {contributionSelectionIsActive && (
                <p className="mt-2 text-xs text-indigo-700">
                  {t("modelCompare.selectedContributionAverageView")}
                </p>
              )}
              <TrainingMeanDetails
                summary={preview?.training_mean_prediction_summary}
                preview={preview}
                originalArtifact={comparePayload?.left_artifact}
                reducedArtifact={comparePayload?.right_artifact}
                language={language}
                t={t}
              />
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <div className="w-full rounded-xl border border-indigo-200 bg-white px-4 py-3">
                  <h3 className="text-base font-semibold text-indigo-950">
                    {t("modelCompare.averageContributionsTitle")}
                  </h3>
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full min-w-[520px] text-sm">
                      <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="whitespace-nowrap px-2 py-2 font-semibold">
                            {t("modelCompare.contributionFeature")}
                          </th>
                          <th className="px-2 py-2 text-right font-semibold">
                            {t("modelCompare.meanPredictionChangeColumn")}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-slate-700">
                        {displayedContributionRows.map((row) => {
                          const isSelected =
                            row.is_internal_feature ||
                            contributionFeatureSelection[row.feature_name] !==
                              false;
                          const rowClassName =
                            row.is_internal_feature
                              ? "bg-slate-50"
                              : !isSelected
                                ? "cursor-pointer bg-slate-50 text-slate-400 grayscale"
                                : "cursor-pointer hover:bg-indigo-50";
                          return (
                            <tr
                              key={row.feature_name}
                              onClick={
                                row.is_internal_feature
                                  ? undefined
                                  : () =>
                                      handleContributionFeatureToggle(
                                        row.feature_name,
                                      )
                              }
                              className={rowClassName}
                            >
                              <td className="whitespace-nowrap px-2 py-2 font-medium">
                                {row.feature_name}
                              </td>
                              <td className="px-2 py-2 text-right tabular-nums">
                                {formatSignedMetricValue(
                                  row.difference,
                                  language,
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                      <tfoot className="border-t-2 border-indigo-200 bg-indigo-50 text-indigo-950">
                        <tr>
                          <th scope="row" className="whitespace-nowrap px-2 py-3 text-left font-semibold">
                            {t("modelCompare.totalMeanDeviationFromOriginal")}
                          </th>
                          <td className="px-2 py-3 text-right font-semibold tabular-nums">
                            {formatSignedMetricValue(
                              averagePredictionScores?.difference,
                              language,
                            )}
                          </td>
                        </tr>
                      </tfoot>
                    </table>
                  </div>
                </div>
                <div className="w-full rounded-xl border border-teal-200 bg-white px-4 py-3">
                  <h3 className="text-base font-semibold text-teal-950">
                    {t("modelCompare.reducedEditedContributionsTitle")}
                  </h3>
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full min-w-[520px] text-sm">
                      <thead className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                        <tr>
                          <th className="whitespace-nowrap px-2 py-2 font-semibold">
                            {t("modelCompare.contributionFeature")}
                          </th>
                          <th className="px-2 py-2 text-right font-semibold">
                            {t("modelCompare.reducedEditedMeanPredictionChangeColumn")}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 text-slate-700">
                        {displayedReducedEditedContributionRows.map((row) => {
                          const isSelected =
                            row.is_internal_feature ||
                            contributionFeatureSelection[row.feature_name] !==
                              false;
                          const rowClassName =
                            row.is_internal_feature
                              ? "bg-slate-50"
                              : !isSelected
                                ? "cursor-pointer bg-slate-50 text-slate-400 grayscale"
                                : "cursor-pointer hover:bg-teal-50";
                          return (
                            <tr
                              key={row.feature_name}
                              onClick={
                                row.is_internal_feature
                                  ? undefined
                                  : () =>
                                      handleContributionFeatureToggle(
                                        row.feature_name,
                                      )
                              }
                              className={rowClassName}
                            >
                              <td className="whitespace-nowrap px-2 py-2 font-medium">
                                {row.feature_name}
                              </td>
                              <td className="px-2 py-2 text-right tabular-nums">
                                {formatSignedMetricValue(
                                  row.edited - row.reduced,
                                  language,
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                      <tfoot className="border-t-2 border-teal-200 bg-teal-50 text-teal-950">
                        <tr>
                          <th scope="row" className="whitespace-nowrap px-2 py-3 text-left font-semibold">
                            {t("modelCompare.totalMeanDeviationFromOriginalReduced")}
                          </th>
                          <td className="px-2 py-3 text-right font-semibold tabular-nums">
                            {formatSignedMetricValue(
                              Number.isFinite(averagePredictionScores?.reduced) &&
                                Number.isFinite(averagePredictionScores?.edited)
                                ? averagePredictionScores.edited - averagePredictionScores.reduced
                                : null,
                              language,
                            )}
                          </td>
                        </tr>
                      </tfoot>
                    </table>
                  </div>
                </div>
              </div>
            </section>

            <AllFeatureMaeChangeCharts
              metrics={preview?.all_feature_mae_change}
              tallCharts={tallCharts}
              t={t}
            />

            <ExpertAverageConfidenceChart
              artifact={comparePayload.right_artifact}
              language={language}
              tallCharts={tallCharts}
              t={t}
            />

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
                    aria-pressed={showAlternateConfidenceAggregation}
                    disabled={showIndividualEditTraces || chartExpertSelectionEnabled}
                    onClick={() =>
                      setShowAlternateConfidenceAggregation((current) => !current)
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                      showAlternateConfidenceAggregation
                        ? "border-violet-200 bg-violet-50 text-violet-900 hover:bg-violet-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <span className="text-sm font-medium">
                      {showAlternateConfidenceAggregation
                        ? t("modelCompare.hideAlternateConfidenceAggregation")
                        : t("modelCompare.showAlternateConfidenceAggregation")}
                    </span>
                  </button>

                  <label className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                    <input
                      type="checkbox"
                      checked={featureMetricsOnOriginalTestSet}
                      disabled={
                        previewLoading ||
                        !comparePayload?.left_artifact?.held_out_test_data
                          ?.feature_rows?.length
                      }
                      onChange={(event) =>
                        handleFeatureMetricsTestSetToggle(event.target.checked)
                      }
                      className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 disabled:cursor-not-allowed"
                    />
                    <span>
                      <span className="block text-sm font-medium text-slate-800">
                        {t("modelCompare.featureMetricsOriginalTestSet")}
                      </span>
                      <span className="block text-xs text-slate-500">
                        {t("modelCompare.featureMetricsUniformGridDescription")}
                      </span>
                      {!comparePayload?.left_artifact?.held_out_test_data
                        ?.feature_rows?.length && (
                        <span className="block text-xs text-slate-500">
                          {t("modelCompare.featureMetricsOriginalTestSetUnavailable")}
                        </span>
                      )}
                    </span>
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
                    onClick={() =>
                      setShowIndividualParticipantMaeChangeChart(
                        (current) => !current,
                      )
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showIndividualParticipantMaeChangeChart
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {showIndividualParticipantMaeChangeChart
                        ? t("modelCompare.hideIndividualMaeChangeChart")
                        : t("modelCompare.showIndividualMaeChangeChart")}
                    </div>
                  </button>

                  <button
                    type="button"
                    aria-pressed={showConfidenceRatedMae}
                    onClick={() => setShowConfidenceRatedMae((current) => !current)}
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showConfidenceRatedMae
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <span className="text-sm font-medium">
                      {showConfidenceRatedMae
                        ? t("modelCompare.hideConfidenceRatedMae")
                        : t("modelCompare.showConfidenceRatedMae")}
                    </span>
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
                    aria-pressed={tallCharts}
                    onClick={() => setTallCharts((current) => !current)}
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      tallCharts
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <span className="text-sm font-medium">
                      {tallCharts
                        ? t("modelCompare.restoreChartHeight")
                        : t("modelCompare.makeChartsTaller")}
                    </span>
                  </button>

                  <button
                    type="button"
                    disabled={chartExpertSelectionEnabled}
                    onClick={() =>
                      setShowIndividualEditTraces((current) => !current)
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      showIndividualEditTraces
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    } disabled:cursor-not-allowed disabled:opacity-50`}
                  >
                    <div className="text-sm font-medium">
                      {showIndividualEditTraces
                        ? t("modelCompare.showAggregatedEdits")
                        : t("modelCompare.showIndividualEdits")}
                    </div>
                  </button>

                  <button
                    type="button"
                    aria-pressed={chartExpertSelectionEnabled}
                    onClick={() =>
                      setChartExpertSelectionEnabled((current) => !current)
                    }
                    className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                      chartExpertSelectionEnabled
                        ? "border-indigo-200 bg-indigo-50 text-indigo-900 hover:bg-indigo-100"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <div className="text-sm font-medium">
                      {chartExpertSelectionEnabled
                        ? t("modelCompare.disableChartExpertSelection")
                        : t("modelCompare.enableChartExpertSelection")}
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
                        {chartExpertSelectionEnabled &&
                          getFeatureChartExperts(featurePreview).length > 0 && (
                            <div className="rounded-xl border border-indigo-200 bg-indigo-50/60 p-3">
                              <div className="text-xs font-medium text-indigo-900">
                                {t("modelCompare.chartExpertSelectionHint")}
                              </div>
                              <div className="mt-2 flex flex-wrap gap-2">
                                {getFeatureChartExperts(featurePreview).map(
                                  ({ key, name }) => (
                                    <label
                                      key={key}
                                      className="flex cursor-pointer items-center gap-2 rounded-full border border-indigo-200 bg-white px-3 py-1.5 text-xs text-slate-800"
                                    >
                                      <input
                                        type="checkbox"
                                        checked={
                                          !hiddenChartExpertsByFeature[
                                            featurePreview.feature_name
                                          ]?.[key]
                                        }
                                        onChange={(event) => {
                                          const featureName =
                                            featurePreview.feature_name;
                                          const visible = event.target.checked;
                                          setHiddenChartExpertsByFeature(
                                            (current) => ({
                                              ...current,
                                              [featureName]: {
                                                ...current[featureName],
                                                [key]: !visible,
                                              },
                                            }),
                                          );
                                        }}
                                        className="h-4 w-4 rounded border-indigo-300 text-indigo-600 focus:ring-indigo-500"
                                      />
                                      {formatChartExpertLabel(name)}
                                    </label>
                                  ),
                                )}
                              </div>
                            </div>
                          )}
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
                    <div className="flex flex-col gap-4 xl:flex-row xl:items-start">
                      <CompareFeatureChart
                        featurePreview={featurePreview}
                        hiddenExpertKeys={
                          chartExpertSelectionEnabled
                            ? hiddenChartExpertsByFeature[
                                featurePreview.feature_name
                              ]
                            : null
                        }
                        showBaseTraces={showBaseTraces}
                        centerCharts={centerCharts}
                        showIndividualEditTraces={
                          showIndividualEditTraces || chartExpertSelectionEnabled
                        }
                        showAlternateConfidenceAggregation={
                          showAlternateConfidenceAggregation
                        }
                        useConfidence={preview?.use_confidence ?? useConfidence}
                        showConfidenceInLegend={showConfidenceInLegend}
                        legendOnRight={legendOnRight}
                        showLineMarkers={showLineMarkers}
                        tallCharts={tallCharts}
                        t={t}
                      />
                      {showIndividualParticipantMaeChangeChart && (
                        <IndividualParticipantMaeChangeChart
                          featurePreview={featurePreview}
                          showConfidenceRatedMae={showConfidenceRatedMae}
                          tallCharts={tallCharts}
                          hiddenExpertKeys={
                            chartExpertSelectionEnabled
                              ? hiddenChartExpertsByFeature[
                                  featurePreview.feature_name
                                ]
                              : null
                          }
                          t={t}
                        />
                      )}
                    </div>
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
