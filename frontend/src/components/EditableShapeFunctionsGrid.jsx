import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  useMemo,
} from "react";
import { createPortal } from "react-dom";
import Plot from "react-plotly.js";
import { createTranslator } from "../i18n";

const NUMERIC_X_PRECISION = 6;
const NUMERIC_BRUSH_SIGMA_RATIO = 0.08;
const NUMERIC_BRUSH_MIN_SIGMA = 0.25;
const NUMERIC_BRUSH_SOFT_MULTIPLIER = 1.8;
const NUMERIC_BRUSH_HARD_MULTIPLIER = 0.1;
const NUMERIC_BRUSH_RADIUS_MULTIPLIER = 3.0;
const NUMERIC_BRUSH_SMOOTHING_WINDOW = 5;
const UNSYNCED_Y_PADDING_RATIO = 0.15;
const UNSYNCED_Y_MIN_PADDING = 0.2;
const FLAT_Y_MIN_HALF_SPAN = 0.5;
const MISSING_CATEGORY_VALUE = "(Missing)";

const roundNumericX = (x) =>
  Math.round(Number(x) * 10 ** NUMERIC_X_PRECISION) / 10 ** NUMERIC_X_PRECISION;

const interpolateYAtX = (sortedPoints, targetX) => {
  if (!sortedPoints || sortedPoints.length === 0) return 0;
  if (sortedPoints.length === 1) return sortedPoints[0].y;

  if (targetX <= sortedPoints[0].x) return sortedPoints[0].y;
  if (targetX >= sortedPoints[sortedPoints.length - 1].x)
    return sortedPoints[sortedPoints.length - 1].y;

  for (let idx = 0; idx < sortedPoints.length - 1; idx += 1) {
    const left = sortedPoints[idx];
    const right = sortedPoints[idx + 1];
    if (targetX >= left.x && targetX <= right.x) {
      const span = right.x - left.x;
      if (span === 0) return left.y;
      const ratio = (targetX - left.x) / span;
      return left.y + ratio * (right.y - left.y);
    }
  }
  return sortedPoints[sortedPoints.length - 1].y;
};

const toSortedCurvePoints = (xValues, yValues) =>
  (xValues || [])
    .map((x, idx) => ({
      x: Number(x),
      y: Number(yValues?.[idx]),
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x);

const normalizeNumericPoints = (points) =>
  (points || [])
    .map((p) => ({
      x: roundNumericX(Number(p.x_value)),
      y: Number(p.y_value),
    }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x);

const buildEditableNumericCurvePoints = (shapeFunction, editedPoints) => {
  const originalCurve = toSortedCurvePoints(
    shapeFunction?.x_values || [],
    shapeFunction?.y_values || [],
  );
  if (originalCurve.length === 0) return [];

  const editMap = new Map();
  normalizeNumericPoints(editedPoints).forEach((point) => {
    editMap.set(String(point.x), point.y);
  });

  return originalCurve.map((point) => {
    const key = String(roundNumericX(point.x));
    return {
      x: point.x,
      y: editMap.has(key) ? editMap.get(key) : point.y,
    };
  });
};

const toEditedPointsFormat = (curvePoints) =>
  curvePoints.map((point) => ({
    x_value: roundNumericX(point.x),
    y_value: point.y,
  }));

const computeDynamicYRange = (values) => {
  const finiteValues = (values || []).filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return [-1, 1];
  }

  const yMin = Math.min(...finiteValues);
  const yMax = Math.max(...finiteValues);
  const span = yMax - yMin;

  if (span < 1e-8) {
    const center = yMin;
    const halfSpan = Math.max(
      Math.abs(center) * UNSYNCED_Y_PADDING_RATIO,
      FLAT_Y_MIN_HALF_SPAN,
    );
    return [center - halfSpan, center + halfSpan];
  }

  const padding = Math.max(
    span * UNSYNCED_Y_PADDING_RATIO,
    UNSYNCED_Y_MIN_PADDING,
  );
  return [yMin - padding, yMax + padding];
};

const buildNumericMissingBucketLayout = (xValues, missingBucket) => {
  if (
    !missingBucket ||
    !Number.isFinite(Number(missingBucket?.count)) ||
    Number(missingBucket.count) <= 0
  ) {
    return null;
  }

  const numericXValues = (xValues || [])
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
  if (numericXValues.length === 0) return null;

  const observedMin = Math.min(...numericXValues);
  const observedMax = Math.max(...numericXValues);
  const span = observedMax - observedMin;
  const scaleBase =
    span > 1e-9 ? span : Math.max(Math.abs(observedMin), Math.abs(observedMax), 1);
  const gap = Math.max(scaleBase * 0.14, 0.6);
  const width = Math.max(scaleBase * 0.12, 0.45);
  const anchor = observedMin - gap * 1.9;
  const dividerX = observedMin - gap * 0.8;
  const rangeMin = anchor - width * 1.1;
  const interactionCutoffX = observedMin - gap * 1.15;

  return {
    anchor,
    width,
    dividerX,
    rangeMin,
    interactionCutoffX,
    label: String(missingBucket.label || MISSING_CATEGORY_VALUE),
  };
};

const applyMovingAverageInWindow = (points, centerIndex, windowSize) => {
  if (windowSize <= 1 || points.length < 3) return points;
  const half = Math.floor(windowSize / 2);
  const result = points.map((point) => ({ ...point }));

  for (
    let idx = Math.max(1, centerIndex - half);
    idx <= Math.min(points.length - 2, centerIndex + half);
    idx += 1
  ) {
    let weightedSum = 0;
    let weightTotal = 0;
    for (
      let j = Math.max(0, idx - half);
      j <= Math.min(points.length - 1, idx + half);
      j += 1
    ) {
      const dist = Math.abs(j - idx);
      const weight = Math.exp(-(dist * dist) / (2 * Math.max(1, half) ** 2));
      weightedSum += points[j].y * weight;
      weightTotal += weight;
    }
    result[idx].y = weightedSum / weightTotal;
  }
  return result;
};

const applyGaussianBrushDeformation = (
  curvePoints,
  cursorX,
  deltaY,
  sigma,
  radiusMultiplier = NUMERIC_BRUSH_RADIUS_MULTIPLIER,
  smoothingWindow = NUMERIC_BRUSH_SMOOTHING_WINDOW,
) => {
  if (!curvePoints || curvePoints.length === 0 || !Number.isFinite(deltaY)) {
    return curvePoints || [];
  }
  if (!Number.isFinite(sigma) || sigma <= 0) return curvePoints;

  const radius = sigma * radiusMultiplier;

  // Core brush logic:
  // We do NOT move a single point. We deform a whole local neighborhood using
  // a Gaussian falloff so influence is strongest at the cursor and fades smoothly.
  const deformed = curvePoints.map((point) => {
    const dx = point.x - cursorX;
    if (Math.abs(dx) > radius) return { ...point };
    const weight = Math.exp(-0.5 * (dx / sigma) ** 2);
    return { ...point, y: point.y + deltaY * weight };
  });

  // Optional local smoothing pass inside the active brush radius to avoid any
  // subtle jaggedness from repeated incremental updates.
  const smoothed = deformed.map((point) => ({ ...point }));
  for (let idx = 0; idx < deformed.length; idx += 1) {
    if (Math.abs(deformed[idx].x - cursorX) > radius) continue;
    const locallySmoothed = applyMovingAverageInWindow(
      deformed,
      idx,
      smoothingWindow,
    );
    const blend = Math.exp(-0.5 * ((deformed[idx].x - cursorX) / sigma) ** 2);
    smoothed[idx].y =
      deformed[idx].y * (1 - blend) + locallySmoothed[idx].y * blend;
  }

  return smoothed;
};

const buildDragHighlightSegments = (
  curvePoints,
  cursorX,
  sigma,
  radiusMultiplier = NUMERIC_BRUSH_RADIUS_MULTIPLIER,
) => {
  if (!curvePoints || curvePoints.length < 2) return [];
  if (!Number.isFinite(cursorX) || !Number.isFinite(sigma) || sigma <= 0) {
    return [];
  }

  const radius = sigma * radiusMultiplier;
  const segments = [];

  for (let idx = 0; idx < curvePoints.length - 1; idx += 1) {
    const left = curvePoints[idx];
    const right = curvePoints[idx + 1];
    const midX = (left.x + right.x) / 2;
    const dist = Math.abs(midX - cursorX);
    if (dist > radius) continue;

    // Highest opacity at the cursor, smoothly fading to both sides.
    const alpha = 0.1 + 0.85 * Math.exp(-0.5 * (dist / sigma) ** 2);
    segments.push({
      x: [left.x, right.x],
      y: [left.y, right.y],
      type: "scatter",
      mode: "lines",
      line: {
        color: `rgba(239, 68, 68, ${alpha.toFixed(3)})`,
        width: 4,
      },
      hoverinfo: "skip",
      showlegend: false,
    });
  }

  return segments;
};

const hasShapeFunctionDistribution = (distribution) => {
  if (!distribution || typeof distribution !== "object") return false;
  if (distribution.chart_type === "numeric") {
    return Array.isArray(distribution.bins) && distribution.bins.length > 0;
  }
  if (distribution.chart_type === "categorical") {
    return Array.isArray(distribution.counts) && distribution.counts.length > 0;
  }
  return false;
};

const getFeatureProvenance = (featureSchemaEntry) =>
  featureSchemaEntry?.feature_provenance || null;

const getConstructionLabel = (constructionType, t) => {
  switch (constructionType) {
    case "item_mean":
      return t("shapeFunctions.detailsItemMean");
    case "iqb_scale":
      return t("shapeFunctions.detailsIqbScale");
    case "raw_source":
      return t("shapeFunctions.detailsRawSource");
    default:
      return t("shapeFunctions.detailsUnknownConstruction");
  }
};

const getDistributionCountSummary = (distribution) => {
  if (!distribution || typeof distribution !== "object") return null;

  if (distribution.chart_type === "numeric") {
    const totalCount = Number(distribution.total_count) || 0;
    const missingCount = Number(distribution.missing_count) || 0;
    if (missingCount > 0) {
      return {
        totalCount,
        missingCount,
        answeredCount: Math.max(0, totalCount - missingCount),
        includesMissing: true,
      };
    }
  }

  if (distribution.chart_type === "categorical") {
    const counts = Array.isArray(distribution.counts) ? distribution.counts : [];
    const missingEntry = counts.find(
      (entry) => String(entry?.x_value) === MISSING_CATEGORY_VALUE,
    );
    if (missingEntry) {
      const missingCount = Number(missingEntry.count) || 0;
      const totalCount = Number(distribution.total_count) || 0;
      return {
        totalCount,
        missingCount,
        answeredCount: Math.max(0, totalCount - missingCount),
        includesMissing: true,
      };
    }
  }

  return {
    totalCount: Number(distribution.total_count) || 0,
    missingCount: 0,
    answeredCount: Number(distribution.total_count) || 0,
    includesMissing: false,
  };
};

const buildSourceRows = (provenance) => {
  if (!provenance) return [];
  if (Array.isArray(provenance.source_details) && provenance.source_details.length) {
    return provenance.source_details.filter(
      (row) => row && (row.variable || row.label),
    );
  }
  const sourceVariables = Array.isArray(provenance.source_variables)
    ? provenance.source_variables
    : [];
  const sourceLabels = Array.isArray(provenance.source_labels)
    ? provenance.source_labels
    : [];
  const maxLength = Math.max(sourceVariables.length, sourceLabels.length);

  return Array.from({ length: maxLength }, (_, index) => ({
    variable: sourceVariables[index] || "",
    label: sourceLabels[index] || "",
  })).filter((row) => row.variable || row.label);
};

const FeatureDetailsDrawer = ({
  isOpen,
  featureName,
  featureSchemaEntry,
  onClose,
  t,
}) => {
  const provenance = getFeatureProvenance(featureSchemaEntry);
  const sourceRows = buildSourceRows(provenance);
  const sourceVariables = Array.isArray(provenance?.source_variables)
    ? provenance.source_variables
    : [];
  const sharedResponseOptions = Array.isArray(provenance?.response_options)
    ? provenance.response_options
    : [];
  const hasMetadata = Boolean(provenance);
  const constructionLabel = getConstructionLabel(
    provenance?.construction_type,
    t,
  );

  useEffect(() => {
    if (!isOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[1100] flex justify-end bg-slate-900/35 backdrop-blur-[1px]"
      onClick={onClose}
    >
      <aside
        className="flex h-full w-full max-w-xl flex-col overflow-hidden bg-white shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="border-b border-slate-200 px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                {t("shapeFunctions.detailsDrawerTitle")}
              </p>
              <h3 className="mt-1 text-lg font-semibold text-slate-900">
                {featureName}
              </h3>
              <p className="mt-1 text-sm text-slate-500">
                {t("shapeFunctions.detailsSummary")}
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700"
              title={t("common.close")}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {t("shapeFunctions.detailsConstruction")}
              </div>
              <div className="mt-1 text-sm font-medium text-slate-800">
                {constructionLabel}
              </div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {t("shapeFunctions.detailsCategory")}
              </div>
              <div className="mt-1 text-sm font-medium text-slate-800">
                {provenance?.category ||
                  featureSchemaEntry?.feature_type ||
                  t("common.unknown")}
              </div>
            </div>
          </div>

          {provenance?.source_count > 0 && (
            <div className="mt-4 inline-flex rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-xs font-medium text-slate-600">
              {t("shapeFunctions.detailsSourceCount", {
                count: provenance.source_count,
              })}
            </div>
          )}

          {!hasMetadata && (
            <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
              {t("shapeFunctions.detailsUnavailable")}
            </div>
          )}

          {sourceVariables.length > 0 && (
            <section className="mt-6">
              <h4 className="text-sm font-semibold text-slate-800">
                {t("shapeFunctions.detailsSourceVariables")}
              </h4>
              <div className="mt-3 flex flex-wrap gap-2">
                {sourceVariables.map((variable) => (
                  <code
                    key={variable}
                    className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-700"
                  >
                    {variable}
                  </code>
                ))}
              </div>
            </section>
          )}

          {sharedResponseOptions.length > 0 && (
            <section className="mt-6">
              <h4 className="text-sm font-semibold text-slate-800">
                {t("shapeFunctions.detailsAnswerScale")}
              </h4>
              <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
                <div className="space-y-2">
                  {sharedResponseOptions.map((option, index) => (
                    <div
                      key={`${option?.value || "option"}-${index}`}
                      className="flex items-start gap-3 text-sm text-slate-700"
                    >
                      <span className="min-w-[3rem] rounded-md bg-white px-2 py-1 text-center font-mono text-xs text-slate-600 shadow-sm">
                        {option?.value || "?"}
                      </span>
                      <span className="pt-1">{option?.label || ""}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}

          {sourceRows.length > 0 && (
            <section className="mt-6">
              <h4 className="text-sm font-semibold text-slate-800">
                {t("shapeFunctions.detailsSourceQuestions")}
              </h4>
              <div className="mt-3 space-y-3">
                {sourceRows.map((row, index) => (
                  <div
                    key={`${row.variable}-${index}`}
                    className="rounded-xl border border-slate-200 bg-white px-4 py-3"
                  >
                    {row.variable && (
                      <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                        {row.variable}
                      </div>
                    )}
                    <div className="mt-1 text-sm text-slate-800">
                      {row.label || row.variable}
                    </div>
                    {sharedResponseOptions.length === 0 &&
                      Array.isArray(row.response_options) &&
                      row.response_options.length > 0 && (
                        <div className="mt-3 space-y-2 border-t border-slate-200 pt-3">
                          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                            {t("shapeFunctions.detailsAnswerScale")}
                          </div>
                          {row.response_options.map((option, optionIndex) => (
                            <div
                              key={`${row.variable || index}-option-${optionIndex}`}
                              className="flex items-start gap-3 text-sm text-slate-700"
                            >
                              <span className="min-w-[3rem] rounded-md bg-slate-100 px-2 py-1 text-center font-mono text-xs text-slate-600">
                                {option?.value || "?"}
                              </span>
                              <span>{option?.label || ""}</span>
                            </div>
                          ))}
                        </div>
                      )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {provenance?.transformation && (
            <section className="mt-6">
              <h4 className="text-sm font-semibold text-slate-800">
                {t("shapeFunctions.detailsTransformation")}
              </h4>
              <p className="mt-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                {provenance.transformation}
              </p>
            </section>
          )}

          {provenance?.missing_value_handling &&
            provenance.missing_value_handling !== provenance.transformation && (
              <section className="mt-6">
                <h4 className="text-sm font-semibold text-slate-800">
                  {t("shapeFunctions.detailsMissingHandling")}
                </h4>
                <p className="mt-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                  {provenance.missing_value_handling}
                </p>
              </section>
            )}

          {provenance?.selection_rationale && (
            <section className="mt-6">
              <h4 className="text-sm font-semibold text-slate-800">
                {t("shapeFunctions.detailsRationale")}
              </h4>
              <p className="mt-2 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
                {provenance.selection_rationale}
              </p>
            </section>
          )}
        </div>
      </aside>
    </div>,
    document.body,
  );
};

const BrushHardnessControl = ({
  brushHardness,
  onChange,
  t,
  className = "",
}) => (
  <div
    className={`bg-white border border-slate-200 rounded-lg p-3 ${className}`.trim()}
  >
    <div className="flex items-center justify-between gap-4 mb-1">
      <span className="text-sm font-medium text-slate-700">
        {t("shapeFunctions.lineBrushHardness")}
      </span>
      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700">
        {brushHardness}
      </span>
    </div>
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-500 w-9">
        {t("shapeFunctions.soft")}
      </span>
      <input
        type="range"
        min="0"
        max="100"
        value={brushHardness}
        onChange={(e) =>
          onChange(Math.max(0, Math.min(100, Number(e.target.value))))
        }
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
        style={{
          background: `linear-gradient(to right, #10b981 0%, #10b981 ${brushHardness}%, #e5e7eb ${brushHardness}%, #e5e7eb 100%)`,
        }}
      />
      <span className="text-xs text-slate-500 w-9 text-right">
        {t("shapeFunctions.hard")}
      </span>
    </div>
  </div>
);

const EditableShapeFunctionChart = ({
  shapeFunction,
  comparisonShapeFunction = null,
  editedPoints,
  onPointEdit,
  isEditing,
  brushHardness = 50,
  enlarged = false,
  sharedYRange = null,
  t,
}) => {
  if (!shapeFunction) return null;

  const { feature_name, x_values, y_values, feature_type, x_tick_labels } =
    shapeFunction;
  const isNumeric = feature_type === "numeric";
  const missingBucket = shapeFunction?.missing_bucket || null;
  const xTickLabels =
    Array.isArray(x_tick_labels) && x_tick_labels.length === x_values.length
      ? x_tick_labels
      : null;
  const containerRef = useRef(null);
  const plotRef = useRef(null);

  // Common drag state
  const [isDragging, setIsDragging] = useState(false);

  // Categorical drag state (index-based)
  const [dragPointIndex, setDragPointIndex] = useState(null);
  const [localYValues, setLocalYValues] = useState(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Numeric continuous drag state (coordinate-based)
  const [dragXValue, setDragXValue] = useState(null);
  const [dragYValue, setDragYValue] = useState(null);
  const [dragCurvePoints, setDragCurvePoints] = useState([]);

  // Precise value entry modal state
  const [preciseEntry, setPreciseEntry] = useState(null);
  const [preciseValue, setPreciseValue] = useState("");
  const [isDistributionOpen, setIsDistributionOpen] = useState(false);

  // Refs for drag handling
  const yAxisRangeRef = useRef({ min: -10, max: 10 });
  const xAxisRangeRef = useRef({ min: 0, max: 1 });
  const plotBoundsRef = useRef({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    height: 1,
    width: 1,
  });
  const hoveredPointRef = useRef(null);
  const dragStartYRef = useRef(null);
  const lastDragYRef = useRef(null);
  const isDraggingRef = useRef(false);
  const dragCurvePointsRef = useRef([]);
  const dragModifiedRef = useRef(false);
  const lastMouseDownRef = useRef({ time: 0, pointIndex: null, xValue: null });
  const preciseEntryOpenRef = useRef(false);

  useEffect(() => {
    preciseEntryOpenRef.current = preciseEntry !== null;
  }, [preciseEntry]);

  const stopDragInteraction = useCallback(() => {
    setIsDragging(false);
    isDraggingRef.current = false;
    setDragPointIndex(null);
    setLocalYValues(null);
    setDragXValue(null);
    setDragYValue(null);
    setDragCurvePoints([]);
    dragCurvePointsRef.current = [];
    dragModifiedRef.current = false;
    lastDragYRef.current = null;
    dragStartYRef.current = null;
  }, []);

  // Get the current y values (with edits applied) - used for categorical
  const getCurrentYValues = useCallback(() => {
    if (!editedPoints || editedPoints.length === 0) {
      return [...y_values];
    }

    const editMap = new Map();
    editedPoints.forEach((p) => {
      editMap.set(String(p.x_value), p.y_value);
    });

    return x_values.map((x, idx) => {
      const xKey = String(x);
      if (editMap.has(xKey)) {
        return editMap.get(xKey);
      }
      return y_values[idx];
    });
  }, [editedPoints, x_values, y_values]);

  // Use local values during drag, otherwise use computed values (categorical only)
  const currentYValues =
    !isNumeric && localYValues !== null ? localYValues : getCurrentYValues();

  // Merge original line data with edits + drag point for numeric display
  const mergedLineData = useMemo(() => {
    if (!isNumeric) return null;
    const currentCurve =
      isDragging && dragCurvePoints.length > 0
        ? dragCurvePoints
        : buildEditableNumericCurvePoints(shapeFunction, editedPoints || []);
    return {
      x: currentCurve.map((point) => point.x),
      y: currentCurve.map((point) => point.y),
    };
  }, [isNumeric, isDragging, dragCurvePoints, shapeFunction, editedPoints]);

  // Linear interpolation to get y at an arbitrary x from the current curve
  const getYAtX = useCallback(
    (targetX) => {
      const currentCurve = buildEditableNumericCurvePoints(
        shapeFunction,
        editedPoints || [],
      );
      return interpolateYAtX(currentCurve, targetX);
    },
    [shapeFunction, editedPoints],
  );

  // Check if any points have been edited
  const hasEdits =
    editedPoints &&
    editedPoints.length > 0 &&
    editedPoints.some((p) => {
      if (isNumeric) {
        const pxNum = Number(p.x_value);
        const idx = x_values.findIndex(
          (x) => Math.abs(Number(x) - pxNum) < 0.0001,
        );
        if (idx === -1) return true;
        return Math.abs(y_values[idx] - p.y_value) > 0.001;
      }
      const idx = x_values.findIndex(
        (x) => String(x) === String(p.x_value) || x === p.x_value,
      );
      return idx !== -1 && Math.abs(y_values[idx] - p.y_value) > 0.001;
    });
  const distribution = shapeFunction?.distribution || null;
  const hasDistribution = useMemo(
    () => hasShapeFunctionDistribution(distribution),
    [distribution],
  );
  const numericMissingLayout = useMemo(
    () =>
      isNumeric ? buildNumericMissingBucketLayout(x_values, missingBucket) : null,
    [isNumeric, x_values, missingBucket],
  );

  // Calculate y-axis range
  const allYValues = isNumeric
    ? [
        ...y_values,
        ...(mergedLineData?.y || []),
        ...(dragYValue !== null ? [dragYValue] : []),
        ...(missingBucket && Number.isFinite(Number(missingBucket.y_value))
          ? [Number(missingBucket.y_value)]
          : []),
        ...(comparisonShapeFunction?.missing_bucket &&
        Number.isFinite(Number(comparisonShapeFunction.missing_bucket.y_value))
          ? [Number(comparisonShapeFunction.missing_bucket.y_value)]
          : []),
      ]
    : [...y_values, ...currentYValues];
  const yRange = sharedYRange || computeDynamicYRange(allYValues);

  // Calculate x-axis range (numeric only, explicit for coordinate conversion)
  const numXVals = isNumeric ? x_values.map(Number) : [];
  const xDataMin = isNumeric ? Math.min(...numXVals) : 0;
  const xDataMax = isNumeric ? Math.max(...numXVals) : 1;
  const xPad = isNumeric ? Math.max((xDataMax - xDataMin) * 0.05, 0.1) : 0;
  const xRange = isNumeric
    ? [
        Math.min(
          xDataMin - xPad,
          numericMissingLayout ? numericMissingLayout.rangeMin : xDataMin - xPad,
        ),
        xDataMax + xPad,
      ]
    : [xDataMin - xPad, xDataMax + xPad];

  useEffect(() => {
    setIsDistributionOpen(false);
  }, [feature_name]);

  useEffect(() => {
    if (!hasDistribution) {
      setIsDistributionOpen(false);
    }
  }, [hasDistribution]);

  // Update axis range refs
  useEffect(() => {
    yAxisRangeRef.current = { min: yRange[0], max: yRange[1] };
  }, [yRange[0], yRange[1]]);

  useEffect(() => {
    if (isNumeric) {
      xAxisRangeRef.current = { min: xRange[0], max: xRange[1] };
    }
  }, [isNumeric, xRange[0], xRange[1]]);

  // Update plot bounds when layout changes
  const updatePlotBounds = useCallback(() => {
    if (!containerRef.current) return;

    const plotArea = containerRef.current.querySelector(
      ".js-plotly-plot .plot-container",
    );
    if (!plotArea) return;

    const rect = plotArea.getBoundingClientRect();
    const marginTop = 40;
    const marginBottom = 50;
    const marginLeft = 50;
    const marginRight = 20;

    plotBoundsRef.current = {
      top: rect.top + marginTop,
      bottom: rect.bottom - marginBottom,
      left: rect.left + marginLeft,
      right: rect.right - marginRight,
      height: rect.height - marginTop - marginBottom,
      width: rect.width - marginLeft - marginRight,
    };
  }, []);

  // Convert client Y to data Y
  const clientYToDataY = useCallback((clientY) => {
    const { top, bottom, height } = plotBoundsRef.current;
    if (height <= 0) return null;

    const normalized = (bottom - clientY) / height;
    const clamped = Math.max(0, Math.min(1, normalized));

    const { min, max } = yAxisRangeRef.current;
    return min + clamped * (max - min);
  }, []);

  // Convert client X to data X (numeric features only)
  const clientXToDataX = useCallback((clientX) => {
    const { left, width } = plotBoundsRef.current;
    if (width <= 0) return null;

    const normalized = (clientX - left) / width;
    const clamped = Math.max(0, Math.min(1, normalized));

    const { min, max } = xAxisRangeRef.current;
    return min + clamped * (max - min);
  }, []);

  // Clamp x for numeric edits (no snapping to existing points)
  const clampX = useCallback(
    (rawX) => Math.max(xDataMin, Math.min(xDataMax, rawX)),
    [xDataMin, xDataMax],
  );

  const getBrushSigma = useCallback(() => {
    const xSpan = xDataMax - xDataMin;
    const baseSigma = Math.max(
      xSpan * NUMERIC_BRUSH_SIGMA_RATIO,
      NUMERIC_BRUSH_MIN_SIGMA,
    );
    const normalizedHardness = Math.max(
      0,
      Math.min(1, Number(brushHardness) / 100),
    );
    const hardnessMultiplier =
      NUMERIC_BRUSH_SOFT_MULTIPLIER +
      (NUMERIC_BRUSH_HARD_MULTIPLIER - NUMERIC_BRUSH_SOFT_MULTIPLIER) *
        normalizedHardness;
    return baseSigma * hardnessMultiplier;
  }, [xDataMax, xDataMin, brushHardness]);

  const applyPreciseNumericEdit = useCallback(
    (centerX, targetY) => {
      const currentCurve = buildEditableNumericCurvePoints(
        shapeFunction,
        editedPoints || [],
      );
      if (currentCurve.length === 0) return;
      const currentY = interpolateYAtX(currentCurve, centerX);
      const deltaY = targetY - currentY;
      if (Math.abs(deltaY) <= 1e-8) return;

      const updatedCurve = applyGaussianBrushDeformation(
        currentCurve,
        centerX,
        deltaY,
        getBrushSigma(),
        NUMERIC_BRUSH_RADIUS_MULTIPLIER,
        1,
      );

      onPointEdit(
        feature_name,
        toEditedPointsFormat(updatedCurve),
        null,
        feature_type,
      );
    },
    [
      shapeFunction,
      editedPoints,
      getBrushSigma,
      onPointEdit,
      feature_name,
      feature_type,
    ],
  );

  // ---- Categorical drag effect (mousemove / mouseup on window) ----
  useEffect(() => {
    if (isNumeric) return;
    if (!isDragging || dragPointIndex === null) return;

    const handleMouseMove = (e) => {
      if (!isDraggingRef.current || preciseEntryOpenRef.current) return;
      e.preventDefault();
      const newY = clientYToDataY(e.clientY);
      if (newY === null) return;

      setLocalYValues((prev) => {
        if (!prev) return prev;
        const updated = [...prev];
        updated[dragPointIndex] = newY;
        return updated;
      });
    };

    const handleMouseUp = (e) => {
      if (!isDraggingRef.current || preciseEntryOpenRef.current) {
        stopDragInteraction();
        return;
      }
      e.preventDefault();
      const movedSignificantly =
        dragStartYRef.current !== null &&
        Math.abs(e.clientY - dragStartYRef.current) > 3;

      if (movedSignificantly && localYValues && dragPointIndex !== null) {
        const xValue = x_values[dragPointIndex];
        const newY = localYValues[dragPointIndex];
        onPointEdit(feature_name, xValue, newY, feature_type);
      }

      stopDragInteraction();
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [
    isNumeric,
    isDragging,
    dragPointIndex,
    localYValues,
    x_values,
    feature_name,
    feature_type,
    onPointEdit,
    clientYToDataY,
    stopDragInteraction,
  ]);

  // ---- Numeric continuous drag effect (mousemove / mouseup on window) ----
  useEffect(() => {
    if (!isNumeric) return;
    if (!isDragging || dragXValue === null) return;

    const handleMouseMove = (e) => {
      if (!isDraggingRef.current || preciseEntryOpenRef.current) return;
      e.preventDefault();
      const newX = clientXToDataX(e.clientX);
      const newY = clientYToDataY(e.clientY);
      if (newX !== null && newY !== null) {
        const clampedX = clampX(newX);
        if (lastDragYRef.current === null) {
          lastDragYRef.current = newY;
        }
        const deltaY = newY - lastDragYRef.current;
        lastDragYRef.current = newY;

        if (Math.abs(deltaY) > 1e-8 && dragCurvePointsRef.current.length > 0) {
          const deformed = applyGaussianBrushDeformation(
            dragCurvePointsRef.current,
            clampedX,
            deltaY,
            getBrushSigma(),
          );
          dragCurvePointsRef.current = deformed;
          setDragCurvePoints(deformed);
          dragModifiedRef.current = true;
        }

        setDragXValue(clampedX);
        setDragYValue(newY);
      }
    };

    const handleMouseUp = (e) => {
      if (!isDraggingRef.current || preciseEntryOpenRef.current) {
        stopDragInteraction();
        return;
      }
      e.preventDefault();
      if (dragModifiedRef.current && dragCurvePointsRef.current.length > 0) {
        onPointEdit(
          feature_name,
          toEditedPointsFormat(dragCurvePointsRef.current),
          null,
          feature_type,
        );
      }

      stopDragInteraction();
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [
    isNumeric,
    isDragging,
    dragXValue,
    feature_name,
    feature_type,
    onPointEdit,
    clientXToDataX,
    clientYToDataY,
    clampX,
    getBrushSigma,
    stopDragInteraction,
  ]);

  // ---- Categorical: start drag helper ----
  const startCategoricalDrag = useCallback(
    (pointIndex, clientY) => {
      if (!isEditing || isDraggingRef.current || preciseEntryOpenRef.current)
        return;
      updatePlotBounds();
      setDragPointIndex(pointIndex);
      setLocalYValues([...getCurrentYValues()]);
      setIsDragging(true);
      isDraggingRef.current = true;
      dragStartYRef.current = clientY;
    },
    [isEditing, getCurrentYValues, updatePlotBounds],
  );

  // ---- Categorical: Plotly hover callbacks ----
  const handleHover = useCallback(
    (eventData) => {
      if (isNumeric || !isEditing) return;
      if (!eventData.points || eventData.points.length === 0) {
        setHoveredPoint(null);
        return;
      }
      const point = eventData.points[0];
      const editableTraceIndex = hasEdits ? 1 : 0;
      if (point.curveNumber === editableTraceIndex) {
        setHoveredPoint(point.pointIndex);
        hoveredPointRef.current = point.pointIndex;
      } else {
        setHoveredPoint(null);
        hoveredPointRef.current = null;
      }
    },
    [isNumeric, isEditing, hasEdits],
  );

  const handleUnhover = useCallback(() => {
    if (isNumeric) return;
    if (!isDragging) {
      setHoveredPoint(null);
      hoveredPointRef.current = null;
    }
  }, [isNumeric, isDragging]);

  // ---- Categorical: native mousedown (hover-then-click) ----
  useEffect(() => {
    if (isNumeric) return;
    const container = containerRef.current;
    if (!container || !isEditing) return;

    const onMouseDown = (e) => {
      if (preciseEntryOpenRef.current) return;
      const pointIndex =
        hoveredPointRef.current ?? lastMouseDownRef.current.pointIndex;
      if (pointIndex === null) return;

      e.preventDefault();
      const now = Date.now();

      if (
        lastMouseDownRef.current.pointIndex === pointIndex &&
        now - lastMouseDownRef.current.time < 400
      ) {
        lastMouseDownRef.current = { time: 0, pointIndex: null, xValue: null };
        stopDragInteraction();

        const xValue = x_values[pointIndex];
        const currentY = getCurrentYValues()[pointIndex];
        const displayX =
          typeof xValue === "number" ? parseFloat(xValue.toFixed(3)) : xValue;
        setPreciseValue(currentY.toFixed(3));
        setPreciseEntry({ xValue, displayX, pointIndex });
      } else {
        lastMouseDownRef.current = { time: now, pointIndex, xValue: null };
        startCategoricalDrag(pointIndex, e.clientY);
      }
    };

    const onDblClick = (e) => e.preventDefault();
    container.addEventListener("mousedown", onMouseDown, true);
    container.addEventListener("dblclick", onDblClick, true);
    return () => {
      container.removeEventListener("mousedown", onMouseDown, true);
      container.removeEventListener("dblclick", onDblClick, true);
    };
  }, [
    isNumeric,
    isEditing,
    x_values,
    startCategoricalDrag,
    getCurrentYValues,
    stopDragInteraction,
  ]);

  // ---- Numeric continuous: native mousedown (click-anywhere) ----
  useEffect(() => {
    if (!isNumeric) return;
    const container = containerRef.current;
    if (!container || !isEditing) return;

    const onMouseDown = (e) => {
      if (preciseEntryOpenRef.current) return;
      // Only respond to clicks inside the plot area
      updatePlotBounds();
      const { left, right, top, bottom } = plotBoundsRef.current;
      if (
        e.clientX < left ||
        e.clientX > right ||
        e.clientY < top ||
        e.clientY > bottom
      )
        return;

      e.preventDefault();
      const rawX = clientXToDataX(e.clientX);
      const rawY = clientYToDataY(e.clientY);
      if (rawX === null || rawY === null) return;
      if (
        numericMissingLayout &&
        rawX <= Number(numericMissingLayout.interactionCutoffX)
      ) {
        return;
      }

      const clampedX = roundNumericX(clampX(rawX));
      const now = Date.now();

      // Double-click detection
      const xSpan = xDataMax - xDataMin;
      const dblClickXThreshold = xSpan > 0 ? xSpan * 0.02 : 0.01;
      if (
        lastMouseDownRef.current.xValue !== null &&
        Math.abs(lastMouseDownRef.current.xValue - clampedX) <
          dblClickXThreshold &&
        now - lastMouseDownRef.current.time < 400
      ) {
        lastMouseDownRef.current = { time: 0, pointIndex: null, xValue: null };
        // Cancel any drag
        stopDragInteraction();

        const currentY = getYAtX(clampedX);
        setPreciseValue(currentY.toFixed(3));
        setPreciseEntry({
          xValue: clampedX,
          displayX:
            typeof clampedX === "number" ? clampedX.toFixed(3) : clampedX,
          pointIndex: null,
        });
        return;
      }

      lastMouseDownRef.current = {
        time: now,
        pointIndex: null,
        xValue: clampedX,
      };

      // Start drag
      const startingCurve = buildEditableNumericCurvePoints(
        shapeFunction,
        editedPoints || [],
      );
      dragCurvePointsRef.current = startingCurve;
      setDragCurvePoints(startingCurve);
      dragModifiedRef.current = false;
      lastDragYRef.current = rawY;
      setDragXValue(clampedX);
      setDragYValue(rawY);
      setIsDragging(true);
      isDraggingRef.current = true;
      dragStartYRef.current = e.clientY;
    };

    const onDblClick = (e) => e.preventDefault();
    container.addEventListener("mousedown", onMouseDown, true);
    container.addEventListener("dblclick", onDblClick, true);
    return () => {
      container.removeEventListener("mousedown", onMouseDown, true);
      container.removeEventListener("dblclick", onDblClick, true);
    };
  }, [
    isNumeric,
    isEditing,
    clientXToDataX,
    clientYToDataY,
    clampX,
    getYAtX,
    updatePlotBounds,
    xDataMax,
    xDataMin,
    stopDragInteraction,
    shapeFunction,
    editedPoints,
    numericMissingLayout,
  ]);

  // Build traces
  let data;

  if (isNumeric) {
    // ---- Numeric continuous traces ----
    const merged = mergedLineData || { x: x_values, y: y_values };

    const originalTrace = {
      x: x_values,
      y: y_values,
      type: "scatter",
      mode: "lines",
      name: t("shapeFunctions.original"),
      line: { color: "#9ca3af", width: 1.5, dash: "dash" },
      opacity: 0.6,
      showlegend: hasEdits,
      hoverinfo: "skip",
    };

    const currentTrace = {
      x: merged.x,
      y: merged.y,
      type: "scatter",
      mode: "lines",
      name: hasEdits
        ? t("shapeFunctions.edited")
        : t("shapeFunctions.current"),
      line: { color: hasEdits ? "#10b981" : "#3b82f6", width: 2.5 },
      fill: "tozeroy",
      fillcolor: hasEdits
        ? "rgba(16, 185, 129, 0.1)"
        : "rgba(59, 130, 246, 0.1)",
      hovertemplate: `<b>%{x:.3f}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<extra></extra>`,
    };

    const comparisonTrace =
      comparisonShapeFunction &&
      Array.isArray(comparisonShapeFunction.x_values) &&
      Array.isArray(comparisonShapeFunction.y_values)
        ? {
            x: comparisonShapeFunction.x_values,
            y: comparisonShapeFunction.y_values,
            type: "scatter",
            mode: "lines",
            name: t("shapeFunctions.comparison"),
            line: { color: "#f97316", width: 2, dash: "dot" },
            hovertemplate: `<b>%{x:.3f}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<extra>${t("shapeFunctions.comparison")}</extra>`,
          }
        : null;
    const missingTrace =
      numericMissingLayout &&
      missingBucket &&
      Number(missingBucket.count) > 0 &&
      Number.isFinite(Number(missingBucket.y_value))
        ? {
            x: [numericMissingLayout.anchor],
            y: [Number(missingBucket.y_value)],
            width: [numericMissingLayout.width],
            type: "bar",
            name: t("shapeFunctions.missing"),
            marker: {
              color: "rgba(244, 63, 94, 0.68)",
              line: { color: "rgba(190, 24, 93, 0.9)", width: 1.2 },
            },
            hovertemplate: `<b>${numericMissingLayout.label}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<br>${t("shapeFunctions.countAxisLabel")}: ${Number(missingBucket.count) || 0}<extra></extra>`,
          }
        : null;
    const comparisonMissingTrace =
      numericMissingLayout &&
      comparisonShapeFunction?.missing_bucket &&
      Number(comparisonShapeFunction.missing_bucket.count) > 0 &&
      Number.isFinite(Number(comparisonShapeFunction.missing_bucket.y_value))
        ? {
            x: [numericMissingLayout.anchor + numericMissingLayout.width * 0.28],
            y: [Number(comparisonShapeFunction.missing_bucket.y_value)],
            width: [numericMissingLayout.width * 0.55],
            type: "bar",
            name: `${t("shapeFunctions.comparison")} ${t("shapeFunctions.missing")}`,
            marker: {
              color: "rgba(249, 115, 22, 0.5)",
              line: { color: "#ea580c", width: 1.1 },
            },
            hovertemplate: `<b>${numericMissingLayout.label}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<br>${t("shapeFunctions.countAxisLabel")}: ${Number(comparisonShapeFunction.missing_bucket.count) || 0}<extra>${t("shapeFunctions.comparison")}</extra>`,
          }
        : null;

    const dragHighlightTraces =
      isDragging && dragXValue !== null && dragCurvePoints.length > 1
        ? buildDragHighlightSegments(
            dragCurvePoints,
            dragXValue,
            getBrushSigma(),
          )
        : [];

    // Small markers at edited positions only
    const showEditMarkers =
      (editedPoints || []).length > 0 &&
      (editedPoints || []).length <
        Math.max(20, Math.floor(x_values.length * 0.5));
    const editMarkerPoints = showEditMarkers
      ? (editedPoints || []).filter((p) => {
          const pxNum = Number(p.x_value);
          const idx = x_values.findIndex(
            (x) => Math.abs(Number(x) - pxNum) < 0.0001,
          );
          if (idx === -1) return true;
          return Math.abs(y_values[idx] - p.y_value) > 0.001;
        })
      : [];

    const editMarkersTrace =
      editMarkerPoints.length > 0
        ? {
            x: editMarkerPoints.map((p) => Number(p.x_value)),
            y: editMarkerPoints.map((p) => p.y_value),
            type: "scatter",
            mode: "markers",
            name: t("shapeFunctions.editPoints"),
            marker: {
              color: "#10b981",
              size: 8,
              symbol: "circle",
              line: { color: "#065f46", width: 1.5 },
            },
            showlegend: false,
            hovertemplate:
              `<b>x: %{x:.3f}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<extra>${t("shapeFunctions.edited")}</extra>`,
          }
        : null;

    data = [
      ...(hasEdits ? [originalTrace] : []),
      ...(missingTrace ? [missingTrace] : []),
      currentTrace,
      ...(comparisonTrace ? [comparisonTrace] : []),
      ...(comparisonMissingTrace ? [comparisonMissingTrace] : []),
      ...dragHighlightTraces,
      ...(editMarkersTrace ? [editMarkersTrace] : []),
    ];
  } else {
    // ---- Categorical bar chart traces (unchanged) ----
    const originalTrace = {
      x: x_values,
      y: y_values,
      type: "bar",
      name: t("shapeFunctions.original"),
      marker: { color: "#9ca3af", size: 4 },
      opacity: 0.6,
      showlegend: hasEdits,
      hoverinfo: "skip",
    };

    const markerColors = currentYValues.map((_, i) => {
      if (i === dragPointIndex) return "#ef4444";
      if (i === hoveredPoint && isEditing) return "#f59e0b";
      return hasEdits ? "#10b981" : "#3b82f6";
    });

    const markerSizes = currentYValues.map((_, i) => {
      if (i === dragPointIndex) return 20;
      if (i === hoveredPoint && isEditing) return 16;
      return isEditing ? 12 : 8;
    });

    const editableTrace = {
      x: x_values,
      y: currentYValues,
      type: "bar",
      name: hasEdits
        ? t("shapeFunctions.edited")
        : t("shapeFunctions.current"),
      customdata: xTickLabels || x_values,
      marker: {
        color: markerColors,
        size: markerSizes,
        symbol: "circle",
        line: { color: "#1e40af", width: isEditing ? 2 : 0 },
      },
      hovertemplate: `<b>%{customdata}</b><br>Effect: %{y:.3f}<extra></extra>`,
    };
    const comparisonTrace =
      comparisonShapeFunction &&
      Array.isArray(comparisonShapeFunction.x_values) &&
      Array.isArray(comparisonShapeFunction.y_values)
        ? {
            x: comparisonShapeFunction.x_values,
            y: comparisonShapeFunction.y_values,
            type: "bar",
            name: t("shapeFunctions.comparison"),
            customdata:
              comparisonShapeFunction.x_tick_labels ||
              comparisonShapeFunction.x_values,
            marker: {
              color: "rgba(249, 115, 22, 0.45)",
              line: { color: "#ea580c", width: 1 },
            },
            hovertemplate: `<b>%{customdata}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<extra>${t("shapeFunctions.comparison")}</extra>`,
          }
        : null;

    data = [
      ...(hasEdits ? [originalTrace] : []),
      editableTrace,
      ...(comparisonTrace ? [comparisonTrace] : []),
    ];
  }

  // No drag crosshair shapes for numeric dragging; highlight is shown directly on the line.
  const layoutShapes = [];
  const layoutAnnotations = [];
  if (isNumeric && numericMissingLayout) {
    layoutShapes.push({
      type: "line",
      x0: numericMissingLayout.dividerX,
      x1: numericMissingLayout.dividerX,
      y0: yRange[0],
      y1: yRange[1],
      line: {
        color: "rgba(148, 163, 184, 0.85)",
        width: 1,
        dash: "dot",
      },
    });
    layoutAnnotations.push({
      x: numericMissingLayout.anchor,
      y: 1.04,
      xref: "x",
      yref: "paper",
      text: t("shapeFunctions.missing"),
      showarrow: false,
      font: { size: enlarged ? 12 : 10, color: "#475569" },
      align: "center",
    });
  }

  const layout = {
    title: {
      text:
        feature_name +
        (isDragging
          ? ` ${t("shapeFunctions.draggingSuffix")}`
          : hasEdits
            ? ` ${t("shapeFunctions.editedSuffix")}`
            : ""),
      font: {
        size: enlarged ? 18 : 14,
        color: isDragging ? "#ef4444" : hasEdits ? "#10b981" : "#374151",
      },
    },
    xaxis: {
      title: { text: feature_name, font: { size: enlarged ? 14 : 12 } },
      gridcolor: "#e5e7eb",
      tickfont: { size: enlarged ? 12 : 10 },
      fixedrange: true,
      ...(!isNumeric && xTickLabels
        ? {
            tickmode: "array",
            tickvals: x_values,
            ticktext: xTickLabels,
            tickangle: -20,
          }
        : {}),
      ...(isNumeric ? { range: xRange } : {}),
    },
    yaxis: {
      title: {
        text: t("shapeFunctions.effect"),
        font: { size: enlarged ? 14 : 12 },
      },
      gridcolor: "#e5e7eb",
      zeroline: true,
      zerolinecolor: "#9ca3af",
      zerolinewidth: 1,
      tickfont: { size: enlarged ? 12 : 10 },
      fixedrange: true,
      range: yRange,
    },
    margin: enlarged
      ? { l: 70, r: 40, t: 50, b: hasEdits ? 90 : 60 }
      : { l: 50, r: 20, t: 40, b: hasEdits ? 78 : 50 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    height: enlarged ? 520 : 280,
    dragmode: false,
    hovermode: "closest",
    showlegend: hasEdits || Boolean(comparisonShapeFunction),
    legend: {
      orientation: "h",
      x: 0,
      xanchor: "left",
      y: -0.24,
      yanchor: "top",
      font: { size: 10 },
      bgcolor: "rgba(255,255,255,0.8)",
    },
    annotations: layoutAnnotations,
    shapes: layoutShapes,
  };

  const config = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: false,
    doubleClick: false,
  };

  const distributionPlot = useMemo(() => {
    if (!hasDistribution) return null;

    if (distribution.chart_type === "numeric") {
      const bins = (distribution.bins || []).filter(
        (bin) =>
          Number.isFinite(Number(bin?.center)) &&
          Number.isFinite(Number(bin?.count)) &&
          Number.isFinite(Number(bin?.x0)) &&
          Number.isFinite(Number(bin?.x1)),
      );
      if (bins.length === 0) return null;

      const numericRange = isNumeric
        ? xRange
        : [
            Math.min(...bins.map((bin) => Number(bin.x0))),
            Math.max(...bins.map((bin) => Number(bin.x1))),
          ];
      const missingCount = Number(distribution.missing_count) || 0;
      const distributionMissingTrace =
        numericMissingLayout && missingCount > 0
          ? {
              x: [numericMissingLayout.anchor],
              y: [missingCount],
              width: [numericMissingLayout.width],
              type: "bar",
              marker: {
                color: "rgba(244, 63, 94, 0.58)",
                line: { color: "rgba(190, 24, 93, 0.82)", width: 1 },
              },
              hovertemplate: `<b>${t("shapeFunctions.missing")}</b><br>${t("shapeFunctions.countAxisLabel")}: %{y}<extra></extra>`,
            }
          : null;

      return {
        data: [
          ...(distributionMissingTrace ? [distributionMissingTrace] : []),
          {
            x: bins.map((bin) => Number(bin.center)),
            y: bins.map((bin) => Number(bin.count)),
            width: bins.map((bin) =>
              Math.max(Number(bin.x1) - Number(bin.x0), 0.001),
            ),
            customdata: bins.map((bin) => [Number(bin.x0), Number(bin.x1)]),
            type: "bar",
            marker: {
              color: "rgba(148, 163, 184, 0.65)",
              line: { color: "rgba(100, 116, 139, 0.45)", width: 1 },
            },
            hovertemplate: `<b>x</b>: %{customdata[0]:.3f} - %{customdata[1]:.3f}<br>${t("shapeFunctions.countAxisLabel")}: %{y}<extra></extra>`,
          },
        ],
        layout: {
          margin: enlarged
            ? { l: 56, r: 20, t: 10, b: 42 }
            : { l: 46, r: 12, t: 8, b: 36 },
          height: enlarged ? 210 : 150,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(255,255,255,0.65)",
          bargap: 0,
          showlegend: false,
          hovermode: "closest",
          dragmode: false,
          xaxis: {
            range: numericRange,
            fixedrange: true,
            gridcolor: "#e2e8f0",
            zeroline: false,
            tickfont: { size: enlarged ? 11 : 10 },
          },
          yaxis: {
            title: {
              text: t("shapeFunctions.countAxisLabel"),
              font: { size: enlarged ? 12 : 11 },
            },
            fixedrange: true,
            gridcolor: "#e2e8f0",
            rangemode: "tozero",
            tickfont: { size: enlarged ? 11 : 10 },
          },
          annotations:
            distributionMissingTrace && numericMissingLayout
              ? [
                  {
                    x: numericMissingLayout.anchor,
                    y: 1.05,
                    xref: "x",
                    yref: "paper",
                    text: t("shapeFunctions.missing"),
                    showarrow: false,
                    font: { size: enlarged ? 11 : 10, color: "#475569" },
                  },
                ]
              : [],
          shapes:
            distributionMissingTrace && numericMissingLayout
              ? [
                  {
                    type: "line",
                    x0: numericMissingLayout.dividerX,
                    x1: numericMissingLayout.dividerX,
                    y0: 0,
                    y1: 1,
                    xref: "x",
                    yref: "paper",
                    line: {
                      color: "rgba(148, 163, 184, 0.85)",
                      width: 1,
                      dash: "dot",
                    },
                  },
                ]
              : [],
        },
      };
    }

    if (distribution.chart_type === "categorical") {
      const counts = (distribution.counts || []).filter(
        (entry) => entry && entry.x_value !== undefined && entry.label !== undefined,
      );
      if (counts.length === 0) return null;

      const categoryValues = counts.map((entry) => String(entry.x_value));
      const categoryLabels = counts.map((entry) => String(entry.label));

      return {
        data: [
          {
            x: categoryValues,
            y: counts.map((entry) => Number(entry.count) || 0),
            customdata: categoryLabels,
            type: "bar",
            marker: {
              color: "rgba(148, 163, 184, 0.72)",
              line: { color: "rgba(100, 116, 139, 0.5)", width: 1 },
            },
            hovertemplate: `<b>%{customdata}</b><br>${t("shapeFunctions.countAxisLabel")}: %{y}<extra></extra>`,
          },
        ],
        layout: {
          margin: enlarged
            ? { l: 56, r: 20, t: 10, b: 54 }
            : { l: 46, r: 12, t: 8, b: 48 },
          height: enlarged ? 210 : 150,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(255,255,255,0.65)",
          bargap: 0.2,
          showlegend: false,
          hovermode: "closest",
          dragmode: false,
          xaxis: {
            fixedrange: true,
            gridcolor: "#e2e8f0",
            categoryorder: "array",
            categoryarray: categoryValues,
            tickmode: "array",
            tickvals: categoryValues,
            ticktext: categoryLabels,
            tickangle: -20,
            tickfont: { size: enlarged ? 11 : 10 },
          },
          yaxis: {
            title: {
              text: t("shapeFunctions.countAxisLabel"),
              font: { size: enlarged ? 12 : 11 },
            },
            fixedrange: true,
            gridcolor: "#e2e8f0",
            rangemode: "tozero",
            tickfont: { size: enlarged ? 11 : 10 },
          },
        },
      };
    }

    return null;
  }, [
    distribution,
    hasDistribution,
    isNumeric,
    xRange,
    enlarged,
    numericMissingLayout,
    t,
  ]);

  const distributionConfig = useMemo(
    () => ({
      responsive: true,
      displayModeBar: false,
      scrollZoom: false,
      doubleClick: false,
      staticPlot: false,
    }),
    [],
  );
  const distributionCountSummary = useMemo(
    () => getDistributionCountSummary(distribution),
    [distribution],
  );

  // Determine cursor style
  let cursorStyle = "default";
  if (isEditing) {
    if (isDragging) {
      cursorStyle = "grabbing";
    } else if (!isNumeric && hoveredPoint !== null) {
      cursorStyle = "grab";
    } else {
      cursorStyle = "crosshair";
    }
  }

  return (
    <div
      ref={containerRef}
      className={`bg-white rounded-lg border overflow-hidden transition-all select-none ${
        isDragging
          ? "border-red-400 ring-2 ring-red-200"
          : hasEdits
            ? "border-green-300 ring-2 ring-green-100"
            : isEditing
              ? "border-blue-300 ring-2 ring-blue-100"
              : "border-gray-200"
      }`}
      style={{ userSelect: "none" }}
    >
      {/* Precise Value Entry Modal */}
      {preciseEntry && (
        <div
          className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50"
          onClick={() => setPreciseEntry(null)}
        >
          <div
            className="bg-white rounded-xl shadow-2xl p-5 max-w-sm w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h4 className="text-base font-semibold text-gray-800 mb-1">
              {t("shapeFunctions.setEffectValue")}
            </h4>
            <p className="text-sm text-gray-500 mb-4">
              {feature_name} = {preciseEntry.displayX}
            </p>
            <input
              type="number"
              step="any"
              value={preciseValue}
              onChange={(e) => setPreciseValue(e.target.value)}
              autoFocus
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-4"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !isNaN(parseFloat(preciseValue))) {
                  if (isNumeric) {
                    applyPreciseNumericEdit(
                      Number(preciseEntry.xValue),
                      parseFloat(preciseValue),
                    );
                  } else {
                    onPointEdit(
                      feature_name,
                      preciseEntry.xValue,
                      parseFloat(preciseValue),
                      feature_type,
                    );
                  }
                  setPreciseEntry(null);
                } else if (e.key === "Escape") {
                  setPreciseEntry(null);
                }
              }}
            />
            <div className="flex gap-3">
              <button
                onClick={() => setPreciseEntry(null)}
                className="flex-1 px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              >
                {t("common.cancel")}
              </button>
              <button
                onClick={() => {
                  if (!isNaN(parseFloat(preciseValue))) {
                    if (isNumeric) {
                      applyPreciseNumericEdit(
                        Number(preciseEntry.xValue),
                        parseFloat(preciseValue),
                      );
                    } else {
                      onPointEdit(
                        feature_name,
                        preciseEntry.xValue,
                        parseFloat(preciseValue),
                        feature_type,
                      );
                    }
                    setPreciseEntry(null);
                  }
                }}
                disabled={isNaN(parseFloat(preciseValue))}
                className="flex-1 px-4 py-2 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {t("common.apply")}
              </button>
            </div>
          </div>
        </div>
      )}
      {isEditing && (
        <div
          className={`px-3 py-1.5 text-xs border-b flex items-center gap-2 transition-colors ${
            isDragging
              ? "bg-red-100 text-red-700 border-red-200"
              : "bg-blue-50 text-blue-700 border-blue-200"
          }`}
        >
          {isDragging ? (
            isNumeric ? (
              <>
                <span className="font-bold animate-pulse">
                  {t("shapeFunctions.brushingAtX", {
                    value:
                      dragXValue != null ? Number(dragXValue).toFixed(3) : "",
                  })}
                </span>
                <span className="text-red-500">|</span>
                <span>
                  {t("shapeFunctions.releaseToApply", {
                    value: dragYValue != null ? dragYValue.toFixed(3) : "",
                  })}
                </span>
              </>
            ) : (
              <>
                <span className="font-bold animate-pulse">
                  {t("shapeFunctions.draggingPoint", {
                    index: dragPointIndex + 1,
                  })}
                </span>
                <span className="text-red-500">|</span>
                <span>
                  {t("shapeFunctions.releaseToSetValue", {
                    value: localYValues?.[dragPointIndex]?.toFixed(2),
                  })}
                </span>
              </>
            )
          ) : isNumeric ? (
            <>
              <span className="font-medium">
                {t("shapeFunctions.clickDragHint")}
              </span>
            </>
          ) : hoveredPoint !== null ? (
            <>
              <span className="font-medium">
                {t("shapeFunctions.pointSelected", {
                  index: hoveredPoint + 1,
                })}
              </span>
              <span className="text-blue-400">|</span>
              <span>{t("shapeFunctions.preciseValueHint")}</span>
            </>
          ) : (
            <>
              <span className="font-medium">
                {t("shapeFunctions.hoverSelectHint")}
              </span>
            </>
          )}
        </div>
      )}
      <Plot
        ref={plotRef}
        data={data}
        layout={layout}
        config={config}
        onHover={!isNumeric ? handleHover : undefined}
        onUnhover={!isNumeric ? handleUnhover : undefined}
        onInitialized={updatePlotBounds}
        onUpdate={updatePlotBounds}
        style={{
          cursor: cursorStyle,
          userSelect: "none",
          WebkitUserSelect: "none",
        }}
        className="w-full"
      />
      {hasDistribution && (
        <div className="border-t border-slate-200 bg-slate-50">
          <button
            type="button"
            onClick={() => setIsDistributionOpen((open) => !open)}
            className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition-colors hover:bg-slate-100"
          >
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-700">
                {t("shapeFunctions.dataDistribution")}
              </div>
              <div className="text-xs text-slate-500">
                {isDistributionOpen
                  ? t("shapeFunctions.hideDistribution")
                  : t("shapeFunctions.showDistribution")}
              </div>
            </div>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`shrink-0 text-slate-500 transition-transform ${
                isDistributionOpen ? "rotate-180" : ""
              }`}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
          {isDistributionOpen && (
            <div className="border-t border-slate-200 px-3 py-3">
              {distributionPlot ? (
                <>
                  <Plot
                    data={distributionPlot.data}
                    layout={distributionPlot.layout}
                    config={distributionConfig}
                    className="w-full"
                  />
                  {distributionCountSummary?.includesMissing ? (
                    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                      <p>
                        {t("shapeFunctions.distributionAnsweredCount", {
                          count: distributionCountSummary.answeredCount,
                        })}
                      </p>
                      <p>
                        {t("shapeFunctions.distributionMissingCount", {
                          count: distributionCountSummary.missingCount,
                        })}
                      </p>
                      <p>
                        {t("shapeFunctions.distributionRepresentedCount", {
                          count: distributionCountSummary.totalCount,
                        })}
                      </p>
                    </div>
                  ) : (
                    <p className="mt-2 text-xs text-slate-500">
                      {t("shapeFunctions.distributionTotalCount", {
                        count: distributionCountSummary?.totalCount || 0,
                      })}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-sm text-slate-500">
                  {t("shapeFunctions.distributionEmpty")}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Sureness Rating Modal Component
const SurenessModal = ({ isOpen, onClose, onConfirm, featureName, t }) => {
  const [sureness, setSureness] = useState(5);
  const [message, setMessage] = useState("");

  if (!isOpen) return null;

  const handleConfirm = () => {
    if (!message.trim()) return; // Require message
    onConfirm(sureness / 10, message.trim()); // Convert 1-10 to 0.1-1.0
    setSureness(5); // Reset for next time
    setMessage("");
  };

  const handleClose = () => {
    setSureness(5);
    setMessage("");
    onClose();
  };

  const isValid = message.trim().length > 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          {t("shapeFunctions.submitEditTitle", { feature: featureName })}
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          {t("shapeFunctions.submitEditDescription")}
        </p>

        {/* Confidence Slider */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t("shapeFunctions.confidenceLevel")}
          </label>
          <div className="flex justify-between text-sm text-gray-500 mb-2">
            <span>{t("shapeFunctions.notSure")}</span>
            <span className="font-bold text-lg text-blue-600">{sureness}</span>
            <span>{t("shapeFunctions.verySure")}</span>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            value={sureness}
            onChange={(e) => setSureness(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
            style={{
              background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${
                (sureness - 1) * 11.11
              }%, #e5e7eb ${(sureness - 1) * 11.11}%, #e5e7eb 100%)`,
            }}
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>1</span>
            <span>5</span>
            <span>10</span>
          </div>
        </div>

        {/* Commit Message */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            {t("shapeFunctions.editDescription")}{" "}
            <span className="text-red-500">*</span>
          </label>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={t("shapeFunctions.editDescriptionPlaceholder")}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm resize-none"
            rows={3}
          />
          {!isValid && message.length > 0 && (
            <p className="text-xs text-red-500 mt-1">
              {t("shapeFunctions.editDescriptionRequired")}
            </p>
          )}
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleClose}
            className="flex-1 px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
          >
            {t("common.cancel")}
          </button>
          <button
            onClick={handleConfirm}
            disabled={!isValid}
            className={`flex-1 px-4 py-2 text-sm rounded-lg transition-colors font-medium ${
              isValid
                ? "bg-green-500 hover:bg-green-600 text-white"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }`}
          >
            {t("shapeFunctions.submitEdit")}
          </button>
        </div>
      </div>
    </div>
  );
};

const FeatureChartSettingsModal = ({
  isOpen,
  shapeFunction,
  onClose,
  onSave,
  t,
}) => {
  const [treatAsCategorical, setTreatAsCategorical] = useState(false);
  const [valueLabels, setValueLabels] = useState({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const chartConfig = shapeFunction?.chart_config || {};
  const featureName = shapeFunction?.feature_name || "";
  const baseFeatureType =
    chartConfig.base_feature_type ||
    (shapeFunction?.feature_type === "categorical" ? "categorical" : "numeric");
  const canBeCategorical =
    Boolean(chartConfig.can_be_categorical) ||
    baseFeatureType === "categorical";

  const availableValues = useMemo(() => {
    const configured = Array.isArray(chartConfig.available_values)
      ? chartConfig.available_values.map((v) => String(v))
      : [];
    if (configured.length > 0) return configured;
    const fallback = Array.isArray(shapeFunction?.x_values)
      ? shapeFunction.x_values.map((v) => String(v))
      : [];
    return fallback;
  }, [chartConfig.available_values, shapeFunction]);

  useEffect(() => {
    if (!isOpen || !shapeFunction) return;
    setTreatAsCategorical(
      baseFeatureType === "numeric"
        ? Boolean(chartConfig.treat_as_categorical)
        : false,
    );
    setValueLabels({ ...(chartConfig.value_labels || {}) });
    setSaving(false);
    setError(null);
  }, [
    isOpen,
    shapeFunction,
    chartConfig.treat_as_categorical,
    chartConfig.value_labels,
    baseFeatureType,
  ]);

  if (!isOpen || !shapeFunction) return null;

  const effectiveCategorical =
    baseFeatureType === "categorical" || treatAsCategorical;

  const handleSubmit = async () => {
    try {
      setSaving(true);
      setError(null);

      const normalizedLabels = {};
      if (effectiveCategorical) {
        availableValues.forEach((rawValue) => {
          const rawKey = String(rawValue);
          const nextLabel = String(valueLabels[rawKey] ?? "").trim();
          if (nextLabel && nextLabel !== rawKey) {
            normalizedLabels[rawKey] = nextLabel;
          }
        });
      }

      await onSave(featureName, {
        treat_as_categorical:
          baseFeatureType === "numeric" ? Boolean(treatAsCategorical) : false,
        value_labels: normalizedLabels,
      });
      onClose();
    } catch (err) {
      setError(err?.message || t("shapeFunctions.chartSettingsSaveError"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-800">
            {t("shapeFunctions.chartMappingTitle", { feature: featureName })}
          </h3>
          <p className="text-sm text-gray-500 mt-1">
            {t("shapeFunctions.chartMappingDescription")}
          </p>
        </div>

        <div className="px-5 py-4 space-y-4">
          {baseFeatureType === "numeric" && (
            <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
              <input
                type="checkbox"
                checked={treatAsCategorical}
                disabled={!canBeCategorical || saving}
                onChange={(e) => setTreatAsCategorical(e.target.checked)}
                className="mt-1 h-4 w-4"
              />
              <div>
                <div className="text-sm font-medium text-slate-700">
                  {t("shapeFunctions.treatAsCategorical")}
                </div>
                <div className="text-xs text-slate-500">
                  {t("shapeFunctions.treatAsCategoricalHint")}
                </div>
              </div>
            </label>
          )}

          {!canBeCategorical && baseFeatureType === "numeric" && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">
              {t("shapeFunctions.cannotConvertCategorical")}
            </div>
          )}

          {effectiveCategorical && (
            <div className="rounded-lg border border-gray-200">
              <div className="px-3 py-2 border-b border-gray-200 bg-gray-50 text-sm font-medium text-gray-700">
                {t("shapeFunctions.xAxisValueLabels")}
              </div>
              <div className="max-h-72 overflow-auto">
                {availableValues.length === 0 ? (
                  <div className="px-3 py-3 text-sm text-gray-500">
                    {t("shapeFunctions.noCategoricalValues")}
                  </div>
                ) : (
                  <div className="divide-y divide-gray-100">
                    {availableValues.map((rawValue) => {
                      const key = String(rawValue);
                      return (
                        <div
                          key={key}
                          className="grid grid-cols-2 gap-3 px-3 py-2 items-center"
                        >
                          <div className="text-sm text-slate-700 font-mono">
                            {key}
                          </div>
                          <input
                            type="text"
                            value={valueLabels[key] ?? ""}
                            onChange={(e) =>
                              setValueLabels((prev) => ({
                                ...prev,
                                [key]: e.target.value,
                              }))
                            }
                            placeholder={t("shapeFunctions.displayLabelFor", {
                              value: key,
                            })}
                            disabled={saving}
                            className="w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm text-slate-800 focus:border-primary-500 focus:ring-2 focus:ring-primary-500"
                          />
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          {!effectiveCategorical && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
              {t("shapeFunctions.mappingAvailableAfterCategorical")}
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}
        </div>

        <div className="px-5 py-3 border-t border-gray-200 flex items-center justify-end gap-2">
          <button
            onClick={onClose}
            disabled={saving}
            className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors disabled:opacity-50"
          >
            {t("common.cancel")}
          </button>
          <button
            onClick={handleSubmit}
            disabled={saving}
            className="px-4 py-2 text-sm bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors font-medium disabled:opacity-50"
          >
            {saving ? t("shapeFunctions.saving") : t("shapeFunctions.saveMapping")}
          </button>
        </div>
      </div>
    </div>
  );
};

// Individual Feature Card with its own submit button
const FeatureEditCard = ({
  shapeFunction,
  comparisonShapeFunction = null,
  featureSchemaEntry = null,
  editedPoints,
  unsavedEditedPoints,
  onPointEdit,
  onFeatureSubmit,
  onFeatureReset,
  onOpenChartSettings,
  onOpenDetails,
  onCycleChartType,
  isEditing,
  brushHardness = 50,
  onBrushHardnessChange = null,
  hasSavedEdits,
  isSuperadmin = false,
  sharedYRange = null,
  t,
}) => {
  const [isEnlarged, setIsEnlarged] = useState(false);
  const [isCyclingChartType, setIsCyclingChartType] = useState(false);

  // Show Submit only when there are genuinely new unsaved edits
  const hasUnsavedEdits = unsavedEditedPoints && unsavedEditedPoints.length > 0;
  const chartConfig = shapeFunction?.chart_config || {};
  const baseFeatureType =
    chartConfig.base_feature_type ||
    (shapeFunction?.feature_type === "categorical" ? "categorical" : "numeric");
  const currentChartType =
    chartConfig.chart_feature_type ||
    (shapeFunction?.feature_type === "categorical" ? "categorical" : "numeric");
  const canToggleToCategorical =
    currentChartType === "numeric" && Boolean(chartConfig.can_be_categorical);
  const canToggleToNumeric =
    currentChartType === "categorical" && Boolean(chartConfig.can_be_numeric);
  const canCycleChartType = canToggleToCategorical || canToggleToNumeric;
  const canConfigureCategoricalChart = currentChartType === "categorical";
  const hasFeatureDetails = Boolean(featureSchemaEntry);

  const handleCycleChartType = useCallback(async () => {
    if (!onCycleChartType || !canCycleChartType || isCyclingChartType) return;
    setIsCyclingChartType(true);
    try {
      let payload;
      if (currentChartType === "categorical") {
        payload =
          baseFeatureType === "categorical"
            ? {
                treat_as_numeric: true,
              }
            : {
                treat_as_categorical: false,
              };
      } else {
        payload =
          baseFeatureType === "numeric"
            ? {
                treat_as_categorical: true,
              }
            : {
                treat_as_numeric: false,
              };
      }
      await onCycleChartType(shapeFunction.feature_name, payload);
    } finally {
      setIsCyclingChartType(false);
    }
  }, [
    onCycleChartType,
    canCycleChartType,
    isCyclingChartType,
    currentChartType,
    baseFeatureType,
    shapeFunction?.feature_name,
  ]);

  return (
    <div className="relative">
      {isSuperadmin && (
        <button
          onClick={handleCycleChartType}
          title={
            canCycleChartType
              ? currentChartType === "categorical"
                ? t("shapeFunctions.switchChartToNumeric")
                : t("shapeFunctions.switchChartToCategorical")
              : t("shapeFunctions.chartTypeUnavailable")
          }
          disabled={!canCycleChartType || isCyclingChartType}
          className={`absolute top-2 left-2 z-10 p-1.5 border rounded-md shadow-sm transition-colors ${
            canCycleChartType
              ? "bg-white/90 hover:bg-gray-100 border-gray-300 text-gray-700"
              : "bg-gray-100/90 border-gray-200 text-gray-400 cursor-not-allowed"
          }`}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={isCyclingChartType ? "animate-spin" : ""}
          >
            <path d="M3 2v6h6" />
            <path d="M21 22v-6h-6" />
            <path d="M21 11A8 8 0 0 0 7.5 5.5L3 8" />
            <path d="M3 13a8 8 0 0 0 13.5 5.5L21 16" />
          </svg>
        </button>
      )}
      {isSuperadmin && canConfigureCategoricalChart && (
        <button
          onClick={() => onOpenChartSettings(shapeFunction)}
          title={t("shapeFunctions.editCategoricalMapping")}
          className="absolute top-2 left-10 z-10 h-7 px-2 bg-white/90 hover:bg-gray-100 border border-gray-300 rounded-md shadow-sm transition-colors text-xs text-gray-700 font-medium"
        >
          {t("shapeFunctions.chartMappingButton")}
        </button>
      )}
      <button
        type="button"
        onClick={() => onOpenDetails(shapeFunction.feature_name)}
        title={t("shapeFunctions.openFeatureDetails")}
        disabled={!hasFeatureDetails}
        className={`absolute top-2 right-11 z-10 h-7 px-2 border rounded-md shadow-sm transition-colors text-xs font-medium ${
          hasFeatureDetails
            ? "bg-white/90 hover:bg-gray-100 border-gray-300 text-gray-700"
            : "bg-gray-100/90 border-gray-200 text-gray-400 cursor-not-allowed"
        }`}
      >
        {t("shapeFunctions.detailsButton")}
      </button>
      {/* Enlarge button */}
      <button
        onClick={() => setIsEnlarged(true)}
        title={t("shapeFunctions.enlargeChart")}
        className="absolute top-2 right-2 z-10 p-1.5 bg-white/80 hover:bg-gray-100 border border-gray-300 rounded-md shadow-sm transition-colors text-gray-500 hover:text-gray-700"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="15 3 21 3 21 9" />
          <polyline points="9 21 3 21 3 15" />
          <line x1="21" y1="3" x2="14" y2="10" />
          <line x1="3" y1="21" x2="10" y2="14" />
        </svg>
      </button>

      <EditableShapeFunctionChart
        shapeFunction={shapeFunction}
        comparisonShapeFunction={comparisonShapeFunction}
        editedPoints={editedPoints}
        onPointEdit={onPointEdit}
        isEditing={isEditing}
        brushHardness={brushHardness}
        sharedYRange={sharedYRange}
        t={t}
      />
      <div className="mt-2 flex justify-between items-center">
        {/* Reset button - show if there are saved or unsaved edits */}
        {(hasSavedEdits || hasUnsavedEdits) && (
          <button
            onClick={() => onFeatureReset(shapeFunction.feature_name)}
            className="px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors font-medium flex items-center gap-1"
          >
            <span>✕</span>
            <span>{t("common.reset")}</span>
          </button>
        )}
        {!(hasSavedEdits || hasUnsavedEdits) && <div />}

        {/* Submit button - only show if there are unsaved edits */}
        {hasUnsavedEdits && (
          <button
            onClick={() => onFeatureSubmit(shapeFunction.feature_name)}
            className="px-3 py-1.5 text-sm bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors font-medium shadow-md flex items-center gap-1"
          >
            <span>✓</span>
            <span>
              {t("shapeFunctions.submitFeature", {
                feature: shapeFunction.feature_name,
              })}
            </span>
          </button>
        )}
      </div>

      {/* Enlarged Chart Modal */}
      {isEnlarged &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-6"
            onClick={() => setIsEnlarged(false)}
          >
            <div
              className="max-h-[90vh] w-full max-w-5xl overflow-auto rounded-2xl bg-white shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Modal header */}
              <div className="flex items-center justify-between border-b border-gray-200 px-5 py-3">
                <h3 className="text-lg font-semibold text-gray-800">
                  {shapeFunction.feature_name}
                </h3>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onOpenDetails(shapeFunction.feature_name)}
                    disabled={!hasFeatureDetails}
                    className={`h-9 rounded-lg border px-3 text-sm font-medium transition-colors ${
                      hasFeatureDetails
                        ? "border-slate-300 text-slate-700 hover:bg-slate-100"
                        : "border-slate-200 text-slate-400 cursor-not-allowed"
                    }`}
                  >
                    {t("shapeFunctions.detailsButton")}
                  </button>
                  <button
                    onClick={() => setIsEnlarged(false)}
                    className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors text-gray-500 hover:text-gray-700"
                    title={t("common.close")}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="20"
                      height="20"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
              </div>
              {/* Enlarged chart */}
              <div className="p-4">
                {isEditing &&
                  shapeFunction?.feature_type === "numeric" &&
                  onBrushHardnessChange && (
                    <BrushHardnessControl
                      brushHardness={brushHardness}
                      onChange={onBrushHardnessChange}
                      t={t}
                      className="mb-4"
                    />
                  )}
                <EditableShapeFunctionChart
                  shapeFunction={shapeFunction}
                  comparisonShapeFunction={comparisonShapeFunction}
                  editedPoints={editedPoints}
                  onPointEdit={onPointEdit}
                  isEditing={isEditing}
                  brushHardness={brushHardness}
                  enlarged
                  sharedYRange={sharedYRange}
                  t={t}
                />
              </div>
              {/* Modal footer actions */}
              <div className="flex items-center justify-between border-t border-gray-200 px-5 py-3">
                {hasSavedEdits || hasUnsavedEdits ? (
                  <button
                    onClick={() => onFeatureReset(shapeFunction.feature_name)}
                    className="px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors font-medium flex items-center gap-1"
                  >
                    <span>✕</span>
                    <span>{t("common.reset")}</span>
                  </button>
                ) : (
                  <div />
                )}
                <div className="flex items-center gap-2">
                  {hasUnsavedEdits && (
                    <button
                      onClick={() => {
                        onFeatureSubmit(shapeFunction.feature_name);
                        setIsEnlarged(false);
                      }}
                      className="px-4 py-2 text-sm bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors font-medium shadow-md flex items-center gap-1"
                    >
                      <span>✓</span>
                      <span>
                        {t("shapeFunctions.submitFeature", {
                          feature: shapeFunction.feature_name,
                        })}
                      </span>
                    </button>
                  )}
                  <button
                    onClick={() => setIsEnlarged(false)}
                    className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                  >
                    {t("common.close")}
                  </button>
                </div>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
};

const EditableShapeFunctionsGrid = ({
  shapeFunctions,
  comparisonShapeFunctions = [],
  featureSchema = [],
  loading,
  onShapeFunctionsEdit,
  onReset,
  onFeatureReset,
  initialEditedPoints = {},
  onUnsavedEditsChange,
  isSuperadmin = false,
  onUpdateFeatureChartSettings,
  language = "en",
}) => {
  const t = createTranslator(language);
  const [isEditing, setIsEditing] = useState(false);
  const [editedPoints, setEditedPoints] = useState({});
  const [hasChanges, setHasChanges] = useState(false);
  const [showSurenessModal, setShowSurenessModal] = useState(false);
  const [pendingFeatureSubmit, setPendingFeatureSubmit] = useState(null);
  const [syncAxes, setSyncAxes] = useState(false);
  const [brushHardness, setBrushHardness] = useState(50);
  const [chartSettingsFeature, setChartSettingsFeature] = useState(null);
  const [detailsFeatureName, setDetailsFeatureName] = useState(null);
  const comparisonShapeFunctionMap = useMemo(
    () =>
      new Map(
        (comparisonShapeFunctions || []).map((sf) => [sf.feature_name, sf]),
      ),
    [comparisonShapeFunctions],
  );
  const featureSchemaMap = useMemo(
    () =>
      new Map(
        (featureSchema || [])
          .filter((entry) => entry && entry.name)
          .map((entry) => [entry.name, entry]),
      ),
    [featureSchema],
  );
  const hasNumericCharts = useMemo(
    () =>
      Array.isArray(shapeFunctions) &&
      shapeFunctions.some((sf) => sf?.feature_type === "numeric"),
    [shapeFunctions],
  );
  const activeFeatureSchemaEntry = detailsFeatureName
    ? featureSchemaMap.get(detailsFeatureName) || null
    : null;

  // Compute global y-range across all features when syncAxes is enabled
  const globalYRange = useMemo(() => {
    if (!syncAxes || !shapeFunctions || shapeFunctions.length === 0)
      return null;
    const allY = [];
    shapeFunctions.forEach((sf) => {
      allY.push(...sf.y_values);
      if (Number.isFinite(Number(sf?.missing_bucket?.y_value))) {
        allY.push(Number(sf.missing_bucket.y_value));
      }
      (initialEditedPoints[sf.feature_name] || []).forEach((p) =>
        allY.push(p.y_value),
      );
      (editedPoints[sf.feature_name] || []).forEach((p) =>
        allY.push(p.y_value),
      );
    });
    (comparisonShapeFunctions || []).forEach((sf) => {
      if (Number.isFinite(Number(sf?.missing_bucket?.y_value))) {
        allY.push(Number(sf.missing_bucket.y_value));
      }
    });
    if (allY.length === 0) return null;
    const yMin = Math.min(...allY);
    const yMax = Math.max(...allY);
    const yPadding = Math.max((yMax - yMin) * 0.3, 5);
    return [yMin - yPadding, yMax + yPadding];
  }, [
    syncAxes,
    shapeFunctions,
    comparisonShapeFunctions,
    initialEditedPoints,
    editedPoints,
  ]);

  // Only reset unsaved edits when a completely new model is trained
  useEffect(() => {
    setEditedPoints({});
    setHasChanges(false);
  }, [shapeFunctions]);

  // Notify parent about unsaved edits so comparison chart can preview instantly.
  useEffect(() => {
    if (onUnsavedEditsChange) {
      onUnsavedEditsChange(editedPoints);
    }
  }, [editedPoints, onUnsavedEditsChange]);

  useEffect(() => {
    if (detailsFeatureName && !featureSchemaMap.has(detailsFeatureName)) {
      setDetailsFeatureName(null);
    }
  }, [detailsFeatureName, featureSchemaMap]);

  // Merge saved (initialEditedPoints) with unsaved (editedPoints) for chart display
  const getMergedEditedPoints = useCallback(
    (featureName) => {
      const saved = initialEditedPoints[featureName] || [];
      const unsaved = editedPoints[featureName] || [];
      if (saved.length === 0 && unsaved.length === 0) return [];
      const merged = new Map();
      saved.forEach((p) => merged.set(String(p.x_value), p));
      unsaved.forEach((p) => merged.set(String(p.x_value), p));
      return Array.from(merged.values());
    },
    [initialEditedPoints, editedPoints],
  );

  const handlePointEdit = useCallback(
    (featureName, xValueOrPoints, yValue, featureType) => {
      setEditedPoints((prev) => {
        if (featureType === "numeric") {
          if (Array.isArray(xValueOrPoints)) {
            const normalized = normalizeNumericPoints(xValueOrPoints).map(
              (point) => ({
                x_value: point.x,
                y_value: point.y,
              }),
            );
            return { ...prev, [featureName]: normalized };
          }

          const featurePoints = prev[featureName] || [];
          const existingIndex = featurePoints.findIndex(
            (p) => String(p.x_value) === String(xValueOrPoints),
          );
          if (existingIndex >= 0) {
            const next = [...featurePoints];
            next[existingIndex] = { x_value: xValueOrPoints, y_value: yValue };
            return { ...prev, [featureName]: next };
          }
          return {
            ...prev,
            [featureName]: [
              ...featurePoints,
              { x_value: xValueOrPoints, y_value: yValue },
            ],
          };
        }

        const featurePoints = prev[featureName] || [];
        const existingIndex = featurePoints.findIndex(
          (p) => String(p.x_value) === String(xValueOrPoints),
        );

        let newPoints;
        if (existingIndex >= 0) {
          newPoints = [...featurePoints];
          newPoints[existingIndex] = {
            x_value: xValueOrPoints,
            y_value: yValue,
          };
        } else {
          newPoints = [
            ...featurePoints,
            { x_value: xValueOrPoints, y_value: yValue },
          ];
        }

        return { ...prev, [featureName]: newPoints };
      });
      setHasChanges(true);
    },
    [],
  );

  // Called when user clicks Submit on a specific feature
  const handleFeatureSubmit = useCallback((featureName) => {
    setPendingFeatureSubmit(featureName);
    setShowSurenessModal(true);
  }, []);

  // Called when user confirms sureness rating
  const handleSurenessConfirm = useCallback(
    (weight, message) => {
      if (!pendingFeatureSubmit) return;

      const featureName = pendingFeatureSubmit;
      const points = editedPoints[featureName] || [];
      const sf = shapeFunctions.find((s) => s.feature_name === featureName);

      if (points.length > 0 && sf) {
        // Add weight and message to each point
        const pointsWithWeight = points.map((p) => ({
          ...p,
          weight: weight,
          message: message,
        }));

        const editedShapeFunction = {
          feature_name: featureName,
          feature_type: sf.feature_type || "numeric",
          edited_points: pointsWithWeight,
        };

        onShapeFunctionsEdit([editedShapeFunction]);

        // Clear the edits for this feature after submitting
        setEditedPoints((prev) => {
          const newEdits = { ...prev };
          delete newEdits[featureName];
          return newEdits;
        });

        // Check if there are any remaining changes
        setHasChanges(
          Object.keys(editedPoints).filter((k) => k !== featureName).length > 0,
        );
      }

      setShowSurenessModal(false);
      setPendingFeatureSubmit(null);
    },
    [pendingFeatureSubmit, editedPoints, shapeFunctions, onShapeFunctionsEdit],
  );

  const handleSurenessModalClose = useCallback(() => {
    setShowSurenessModal(false);
    setPendingFeatureSubmit(null);
  }, []);

  const handleReset = useCallback(() => {
    setEditedPoints({});
    setHasChanges(false);
    if (onReset) {
      onReset();
    }
  }, [onReset]);

  const handleFeatureReset = useCallback(
    (featureName) => {
      // Clear local unsaved edits for this feature as part of reset.
      setEditedPoints((prev) => {
        if (!prev[featureName]) {
          return prev;
        }

        const next = { ...prev };
        delete next[featureName];
        setHasChanges(Object.keys(next).length > 0);
        return next;
      });

      if (onFeatureReset) {
        onFeatureReset(featureName);
      }
    },
    [onFeatureReset],
  );

  const handleOpenChartSettings = useCallback((shapeFunction) => {
    setChartSettingsFeature(shapeFunction);
  }, []);

  const handleCloseChartSettings = useCallback(() => {
    setChartSettingsFeature(null);
  }, []);

  const handleOpenDetails = useCallback((featureName) => {
    setDetailsFeatureName(featureName);
  }, []);

  const handleCloseDetails = useCallback(() => {
    setDetailsFeatureName(null);
  }, []);

  const handleSaveChartSettings = useCallback(
    async (featureName, payload) => {
      if (!onUpdateFeatureChartSettings) return;
      await onUpdateFeatureChartSettings(featureName, payload);
    },
    [onUpdateFeatureChartSettings],
  );

  const handleCycleChartType = useCallback(
    async (featureName, payload) => {
      if (!onUpdateFeatureChartSettings) return;
      await onUpdateFeatureChartSettings(featureName, payload);
    },
    [onUpdateFeatureChartSettings],
  );

  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          {t("shapeFunctions.interactiveTitle")}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className="h-72 bg-gray-100 rounded-lg animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  if (!shapeFunctions || shapeFunctions.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          {t("shapeFunctions.interactiveTitle")}
        </h3>
        <p className="text-gray-500 text-center py-8">
          {t("shapeFunctions.interactiveEmptyDescription")}
          <br />
          <span className="text-sm">{t("shapeFunctions.sharedDescription")}</span>
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <SurenessModal
        isOpen={showSurenessModal}
        onClose={handleSurenessModalClose}
        onConfirm={handleSurenessConfirm}
        featureName={pendingFeatureSubmit || ""}
        t={t}
      />
      <FeatureChartSettingsModal
        isOpen={Boolean(chartSettingsFeature)}
        shapeFunction={chartSettingsFeature}
        onClose={handleCloseChartSettings}
        onSave={handleSaveChartSettings}
        t={t}
      />
      <FeatureDetailsDrawer
        isOpen={Boolean(detailsFeatureName)}
        featureName={detailsFeatureName || ""}
        featureSchemaEntry={activeFeatureSchemaEntry}
        onClose={handleCloseDetails}
        t={t}
      />

      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-700">
            {t("shapeFunctions.interactiveTitle")}
          </h3>
          <p className="text-sm text-gray-500">
            {isEditing
              ? t("shapeFunctions.interactiveEditDescription")
              : t("shapeFunctions.interactiveViewDescription")}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasChanges && (
            <button
              onClick={handleReset}
              className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
            >
              {t("shapeFunctions.resetAll")}
            </button>
          )}
          <button
            onClick={() => setSyncAxes((v) => !v)}
            title={
              syncAxes
                ? t("shapeFunctions.syncAxesHintOn")
                : t("shapeFunctions.syncAxesHintOff")
            }
            className={`px-3 py-1.5 text-sm rounded-lg transition-colors font-medium flex items-center gap-1.5 ${
              syncAxes
                ? "bg-indigo-500 text-white shadow-md"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
            {syncAxes
              ? t("shapeFunctions.axesSynced")
              : t("shapeFunctions.syncAxes")}
          </button>
          <button
            onClick={() => setIsEditing(!isEditing)}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors font-medium ${
              isEditing
                ? "bg-blue-500 text-white shadow-md"
                : "bg-blue-100 text-blue-700 hover:bg-blue-200"
            }`}
          >
            {isEditing
              ? t("shapeFunctions.editingModeOn")
              : t("shapeFunctions.enableEditing")}
          </button>
        </div>
      </div>

      {isEditing && (
        <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-700">{t("shapeFunctions.editHelp")}</p>
        </div>
      )}

      {isEditing && hasNumericCharts && (
        <BrushHardnessControl
          brushHardness={brushHardness}
          onChange={setBrushHardness}
          t={t}
          className="mb-4"
        />
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {shapeFunctions.map((sf, index) => (
          <FeatureEditCard
            key={sf.feature_name || index}
            shapeFunction={sf}
            comparisonShapeFunction={
              comparisonShapeFunctionMap.get(sf.feature_name) || null
            }
            featureSchemaEntry={featureSchemaMap.get(sf.feature_name) || null}
            editedPoints={getMergedEditedPoints(sf.feature_name)}
            unsavedEditedPoints={editedPoints[sf.feature_name] || []}
            onPointEdit={handlePointEdit}
            onFeatureSubmit={handleFeatureSubmit}
            onFeatureReset={handleFeatureReset}
            onOpenChartSettings={handleOpenChartSettings}
            onOpenDetails={handleOpenDetails}
            onCycleChartType={handleCycleChartType}
            isEditing={isEditing}
            brushHardness={brushHardness}
            onBrushHardnessChange={setBrushHardness}
            hasSavedEdits={
              initialEditedPoints[sf.feature_name] &&
              initialEditedPoints[sf.feature_name].length > 0
            }
            isSuperadmin={isSuperadmin}
            sharedYRange={globalYRange}
            t={t}
          />
        ))}
      </div>
    </div>
  );
};

export default EditableShapeFunctionsGrid;
