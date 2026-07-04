import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  useMemo,
} from "react";
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

const EditableShapeFunctionChart = ({
  shapeFunction,
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

  // Calculate y-axis range
  const allYValues = isNumeric
    ? [
        ...y_values,
        ...(mergedLineData?.y || []),
        ...(dragYValue !== null ? [dragYValue] : []),
      ]
    : [...y_values, ...currentYValues];
  const yRange = sharedYRange || computeDynamicYRange(allYValues);

  // Calculate x-axis range (numeric only, explicit for coordinate conversion)
  const numXVals = isNumeric ? x_values.map(Number) : [];
  const xDataMin = isNumeric ? Math.min(...numXVals) : 0;
  const xDataMax = isNumeric ? Math.max(...numXVals) : 1;
  const xPad = isNumeric ? Math.max((xDataMax - xDataMin) * 0.05, 0.1) : 0;
  const xRange = [xDataMin - xPad, xDataMax + xPad];

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
      currentTrace,
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

    data = hasEdits ? [originalTrace, editableTrace] : [editableTrace];
  }

  // No drag crosshair shapes for numeric dragging; highlight is shown directly on the line.
  const layoutShapes = [];

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
    showlegend: hasEdits,
    legend: {
      orientation: "h",
      x: 0,
      xanchor: "left",
      y: -0.24,
      yanchor: "top",
      font: { size: 10 },
      bgcolor: "rgba(255,255,255,0.8)",
    },
    shapes: layoutShapes,
  };

  const config = {
    responsive: true,
    displayModeBar: false,
    scrollZoom: false,
    doubleClick: false,
  };

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
  editedPoints,
  unsavedEditedPoints,
  onPointEdit,
  onFeatureSubmit,
  onFeatureReset,
  onOpenChartSettings,
  onCycleChartType,
  isEditing,
  brushHardness = 50,
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
      {isEnlarged && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-6"
          onClick={() => setIsEnlarged(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800">
                {shapeFunction.feature_name}
              </h3>
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
            {/* Enlarged chart */}
            <div className="p-4">
              <EditableShapeFunctionChart
                shapeFunction={shapeFunction}
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
            <div className="px-5 py-3 border-t border-gray-200 flex justify-between items-center">
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
        </div>
      )}
    </div>
  );
};

const EditableShapeFunctionsGrid = ({
  shapeFunctions,
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
  const hasNumericCharts = useMemo(
    () =>
      Array.isArray(shapeFunctions) &&
      shapeFunctions.some((sf) => sf?.feature_type === "numeric"),
    [shapeFunctions],
  );

  // Compute global y-range across all features when syncAxes is enabled
  const globalYRange = useMemo(() => {
    if (!syncAxes || !shapeFunctions || shapeFunctions.length === 0)
      return null;
    const allY = [];
    shapeFunctions.forEach((sf) => {
      allY.push(...sf.y_values);
      (initialEditedPoints[sf.feature_name] || []).forEach((p) =>
        allY.push(p.y_value),
      );
      (editedPoints[sf.feature_name] || []).forEach((p) =>
        allY.push(p.y_value),
      );
    });
    if (allY.length === 0) return null;
    const yMin = Math.min(...allY);
    const yMax = Math.max(...allY);
    const yPadding = Math.max((yMax - yMin) * 0.3, 5);
    return [yMin - yPadding, yMax + yPadding];
  }, [syncAxes, shapeFunctions, initialEditedPoints, editedPoints]);

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
        <div className="mb-4 p-3 bg-white border border-slate-200 rounded-lg">
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
                setBrushHardness(
                  Math.max(0, Math.min(100, Number(e.target.value))),
                )
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
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {shapeFunctions.map((sf, index) => (
          <FeatureEditCard
            key={sf.feature_name || index}
            shapeFunction={sf}
            editedPoints={getMergedEditedPoints(sf.feature_name)}
            unsavedEditedPoints={editedPoints[sf.feature_name] || []}
            onPointEdit={handlePointEdit}
            onFeatureSubmit={handleFeatureSubmit}
            onFeatureReset={handleFeatureReset}
            onOpenChartSettings={handleOpenChartSettings}
            onCycleChartType={handleCycleChartType}
            isEditing={isEditing}
            brushHardness={brushHardness}
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
