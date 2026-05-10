import React, {
  useState,
  useCallback,
  useEffect,
  useRef,
  useMemo,
} from "react";
import Plot from "react-plotly.js";

const NUMERIC_X_PRECISION = 6;
const NUMERIC_BRUSH_SIGMA_RATIO = 0.08;
const NUMERIC_BRUSH_MIN_SIGMA = 0.25;
const NUMERIC_BRUSH_RADIUS_MULTIPLIER = 3.0;
const NUMERIC_BRUSH_SMOOTHING_WINDOW = 5;
const UNSYNCED_Y_PADDING_RATIO = 0.15;
const UNSYNCED_Y_MIN_PADDING = 0.2;
const FLAT_Y_MIN_HALF_SPAN = 0.5;

const roundNumericX = (x) =>
  Math.round(Number(x) * 10 ** NUMERIC_X_PRECISION) /
  10 ** NUMERIC_X_PRECISION;

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

  const padding = Math.max(span * UNSYNCED_Y_PADDING_RATIO, UNSYNCED_Y_MIN_PADDING);
  return [yMin - padding, yMax + padding];
};

const applyMovingAverageInWindow = (points, centerIndex, windowSize) => {
  if (windowSize <= 1 || points.length < 3) return points;
  const half = Math.floor(windowSize / 2);
  const result = points.map((point) => ({ ...point }));

  for (let idx = Math.max(1, centerIndex - half); idx <= Math.min(points.length - 2, centerIndex + half); idx += 1) {
    let weightedSum = 0;
    let weightTotal = 0;
    for (let j = Math.max(0, idx - half); j <= Math.min(points.length - 1, idx + half); j += 1) {
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

const EditableShapeFunctionChart = ({
  shapeFunction,
  editedPoints,
  onPointEdit,
  isEditing,
  enlarged = false,
  sharedYRange = null,
}) => {
  if (!shapeFunction) return null;

  const { feature_name, x_values, y_values, feature_type } = shapeFunction;
  const isNumeric = feature_type === "numeric";
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
  }, [
    isNumeric,
    isDragging,
    dragCurvePoints,
    shapeFunction,
    editedPoints,
  ]);

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
    return Math.max(xSpan * NUMERIC_BRUSH_SIGMA_RATIO, NUMERIC_BRUSH_MIN_SIGMA);
  }, [xDataMax, xDataMin]);

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
      );

      onPointEdit(
        feature_name,
        toEditedPointsFormat(updatedCurve),
        null,
        feature_type,
      );
    },
    [shapeFunction, editedPoints, getBrushSigma, onPointEdit, feature_name, feature_type],
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
      name: "Original",
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
      name: hasEdits ? "Edited" : "Current",
      line: { color: hasEdits ? "#10b981" : "#3b82f6", width: 2.5 },
      fill: "tozeroy",
      fillcolor: hasEdits
        ? "rgba(16, 185, 129, 0.1)"
        : "rgba(59, 130, 246, 0.1)",
      hovertemplate: "<b>%{x:.3f}</b><br>Effect: %{y:.3f}<extra></extra>",
    };

    // Small markers at edited positions only
    const showEditMarkers =
      (editedPoints || []).length > 0 &&
      (editedPoints || []).length < Math.max(20, Math.floor(x_values.length * 0.5));
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
            name: "Edit points",
            marker: {
              color: "#10b981",
              size: 8,
              symbol: "circle",
              line: { color: "#065f46", width: 1.5 },
            },
            showlegend: false,
            hovertemplate:
              "<b>x: %{x:.3f}</b><br>Effect: %{y:.3f}<extra>Edited</extra>",
          }
        : null;

    // Drag preview marker
    const dragPreviewTrace =
      isDragging && dragXValue !== null && dragYValue !== null
        ? {
            x: [Number(dragXValue)],
            y: [dragYValue],
            type: "scatter",
            mode: "markers",
            name: "Dragging",
            marker: {
              color: "#ef4444",
              size: 14,
              symbol: "circle",
              line: { color: "#991b1b", width: 2 },
            },
            showlegend: false,
            hoverinfo: "skip",
          }
        : null;

    data = [
      ...(hasEdits ? [originalTrace] : []),
      currentTrace,
      ...(editMarkersTrace ? [editMarkersTrace] : []),
      ...(dragPreviewTrace ? [dragPreviewTrace] : []),
    ];
  } else {
    // ---- Categorical bar chart traces (unchanged) ----
    const originalTrace = {
      x: x_values,
      y: y_values,
      type: "bar",
      name: "Original",
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
      name: hasEdits ? "Edited" : "Current",
      marker: {
        color: markerColors,
        size: markerSizes,
        symbol: "circle",
        line: { color: "#1e40af", width: isEditing ? 2 : 0 },
      },
      hovertemplate: `<b>%{x}</b><br>Effect: %{y:.3f}<extra></extra>`,
    };

    data = hasEdits ? [originalTrace, editableTrace] : [editableTrace];
  }

  // Build layout shapes for drag crosshairs (numeric only)
  const layoutShapes = [];
  if (isNumeric && isDragging && dragXValue !== null) {
    layoutShapes.push({
      type: "line",
      x0: dragXValue,
      x1: dragXValue,
      y0: yRange[0],
      y1: yRange[1],
      line: { color: "#ef4444", width: 1, dash: "dot" },
    });
    if (dragYValue !== null) {
      layoutShapes.push({
        type: "line",
        x0: xRange[0],
        x1: xRange[1],
        y0: dragYValue,
        y1: dragYValue,
        line: { color: "#ef4444", width: 1, dash: "dot" },
      });
    }
  }

  const layout = {
    title: {
      text:
        feature_name +
        (isDragging ? " (dragging...)" : hasEdits ? " (edited)" : ""),
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
      ...(isNumeric ? { range: xRange } : {}),
    },
    yaxis: {
      title: { text: "Effect", font: { size: enlarged ? 14 : 12 } },
      gridcolor: "#e5e7eb",
      zeroline: true,
      zerolinecolor: "#9ca3af",
      zerolinewidth: 1,
      tickfont: { size: enlarged ? 12 : 10 },
      fixedrange: true,
      range: yRange,
    },
    margin: enlarged
      ? { l: 70, r: 40, t: 50, b: 60 }
      : { l: 50, r: 20, t: 40, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: isDragging ? "#fef2f2" : "white",
    height: enlarged ? 520 : 280,
    dragmode: false,
    hovermode: "closest",
    showlegend: hasEdits,
    legend: {
      x: 0.02,
      y: 0.98,
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
              Set Effect Value
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
                Cancel
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
                Apply
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
                  Brushing at x ={" "}
                  {dragXValue != null ? Number(dragXValue).toFixed(3) : ""}
                </span>
                <span className="text-red-500">|</span>
                <span>
                  Release to apply smoothed stroke:{" "}
                  {dragYValue != null ? dragYValue.toFixed(3) : ""}
                </span>
              </>
            ) : (
              <>
                <span className="font-bold animate-pulse">
                  Dragging point {dragPointIndex + 1}
                </span>
                <span className="text-red-500">|</span>
                <span>
                  Release to set value:{" "}
                  {localYValues?.[dragPointIndex]?.toFixed(2)}
                </span>
              </>
            )
          ) : isNumeric ? (
            <>
              <span className="font-medium">
                Click and drag to brush the line. Double-click for precise entry.
              </span>
            </>
          ) : hoveredPoint !== null ? (
            <>
              <span className="font-medium">
                Point {hoveredPoint + 1} selected
              </span>
              <span className="text-blue-400">|</span>
              <span>Double-click for precise value</span>
            </>
          ) : (
            <>
              <span className="font-medium">
                Hover over a point to select it
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
const SurenessModal = ({ isOpen, onClose, onConfirm, featureName }) => {
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
          Submit Edit for {featureName}
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Rate your confidence and provide a description for your edit.
        </p>

        {/* Confidence Slider */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence Level
          </label>
          <div className="flex justify-between text-sm text-gray-500 mb-2">
            <span>Not sure</span>
            <span className="font-bold text-lg text-blue-600">{sureness}</span>
            <span>Very sure</span>
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
            Edit Description <span className="text-red-500">*</span>
          </label>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Describe why you made this edit..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm resize-none"
            rows={3}
          />
          {!isValid && message.length > 0 && (
            <p className="text-xs text-red-500 mt-1">
              Please enter a description for your edit.
            </p>
          )}
        </div>

        <div className="flex gap-3">
          <button
            onClick={handleClose}
            className="flex-1 px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
          >
            Cancel
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
            Submit Edit
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
  isEditing,
  hasSavedEdits,
  sharedYRange = null,
}) => {
  const [isEnlarged, setIsEnlarged] = useState(false);

  // Show Submit only when there are genuinely new unsaved edits
  const hasUnsavedEdits = unsavedEditedPoints && unsavedEditedPoints.length > 0;

  return (
    <div className="relative">
      {/* Enlarge button */}
      <button
        onClick={() => setIsEnlarged(true)}
        title="Enlarge chart"
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
        sharedYRange={sharedYRange}
      />
      <div className="mt-2 flex justify-between items-center">
        {/* Reset button - show if there are saved or unsaved edits */}
        {(hasSavedEdits || hasUnsavedEdits) && (
          <button
            onClick={() => onFeatureReset(shapeFunction.feature_name)}
            className="px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors font-medium flex items-center gap-1"
          >
            <span>✕</span>
            <span>Reset</span>
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
            <span>Submit {shapeFunction.feature_name}</span>
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
                title="Close"
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
                enlarged
                sharedYRange={sharedYRange}
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
                  <span>Reset</span>
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
                    <span>Submit {shapeFunction.feature_name}</span>
                  </button>
                )}
                <button
                  onClick={() => setIsEnlarged(false)}
                  className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
                >
                  Close
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
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedPoints, setEditedPoints] = useState({});
  const [hasChanges, setHasChanges] = useState(false);
  const [showSurenessModal, setShowSurenessModal] = useState(false);
  const [pendingFeatureSubmit, setPendingFeatureSubmit] = useState(null);
  const [syncAxes, setSyncAxes] = useState(false);

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
    (
      featureName,
      xValueOrPoints,
      yValue,
      featureType,
    ) => {
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

  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Interactive Shape Functions
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
          Interactive Shape Functions
        </h3>
        <p className="text-gray-500 text-center py-8">
          Train the model to see and edit feature shape functions.
          <br />
          <span className="text-sm">
            Shape functions show how each feature affects the prediction.
          </span>
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
      />

      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-700">
            Interactive Shape Functions
          </h3>
          <p className="text-sm text-gray-500">
            {isEditing
              ? "Edit points, then click Submit on each feature to save with your confidence rating."
              : "Enable editing mode to interactively modify shape functions."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasChanges && (
            <button
              onClick={handleReset}
              className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
            >
              Reset All
            </button>
          )}
          <button
            onClick={() => setSyncAxes((v) => !v)}
            title={
              syncAxes
                ? "Axes are synced — click to use per-chart scale"
                : "Click to sync all chart axes to the same scale"
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
            {syncAxes ? "Axes Synced" : "Sync Axes"}
          </button>
          <button
            onClick={() => setIsEditing(!isEditing)}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors font-medium ${
              isEditing
                ? "bg-blue-500 text-white shadow-md"
                : "bg-blue-100 text-blue-700 hover:bg-blue-200"
            }`}
          >
            {isEditing ? "Editing Mode ON" : "Enable Editing"}
          </button>
        </div>
      </div>

      {isEditing && (
        <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-700">
            <strong>How to edit:</strong> For numeric features, click and drag
            to brush edits along the curve; smoothing and slope limits prevent
            sharp spikes. For categorical features, hover a point, then drag it
            up or down. Double-click for precise value entry. When done editing
            a feature, click its <strong>Submit</strong> button and rate your
            confidence.
          </p>
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
            isEditing={isEditing}
            hasSavedEdits={
              initialEditedPoints[sf.feature_name] &&
              initialEditedPoints[sf.feature_name].length > 0
            }
            sharedYRange={globalYRange}
          />
        ))}
      </div>
    </div>
  );
};

export default EditableShapeFunctionsGrid;
