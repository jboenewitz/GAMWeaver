import React, { useState, useCallback, useEffect, useRef } from "react";
import Plot from "react-plotly.js";

const EditableShapeFunctionChart = ({
  shapeFunction,
  editedPoints,
  onPointEdit,
  isEditing,
}) => {
  if (!shapeFunction) return null;

  const { feature_name, x_values, y_values, feature_type } = shapeFunction;
  const isNumeric = feature_type === "numeric";
  const containerRef = useRef(null);
  const plotRef = useRef(null);

  // Drag state
  const [isDragging, setIsDragging] = useState(false);
  const [dragPointIndex, setDragPointIndex] = useState(null);
  const [localYValues, setLocalYValues] = useState(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Precise value entry modal state
  const [preciseEntry, setPreciseEntry] = useState(null);
  const [preciseValue, setPreciseValue] = useState("");

  // Refs for drag handling
  const yAxisRangeRef = useRef({ min: -10, max: 10 });
  const plotBoundsRef = useRef({ top: 0, bottom: 0, height: 1 });
  const hoveredPointRef = useRef(null);
  const dragStartYRef = useRef(null);
  const isDraggingRef = useRef(false);
  const lastMouseDownRef = useRef({ time: 0, pointIndex: null });

  // Get the current y values (with edits applied)
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

  // Use local values during drag, otherwise use computed values
  const currentYValues =
    localYValues !== null ? localYValues : getCurrentYValues();

  // Check if any points have been edited
  const hasEdits =
    editedPoints &&
    editedPoints.some((p) => {
      const idx = x_values.findIndex(
        (x) => String(x) === String(p.x_value) || x === p.x_value,
      );
      return idx !== -1 && Math.abs(y_values[idx] - p.y_value) > 0.001;
    });

  // Calculate y-axis range
  const allYValues = [...y_values, ...currentYValues];
  const yMin = Math.min(...allYValues);
  const yMax = Math.max(...allYValues);
  const yPadding = Math.max((yMax - yMin) * 0.3, 5);
  const yRange = [yMin - yPadding, yMax + yPadding];

  // Update y-axis range ref
  useEffect(() => {
    yAxisRangeRef.current = { min: yRange[0], max: yRange[1] };
  }, [yRange[0], yRange[1]]);

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

    plotBoundsRef.current = {
      top: rect.top + marginTop,
      bottom: rect.bottom - marginBottom,
      height: rect.height - marginTop - marginBottom,
    };
  }, []);

  // Convert client Y to data Y
  const clientYToDataY = useCallback((clientY) => {
    const { top, bottom, height } = plotBoundsRef.current;
    if (height <= 0) return null;

    // Normalize: 0 at bottom, 1 at top
    const normalized = (bottom - clientY) / height;
    const clamped = Math.max(0, Math.min(1, normalized));

    // Convert to data coords
    const { min, max } = yAxisRangeRef.current;
    return min + clamped * (max - min);
  }, []);

  // Handle mouse move during drag
  useEffect(() => {
    if (!isDragging || dragPointIndex === null) return;

    const handleMouseMove = (e) => {
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
      e.preventDefault();

      // Only commit the edit if the mouse actually moved (distinguishes drag from click/dblclick)
      const movedSignificantly =
        dragStartYRef.current !== null &&
        Math.abs(e.clientY - dragStartYRef.current) > 3;

      if (movedSignificantly && localYValues && dragPointIndex !== null) {
        const xValue = x_values[dragPointIndex];
        const newY = localYValues[dragPointIndex];
        onPointEdit(feature_name, xValue, newY, feature_type);
      }

      setIsDragging(false);
      isDraggingRef.current = false;
      setDragPointIndex(null);
      setLocalYValues(null);
      dragStartYRef.current = null;
    };

    // Add listeners to window to capture mouse even outside the chart
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [
    isDragging,
    dragPointIndex,
    localYValues,
    x_values,
    feature_name,
    feature_type,
    onPointEdit,
    clientYToDataY,
  ]);

  // Start dragging when mouse is pressed on a hovered point
  const startDrag = useCallback(
    (pointIndex, clientY) => {
      if (!isEditing || isDraggingRef.current) return;

      updatePlotBounds();
      setDragPointIndex(pointIndex);
      setLocalYValues([...getCurrentYValues()]);
      setIsDragging(true);
      isDraggingRef.current = true;
      dragStartYRef.current = clientY;
    },
    [isEditing, getCurrentYValues, updatePlotBounds],
  );

  // Handle hover to track which point is under cursor
  const handleHover = useCallback(
    (eventData) => {
      if (!isEditing) return;
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
    [isEditing, hasEdits],
  );

  const handleUnhover = useCallback(() => {
    if (!isDragging) {
      setHoveredPoint(null);
      hoveredPointRef.current = null;
    }
  }, [isDragging]);

  // Native mousedown listener using capture phase to ensure
  // we intercept events before Plotly can stop propagation.
  // Double-click is detected manually from two rapid mousedowns
  // because the native dblclick fires too late (hoveredPointRef
  // gets cleared by the drag-end → re-render → unhover cycle).
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !isEditing) return;

    const onMouseDown = (e) => {
      // Use hovered point from Plotly, or fall back to the point from the
      // previous mousedown (it may have been cleared between clicks).
      const pointIndex =
        hoveredPointRef.current ?? lastMouseDownRef.current.pointIndex;
      if (pointIndex === null) return;

      e.preventDefault();
      const now = Date.now();

      if (
        lastMouseDownRef.current.pointIndex === pointIndex &&
        now - lastMouseDownRef.current.time < 400
      ) {
        // --- Double-click detected ---
        lastMouseDownRef.current = { time: 0, pointIndex: null };

        // Cancel any ongoing drag without committing
        setIsDragging(false);
        isDraggingRef.current = false;
        setDragPointIndex(null);
        setLocalYValues(null);
        dragStartYRef.current = null;

        const xValue = x_values[pointIndex];
        const currentY = getCurrentYValues()[pointIndex];
        const displayX =
          typeof xValue === "number" ? parseFloat(xValue.toFixed(3)) : xValue;

        setPreciseValue(currentY.toFixed(3));
        setPreciseEntry({ xValue, displayX, pointIndex });
      } else {
        // --- First click — start drag ---
        lastMouseDownRef.current = { time: now, pointIndex };
        startDrag(pointIndex, e.clientY);
      }
    };

    // Suppress native dblclick so Plotly doesn't reset zoom / interfere
    const onDblClick = (e) => e.preventDefault();

    container.addEventListener("mousedown", onMouseDown, true);
    container.addEventListener("dblclick", onDblClick, true);
    return () => {
      container.removeEventListener("mousedown", onMouseDown, true);
      container.removeEventListener("dblclick", onDblClick, true);
    };
  }, [isEditing, x_values, startDrag, getCurrentYValues]);

  // Build traces
  const originalTrace = {
    x: x_values,
    y: y_values,
    type: isNumeric ? "scatter" : "bar",
    mode: isNumeric ? "lines" : undefined,
    name: "Original",
    marker: { color: "#9ca3af", size: 4 },
    line: isNumeric
      ? { color: "#9ca3af", width: 1.5, dash: "dash" }
      : undefined,
    opacity: 0.6,
    showlegend: hasEdits,
    hoverinfo: "skip",
  };

  // Build marker colors and sizes based on state
  const markerColors = currentYValues.map((_, i) => {
    if (i === dragPointIndex) return "#ef4444"; // Red when dragging
    if (i === hoveredPoint && isEditing) return "#f59e0b"; // Amber when hovered
    return hasEdits ? "#10b981" : "#3b82f6"; // Green if edited, blue otherwise
  });

  const markerSizes = currentYValues.map((_, i) => {
    if (i === dragPointIndex) return 20; // Large when dragging
    if (i === hoveredPoint && isEditing) return 16; // Medium when hovered
    return isEditing ? 12 : 8; // Normal size
  });

  const editableTrace = {
    x: x_values,
    y: currentYValues,
    type: isNumeric ? "scatter" : "bar",
    mode: isNumeric ? "lines+markers" : undefined,
    name: hasEdits ? "Edited" : "Current",
    marker: {
      color: markerColors,
      size: markerSizes,
      symbol: "circle",
      line: { color: "#1e40af", width: isEditing ? 2 : 0 },
    },
    line: isNumeric
      ? { color: hasEdits ? "#10b981" : "#3b82f6", width: 2 }
      : undefined,
    fill: isNumeric ? "tozeroy" : undefined,
    fillcolor: hasEdits ? "rgba(16, 185, 129, 0.1)" : "rgba(59, 130, 246, 0.1)",
    hovertemplate: isEditing
      ? `<b>%{x}</b><br>Effect: %{y:.3f}<extra></extra>`
      : `<b>%{x}</b><br>Effect: %{y:.3f}<extra></extra>`,
  };

  const data = hasEdits ? [originalTrace, editableTrace] : [editableTrace];

  const layout = {
    title: {
      text:
        feature_name +
        (isDragging ? " (dragging...)" : hasEdits ? " (edited)" : ""),
      font: {
        size: 14,
        color: isDragging ? "#ef4444" : hasEdits ? "#10b981" : "#374151",
      },
    },
    xaxis: {
      title: feature_name,
      gridcolor: "#e5e7eb",
      tickfont: { size: 10 },
      fixedrange: true,
    },
    yaxis: {
      title: "Effect",
      gridcolor: "#e5e7eb",
      zeroline: true,
      zerolinecolor: "#9ca3af",
      zerolinewidth: 1,
      tickfont: { size: 10 },
      fixedrange: true,
      range: yRange,
    },
    margin: { l: 50, r: 20, t: 40, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: isDragging ? "#fef2f2" : "white", // Light red background when dragging
    height: 280,
    dragmode: false,
    hovermode: "closest",
    showlegend: hasEdits,
    legend: {
      x: 0.02,
      y: 0.98,
      font: { size: 10 },
      bgcolor: "rgba(255,255,255,0.8)",
    },
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
    } else if (hoveredPoint !== null) {
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
                  onPointEdit(
                    feature_name,
                    preciseEntry.xValue,
                    parseFloat(preciseValue),
                    feature_type,
                  );
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
                    onPointEdit(
                      feature_name,
                      preciseEntry.xValue,
                      parseFloat(preciseValue),
                      feature_type,
                    );
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
          ) : hoveredPoint !== null ? (
            <>
              <span className="font-medium">
                Point {hoveredPoint + 1} selected
              </span>
              <span className="text-blue-400">|</span>
              <span>
                Click and drag to edit, double-click for precise value
              </span>
            </>
          ) : (
            <>
              <span className="font-medium">
                Hover over a point to select it
              </span>
              <span className="text-blue-400">|</span>
              <span>Then drag up/down to edit</span>
            </>
          )}
        </div>
      )}
      <Plot
        ref={plotRef}
        data={data}
        layout={layout}
        config={config}
        onHover={handleHover}
        onUnhover={handleUnhover}
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
}) => {
  // Show Submit only when there are genuinely new unsaved edits
  const hasUnsavedEdits = unsavedEditedPoints && unsavedEditedPoints.length > 0;

  return (
    <div className="relative">
      <EditableShapeFunctionChart
        shapeFunction={shapeFunction}
        editedPoints={editedPoints}
        onPointEdit={onPointEdit}
        isEditing={isEditing}
      />
      <div className="mt-2 flex justify-between items-center">
        {/* Reset button - only show if there are saved edits */}
        {hasSavedEdits && (
          <button
            onClick={() => onFeatureReset(shapeFunction.feature_name)}
            className="px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg transition-colors font-medium flex items-center gap-1"
          >
            <span>✕</span>
            <span>Reset</span>
          </button>
        )}
        {!hasSavedEdits && <div />}

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
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedPoints, setEditedPoints] = useState({});
  const [hasChanges, setHasChanges] = useState(false);
  const [showSurenessModal, setShowSurenessModal] = useState(false);
  const [pendingFeatureSubmit, setPendingFeatureSubmit] = useState(null);

  // Only reset unsaved edits when a completely new model is trained
  useEffect(() => {
    setEditedPoints({});
    setHasChanges(false);
  }, [shapeFunctions]);

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
    (featureName, xValue, yValue, featureType) => {
      setEditedPoints((prev) => {
        const featurePoints = prev[featureName] || [];
        const existingIndex = featurePoints.findIndex(
          (p) => String(p.x_value) === String(xValue),
        );

        let newPoints;
        if (existingIndex >= 0) {
          newPoints = [...featurePoints];
          newPoints[existingIndex] = { x_value: xValue, y_value: yValue };
        } else {
          newPoints = [...featurePoints, { x_value: xValue, y_value: yValue }];
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
            <strong>How to edit:</strong> Hover over a point (it will
            highlight), then click and drag up or down to change its value.
            Double-click for precise value entry. When done editing a feature,
            click its <strong>Submit</strong> button and rate your confidence.
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
            onFeatureReset={onFeatureReset}
            isEditing={isEditing}
            hasSavedEdits={
              initialEditedPoints[sf.feature_name] &&
              initialEditedPoints[sf.feature_name].length > 0
            }
          />
        ))}
      </div>
    </div>
  );
};

export default EditableShapeFunctionsGrid;
