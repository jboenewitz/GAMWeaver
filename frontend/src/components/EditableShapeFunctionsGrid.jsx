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

  // Refs for drag handling
  const yAxisRangeRef = useRef({ min: -10, max: 10 });
  const plotBoundsRef = useRef({ top: 0, bottom: 0, height: 1 });

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
        (x) => String(x) === String(p.x_value) || x === p.x_value
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
      ".js-plotly-plot .plot-container"
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

      if (localYValues && dragPointIndex !== null) {
        const xValue = x_values[dragPointIndex];
        const newY = localYValues[dragPointIndex];
        onPointEdit(feature_name, xValue, newY, feature_type);
      }

      setIsDragging(false);
      setDragPointIndex(null);
      setLocalYValues(null);
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

  // Start dragging when clicking on a point
  const startDrag = useCallback(
    (pointIndex) => {
      if (!isEditing) return;

      updatePlotBounds();
      setDragPointIndex(pointIndex);
      setLocalYValues([...getCurrentYValues()]);
      setIsDragging(true);
    },
    [isEditing, getCurrentYValues, updatePlotBounds]
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
      } else {
        setHoveredPoint(null);
      }
    },
    [isEditing, hasEdits]
  );

  const handleUnhover = useCallback(() => {
    if (!isDragging) {
      setHoveredPoint(null);
    }
  }, [isDragging]);

  // Handle mousedown on the plot container
  const handleMouseDown = useCallback(
    (e) => {
      if (!isEditing) return;
      if (hoveredPoint === null) return;

      e.preventDefault();
      startDrag(hoveredPoint);
    },
    [isEditing, hoveredPoint, startDrag]
  );

  // Handle double-click for precise value entry
  const handleDoubleClick = useCallback(
    (e) => {
      if (!isEditing || hoveredPoint === null) return;

      e.preventDefault();

      const pointIndex = hoveredPoint;
      const xValue = x_values[pointIndex];
      const currentY = currentYValues[pointIndex];

      const newY = prompt(
        `Enter new effect value for ${feature_name} = ${xValue}:`,
        currentY.toFixed(4)
      );

      if (newY !== null && !isNaN(parseFloat(newY))) {
        onPointEdit(feature_name, xValue, parseFloat(newY), feature_type);
      }
    },
    [
      isEditing,
      hoveredPoint,
      x_values,
      currentYValues,
      feature_name,
      feature_type,
      onPointEdit,
    ]
  );

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
        feature_name + (isDragging ? " (dragging...)" : hasEdits ? " ✏️" : ""),
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
      onMouseDown={handleMouseDown}
      onDoubleClick={handleDoubleClick}
    >
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
                ⬆️⬇️ Dragging point {dragPointIndex + 1}
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
                🎯 Point {hoveredPoint + 1} selected
              </span>
              <span className="text-blue-400">|</span>
              <span>
                Click and drag to edit, double-click for precise value
              </span>
            </>
          ) : (
            <>
              <span className="font-medium">
                🖱️ Hover over a point to select it
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

const EditableShapeFunctionsGrid = ({
  shapeFunctions,
  loading,
  onShapeFunctionsEdit,
  onReset,
  initialEditedPoints = {},
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedPoints, setEditedPoints] = useState({});
  const [hasChanges, setHasChanges] = useState(false);

  // Initialize with saved edits when shape functions or initialEditedPoints change
  useEffect(() => {
    if (Object.keys(initialEditedPoints).length > 0) {
      setEditedPoints(initialEditedPoints);
      setHasChanges(false); // These are saved edits, not new changes
    } else {
      setEditedPoints({});
      setHasChanges(false);
    }
  }, [shapeFunctions, initialEditedPoints]);

  const handlePointEdit = useCallback(
    (featureName, xValue, yValue, featureType) => {
      setEditedPoints((prev) => {
        const featurePoints = prev[featureName] || [];
        const existingIndex = featurePoints.findIndex(
          (p) => String(p.x_value) === String(xValue)
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
    []
  );

  const handleApplyChanges = useCallback(() => {
    const editedShapeFunctions = Object.entries(editedPoints)
      .filter(([_, points]) => points.length > 0)
      .map(([featureName, points]) => {
        const sf = shapeFunctions.find((s) => s.feature_name === featureName);
        return {
          feature_name: featureName,
          feature_type: sf?.feature_type || "numeric",
          edited_points: points,
        };
      });

    if (editedShapeFunctions.length > 0) {
      onShapeFunctionsEdit(editedShapeFunctions);
    }
  }, [editedPoints, shapeFunctions, onShapeFunctionsEdit]);

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
          📊 Interactive Shape Functions
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
          📊 Interactive Shape Functions
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
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-700">
            📊 Interactive Shape Functions
          </h3>
          <p className="text-sm text-gray-500">
            {isEditing
              ? "Hover over points and drag them up/down to modify the shape functions."
              : "Enable editing mode to interactively modify shape functions."}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {hasChanges && (
            <>
              <button
                onClick={handleReset}
                className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              >
                Reset
              </button>
              <button
                onClick={handleApplyChanges}
                className="px-3 py-1.5 text-sm bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors font-medium shadow-md"
              >
                ✓ Apply Changes
              </button>
            </>
          )}
          <button
            onClick={() => setIsEditing(!isEditing)}
            className={`px-4 py-1.5 text-sm rounded-lg transition-colors font-medium ${
              isEditing
                ? "bg-blue-500 text-white shadow-md"
                : "bg-blue-100 text-blue-700 hover:bg-blue-200"
            }`}
          >
            {isEditing ? "✓ Editing Mode ON" : "🖊️ Enable Editing"}
          </button>
        </div>
      </div>

      {isEditing && (
        <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-700">
            <strong>How to edit:</strong> Hover over a point (it will
            highlight), then click and drag up or down to change its value.
            Double-click for precise value entry. The gray dashed line shows the
            original shape function. Click "Apply Changes" when done.
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {shapeFunctions.map((sf, index) => (
          <EditableShapeFunctionChart
            key={sf.feature_name || index}
            shapeFunction={sf}
            editedPoints={editedPoints[sf.feature_name] || []}
            onPointEdit={handlePointEdit}
            isEditing={isEditing}
          />
        ))}
      </div>
    </div>
  );
};

export default EditableShapeFunctionsGrid;
