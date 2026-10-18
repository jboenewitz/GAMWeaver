export function selectCombinedResultsData(
  comparisonData,
  unweightedComparisonData,
  useWeighting,
) {
  return (useWeighting ? comparisonData : unweightedComparisonData) ?? comparisonData;
}

export function buildUserOverlayView(activeData, perUserShapeFunctions) {
  const shapes = activeData?.combined_shape_functions_display ?? [];
  const users = Array.isArray(perUserShapeFunctions) ? perUserShapeFunctions : [];
  const userTracesMap = {};

  for (const user of users) {
    const userShapes = Array.isArray(user?.shape_functions)
      ? user.shape_functions
      : [];
    for (const shape of userShapes) {
      if (!shape?.feature_name || !Array.isArray(shape.y_values)) continue;
      if (!userTracesMap[shape.feature_name]) {
        userTracesMap[shape.feature_name] = {};
      }
      userTracesMap[shape.feature_name][user.user_name] = shape.y_values;
    }
  }

  return {
    userNames: users.map((user) => user?.user_name).filter(Boolean),
    userTracesMap,
    hasComparisonOverlay: shapes.some(
      (shape) =>
        Array.isArray(shape?.comparison_y_values) &&
        Array.isArray(shape?.x_values) &&
        shape.comparison_y_values.length === shape.x_values.length,
    ),
  };
}

export function getCombinedRequestErrorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (typeof error?.message === "string" && error.message.trim()) {
    return error.message;
  }
  return fallback;
}
