export const getChartDisplayTitle = (chartConfig = {}) => {
  const displayTitle = String(chartConfig?.display_title ?? "").trim();
  return displayTitle;
};

export const getShapeFunctionDisplayName = (shapeFunction) => {
  const displayTitle = getChartDisplayTitle(shapeFunction?.chart_config);
  if (displayTitle) return displayTitle;
  return String(shapeFunction?.feature_name ?? "").trim();
};

export const getFeatureDisplayName = (feature) => {
  const displayName = String(feature?.display_name ?? "").trim();
  if (displayName) return displayName;
  return String(feature?.name ?? feature?.feature_name ?? "").trim();
};
