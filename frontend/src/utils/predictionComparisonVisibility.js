export function isPredictionComparisonVisible(modelStatus, currentUser) {
  if (!modelStatus || !currentUser) return false;
  return currentUser.is_superadmin
    ? Boolean(modelStatus.show_prediction_comparison_superadmin)
    : Boolean(modelStatus.show_prediction_comparison_users);
}
