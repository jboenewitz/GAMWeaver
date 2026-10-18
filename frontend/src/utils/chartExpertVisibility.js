export const getChartExpertKey = (trace) =>
  String(trace?.user_name || trace?.submission_id || "")
    .trim()
    .toLowerCase();

export const getFeatureChartExperts = (featurePreview) => {
  const traces = [
    ...(featurePreview.left_individual_edit_traces || []),
    ...(featurePreview.right_individual_edit_traces || []),
  ];
  return Array.from(
    new Map(
      traces.map((trace) => [
        getChartExpertKey(trace),
        trace.user_name || trace.submission_id,
      ]),
    ),
    ([key, name]) => ({ key, name }),
  );
};

export const isChartExpertVisible = (trace, hiddenExpertKeys) =>
  !hiddenExpertKeys?.[getChartExpertKey(trace)];
