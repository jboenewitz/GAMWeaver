const MODEL_FIELDS = {
  full: { metricField: "baseline", labelKey: "fullData" },
  reduced: { metricField: "reduced", labelKey: "reducedData" },
  edited: { metricField: "edited", labelKey: "editedReducedData" },
};

const FULL_REDUCED = { from: "full", to: "reduced", labelKey: "reducedMinusFull" };
const FULL_EDITED = { from: "full", to: "edited", labelKey: "editedMinusFull" };
const REDUCED_EDITED = { from: "reduced", to: "edited", labelKey: "editedMinusReduced" };

export const HELD_OUT_COMPARISONS = {
  reducedEdited: {
    modelKeys: ["reduced", "edited"],
    differencePairs: [REDUCED_EDITED],
    descriptionKey: "reducedToEdited",
    headingKey: "effectOfExpertEdits",
  },
  allThree: {
    modelKeys: ["full", "reduced", "edited"],
    differencePairs: [FULL_REDUCED, FULL_EDITED, REDUCED_EDITED],
    descriptionKey: "allThreeComparison",
    headingKey: "selectedModelComparison",
  },
  fullReduced: {
    modelKeys: ["full", "reduced"],
    differencePairs: [FULL_REDUCED],
    descriptionKey: "fullToReduced",
    headingKey: "selectedModelComparison",
  },
  fullEdited: {
    modelKeys: ["full", "edited"],
    differencePairs: [FULL_EDITED],
    descriptionKey: "fullToEdited",
    headingKey: "selectedModelComparison",
  },
};

export const getHeldOutMetricComparison = (performance, selection, metricKey) => {
  const comparison = HELD_OUT_COMPARISONS[selection] || HELD_OUT_COMPARISONS.reducedEdited;
  const valueFor = (modelKey) => {
    const rawValue = performance?.[MODEL_FIELDS[modelKey].metricField]?.[metricKey];
    return metricKey === "mean_error" && Number.isFinite(rawValue)
      ? Math.abs(rawValue)
      : rawValue;
  };
  const models = comparison.modelKeys.map((modelKey) => ({
    key: modelKey,
    labelKey: MODEL_FIELDS[modelKey].labelKey,
    value: valueFor(modelKey),
  }));
  const differences = comparison.differencePairs.map(({ from, to, labelKey }) => {
    const before = valueFor(from);
    const after = valueFor(to);
    const change = Number.isFinite(before) && Number.isFinite(after)
      ? after - before
      : null;
    const relativeChange = change !== null && Math.abs(before) > 1e-12
      ? (100 * change) / Math.abs(before)
      : null;
    const status = change === null || metricKey === "std_dev"
      ? null
      : change === 0
        ? "unchanged"
        : (metricKey === "r2" ? change > 0 : change < 0)
          ? "improved"
          : "worsened";
    return { from, to, labelKey, change, relativeChange, status };
  });
  return { models, differences };
};
