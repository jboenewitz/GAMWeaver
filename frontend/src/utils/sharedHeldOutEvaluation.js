const unique = (values) => new Set(values).size === values.length;

export const needsIqbSharedHeldOut = (leftArtifact) =>
  String(leftArtifact?.metadata?.dataset_name || "").endsWith("random100pct_pv_mean_old.csv") &&
  leftArtifact?.held_out_test_data?.feature_rows?.length === 5249;

export const buildSharedHeldOutArtifacts = (leftArtifact, rightArtifact, lineage) => {
  if (!lineage) throw new Error("Verified source-row lineage is required");
  const test = leftArtifact?.held_out_test_data;
  const sourceIds = lineage.original_test_source_ids || [];
  const kept = lineage.kept_original_test_indices || [];
  const fullTrainingIds = lineage.original_training_source_ids || [];
  const reducedTrainingIds = lineage.reduced_training_source_ids || [];
  const fullTrainingSet = new Set(fullTrainingIds);
  const reducedTrainingSet = new Set(reducedTrainingIds);
  const selectedIds = kept.map((index) => sourceIds[index]);
  if (needsIqbSharedHeldOut(leftArtifact) && (
    sourceIds.length !== 5249 || reducedTrainingIds.length !== 1047 ||
    sourceIds.length - kept.length !== 219 || kept.length !== 5030
  )) {
    throw new Error("Original IQB shared held-out counts do not match 5,249 − 219 = 5,030");
  }
  if (
    !test || test.feature_rows?.length !== sourceIds.length ||
    test.target_values?.length !== sourceIds.length ||
    kept.length !== lineage.shared_test_row_count ||
    !unique(sourceIds) || !unique(selectedIds) ||
    selectedIds.some((id) => id === undefined) ||
    selectedIds.some((id) => fullTrainingSet.has(id) || reducedTrainingSet.has(id)) ||
    reducedTrainingIds.length !== rightArtifact?.training_feature_rows?.length
  ) {
    throw new Error("Shared held-out row identities do not match the prepared artifacts");
  }
  const rightTest = rightArtifact?.held_out_test_data;
  const reducedTestIds = lineage.reduced_test_source_ids || [];
  if (rightTest?.feature_rows?.length && rightTest.feature_rows.length !== reducedTestIds.length) {
    throw new Error("Reduced held-out row identities do not match the prepared artifact");
  }
  return {
    left: {
      ...leftArtifact,
      sample_id_namespace: lineage.source_row_namespace,
      training_sample_ids: fullTrainingIds,
      numeric_missing_placeholder_values:
        lineage.original_numeric_missing_placeholder_values || {},
      held_out_test_data: {
        ...test,
        feature_rows: kept.map((index) => test.feature_rows[index]),
        target_values: kept.map((index) => test.target_values[index]),
        sample_ids: selectedIds,
      },
    },
    right: {
      ...rightArtifact,
      sample_id_namespace: lineage.source_row_namespace,
      training_sample_ids: reducedTrainingIds,
      numeric_missing_placeholder_values:
        lineage.reduced_numeric_missing_placeholder_values || {},
      held_out_test_data: rightTest?.feature_rows?.length
        ? { ...rightTest, sample_ids: reducedTestIds }
        : rightTest,
    },
  };
};
