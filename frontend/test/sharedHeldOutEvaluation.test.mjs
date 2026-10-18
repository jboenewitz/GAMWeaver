import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildSharedHeldOutArtifacts,
  needsIqbSharedHeldOut,
} from "../src/utils/sharedHeldOutEvaluation.js";

test("source lineage removes reduced training rows before the page requests held-out metrics", () => {
  const left = {
    held_out_test_data: {
      feature_rows: [{ x: 8 }, { x: 1 }],
      target_values: [108, 101],
      sample_ids: ["local-8", "local-1"],
    },
    training_sample_ids: ["local-train"],
  };
  const right = {
    training_feature_rows: [{ x: 1 }],
    training_sample_ids: ["local-reduced"],
    held_out_test_data: { feature_rows: [{ x: 2 }], target_values: [102], sample_ids: ["local-2"] },
  };
  const lineage = {
    source_row_namespace: "source-hash",
    original_training_source_ids: ["0", "2"],
    original_test_source_ids: ["8", "1"],
    reduced_training_source_ids: ["1"],
    reduced_test_source_ids: ["2"],
    kept_original_test_indices: [0],
    shared_test_row_count: 1,
    original_numeric_missing_placeholder_values: { x: 55.25 },
    reduced_numeric_missing_placeholder_values: { x: 56 },
  };
  const result = buildSharedHeldOutArtifacts(left, right, lineage);
  assert.deepEqual(result.left.held_out_test_data, {
    feature_rows: [{ x: 8 }], target_values: [108], sample_ids: ["8"],
  });
  assert.deepEqual(result.right.training_sample_ids, ["1"]);
  assert.deepEqual(result.right.held_out_test_data.sample_ids, ["2"]);
  assert.equal(result.left.sample_id_namespace, result.right.sample_id_namespace);
  assert.equal(result.left.numeric_missing_placeholder_values.x, 55.25);
  assert.equal(result.right.numeric_missing_placeholder_values.x, 56);
  assert.equal(left.held_out_test_data.feature_rows.length, 2);
});

test("raw old IQB export requires lineage, while an already filtered export does not", () => {
  const old = {
    metadata: { dataset_name: "IQB_random100pct_pv_mean_old.csv" },
    held_out_test_data: { feature_rows: Array(5249) },
  };
  assert.equal(needsIqbSharedHeldOut(old), true);
  assert.equal(needsIqbSharedHeldOut({
    ...old, held_out_test_data: { feature_rows: Array(5030) },
  }), false);
  assert.throws(() => buildSharedHeldOutArtifacts(
    { held_out_test_data: { feature_rows: [{ x: 1 }], target_values: [1] } },
    { training_feature_rows: [{ x: 1 }] },
    { original_test_source_ids: ["1"], kept_original_test_indices: [0],
      shared_test_row_count: 1, original_training_source_ids: [],
      reduced_training_source_ids: ["1"] },
  ), /row identities/);
});
