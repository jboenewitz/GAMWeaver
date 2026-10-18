import assert from "node:assert/strict";
import { test } from "node:test";
import {
  getFeatureChartExperts,
  isChartExpertVisible,
} from "../src/utils/chartExpertVisibility.js";

test("chart expert visibility is scoped to one feature and leaves its source traces intact", () => {
  const firstFeature = {
    left_individual_edit_traces: [{ user_name: "E1", submission_id: "left-1" }],
    right_individual_edit_traces: [
      { user_name: "E1", submission_id: "right-1" },
      { user_name: "E2", submission_id: "right-2" },
    ],
  };
  const secondFeature = {
    right_individual_edit_traces: [{ user_name: "E1", submission_id: "other-1" }],
  };
  const hiddenByFeature = { featureA: { e1: true } };

  assert.deepEqual(getFeatureChartExperts(firstFeature), [
    { key: "e1", name: "E1" },
    { key: "e2", name: "E2" },
  ]);
  assert.deepEqual(
    firstFeature.right_individual_edit_traces
      .filter((trace) => isChartExpertVisible(trace, hiddenByFeature.featureA))
      .map((trace) => trace.user_name),
    ["E2"],
  );
  assert.equal(
    isChartExpertVisible(
      secondFeature.right_individual_edit_traces[0],
      hiddenByFeature.featureB,
    ),
    true,
  );
  assert.equal(firstFeature.right_individual_edit_traces.length, 2);
});
