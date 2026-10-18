import assert from "node:assert/strict";
import { test } from "node:test";
import { HELD_OUT_COMPARISONS, getHeldOutMetricComparison } from "../src/utils/heldOutComparison.js";

const performance = {
  baseline: { mae: 10, r2: 0.7, mean_error: -3, std_dev: 2 },
  reduced: { mae: 12, r2: 0.5, mean_error: -1, std_dev: 4 },
  edited: { mae: 11, r2: 0.6, mean_error: 2, std_dev: 5 },
};

test("the selected held-out comparison controls model values and deltas", () => {
  const expected = {
    reducedEdited: { values: [12, 11], changes: [-1] },
    allThree: { values: [10, 12, 11], changes: [2, 1, -1] },
    fullReduced: { values: [10, 12], changes: [2] },
    fullEdited: { values: [10, 11], changes: [1] },
  };
  for (const [selection, result] of Object.entries(expected)) {
    const comparison = getHeldOutMetricComparison(performance, selection, "mae");
    assert.deepEqual(comparison.models.map((model) => model.value), result.values);
    assert.deepEqual(comparison.differences.map((difference) => difference.change), result.changes);
    assert.deepEqual(
      HELD_OUT_COMPARISONS[selection].modelKeys,
      comparison.models.map((model) => model.key),
    );
  }
});

test("absolute mean prediction error compares bias magnitudes", () => {
  const comparison = getHeldOutMetricComparison(performance, "reducedEdited", "mean_error");
  assert.deepEqual(comparison.models.map((model) => model.value), [1, 2]);
  assert.equal(comparison.differences[0].change, 1);
  assert.equal(comparison.differences[0].status, "worsened");
});

test("prediction standard deviation has a delta without an accuracy verdict", () => {
  const comparison = getHeldOutMetricComparison(performance, "allThree", "std_dev");
  assert.deepEqual(comparison.differences.map((difference) => difference.change), [2, 3, 1]);
  assert.ok(comparison.differences.every((difference) => difference.status === null));
});
