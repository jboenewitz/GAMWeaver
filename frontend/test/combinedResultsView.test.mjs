import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildUserOverlayView,
  getCombinedRequestErrorMessage,
  selectCombinedResultsData,
} from "../src/utils/combinedResultsView.js";

test("combined results select weighted or unweighted curves", () => {
  const weighted = { combined_shape_functions_display: [{ y_values: [2] }] };
  const unweighted = { combined_shape_functions_display: [{ y_values: [3] }] };
  assert.equal(selectCombinedResultsData(weighted, unweighted, true), weighted);
  assert.equal(selectCombinedResultsData(weighted, unweighted, false), unweighted);
  assert.equal(selectCombinedResultsData(weighted, null, false), weighted);
});

test("user overlays include traces and the comparison legend when available", () => {
  const activeData = {
    combined_shape_functions_display: [
      { feature_name: "age", x_values: [0, 10], comparison_y_values: [1, 2] },
      { feature_name: "month", x_values: ["1"], comparison_y_values: null },
    ],
  };
  const users = [
    { user_name: "Alice", shape_functions: [{ feature_name: "age", y_values: [2, 4] }] },
    { user_name: "Bob", shape_functions: [{ feature_name: "age", y_values: [3, 5] }] },
  ];
  assert.deepEqual(buildUserOverlayView(activeData, users), {
    userNames: ["Alice", "Bob"],
    userTracesMap: { age: { Alice: [2, 4], Bob: [3, 5] } },
    hasComparisonOverlay: true,
  });
  assert.deepEqual(buildUserOverlayView(null, null), {
    userNames: [],
    userTracesMap: {},
    hasComparisonOverlay: false,
  });
});

test("overlay errors remain readable when the API detail is structured", () => {
  assert.equal(
    getCombinedRequestErrorMessage(
      { response: { data: { detail: { code: "failed" } } }, message: "Request failed" },
      "Could not load overlay",
    ),
    "Request failed",
  );
  assert.equal(getCombinedRequestErrorMessage({}, "Could not load overlay"), "Could not load overlay");
});
