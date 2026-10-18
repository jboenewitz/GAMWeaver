import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { isPredictionComparisonVisible } from "../src/utils/predictionComparisonVisibility.js";

test("prediction comparison is hidden by default for both audiences", () => {
  assert.equal(isPredictionComparisonVisible(null, { is_superadmin: true }), false);
  assert.equal(isPredictionComparisonVisible({}, { is_superadmin: true }), false);
  assert.equal(isPredictionComparisonVisible({}, { is_superadmin: false }), false);
});

test("superadmin and other users have independent visibility controls", () => {
  const admin = { is_superadmin: true };
  const user = { is_superadmin: false };
  const status = {
    show_prediction_comparison_superadmin: true,
    show_prediction_comparison_users: false,
  };
  assert.equal(isPredictionComparisonVisible(status, admin), true);
  assert.equal(isPredictionComparisonVisible(status, user), false);
  assert.equal(
    isPredictionComparisonVisible(
      { ...status, show_prediction_comparison_users: true },
      user,
    ),
    true,
  );
});

test("main page renders Prediction Comparison according to the setting", () => {
  const appSource = readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(appSource, /isPredictionComparisonVisible\(\s*modelStatus,\s*currentUser/);
  assert.match(
    appSource,
    /\{showPredictionComparisonOnMain && \([\s\S]*?<PredictionComparisonChart/,
  );
});
