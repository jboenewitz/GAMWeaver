import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const appSource = readFileSync(
  new URL("../src/App.jsx", import.meta.url),
  "utf8",
);

test("main dashboard renders the actual-versus-predicted chart", () => {
  assert.match(
    appSource,
    /const showPredictionComparisonOnMain = true;/,
    "the main dashboard must enable the prediction-comparison section",
  );
  assert.match(
    appSource,
    /\{showPredictionComparisonOnMain && \([\s\S]*?<PredictionComparisonChart/,
    "the enabled section must render the actual-versus-predicted chart component",
  );
});
