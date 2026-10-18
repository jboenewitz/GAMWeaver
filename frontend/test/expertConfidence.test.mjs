import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { createTranslator } from "../src/i18n.js";
import { buildExpertConfidenceBars } from "../src/utils/expertConfidence.js";

test("average confidence includes every exported feature submission and sorts experts descending", () => {
  const artifact = {
    submissions_by_feature: [
      { submissions: [
        { user_name: "E1", sureness: 8 },
        { user_name: "E2", sureness: 10 },
        { user_name: "E3", sureness: 9 },
      ] },
      { submissions: [
        { user_name: "E1", sureness: 6 },
        { user_name: "E2", sureness: 2 },
        { user_name: "E3", sureness: 5 },
        { user_name: "E4", sureness: null },
      ] },
    ],
  };

  assert.deepEqual(buildExpertConfidenceBars(artifact), [
    { userName: "E1", averageConfidence: 7, ratingCount: 2 },
    { userName: "E3", averageConfidence: 7, ratingCount: 2 },
    { userName: "E2", averageConfidence: 6, ratingCount: 2 },
  ]);
  assert.deepEqual(buildExpertConfidenceBars(null), []);
});

test("average confidence chart follows the aggregated MAE section", () => {
  const source = readFileSync(
    new URL("../src/components/ModelComparePage.jsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /const ExpertAverageConfidenceChart = \(/);
  assert.match(source, /autorange: "reversed"/);
  assert.match(source, /artifact=\{comparePayload\.right_artifact\}/);
  assert.match(
    source,
    /<AllFeatureMaeChangeCharts[\s\S]*?\/>\s*<ExpertAverageConfidenceChart/,
  );
  assert.equal(
    createTranslator("en")("modelCompare.expertAverageConfidence"),
    "Average confidence by expert",
  );
});
