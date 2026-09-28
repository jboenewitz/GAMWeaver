import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import { createTranslator } from "../src/i18n.js";

const componentSource = readFileSync(
  new URL("../src/components/ModelComparePage.jsx", import.meta.url),
  "utf8",
);
const colorLogic = componentSource
  .match(
    /const INDIVIDUAL_EDIT_COLORS = \[[\s\S]*?const buildSelectionMap/,
  )[0]
  .replace("const buildSelectionMap", "globalThis.getColor = getIndividualEditColor;");
const context = {};
vm.runInNewContext(colorLogic, context);

test("model comparison assigns every expert a stable predefined chart color", () => {
  const expectedExpertColors = {
    E1: "#17eb65",
    E2: "#abf5ff",
    E3: "#ff3d3d",
    E4: "#fdeb74",
    E5: "#c4b5fd",
    E6: "#bef264",
    E7: "#f472b6",
  };

  for (const [expertId, color] of Object.entries(expectedExpertColors)) {
    assert.equal(context.getColor(expertId), color);
  }

  assert.equal(context.getColor("P1"), expectedExpertColors.E1);
  assert.equal(context.getColor("E1"), "#17eb65");
});

test("model comparison legends identify anonymized edits as experts", () => {
  assert.match(
    componentSource,
    /const formatChartExpertLabel = \(userName\) =>[\s\S]*?`Expert\$\{expertMatch\[1\]\} \(E\$\{expertMatch\[1\]\}\)`/,
  );
});

test("individual edit legends omit the edited-model prefix", () => {
  assert.match(componentSource, /const name = `\$\{participantLabel\}\$\{/);
});

test("aggregate comparison traces refer to expert edits", () => {
  assert.equal(
    createTranslator("en")("modelCompare.aggregatedParticipantEdits"),
    "Aggregated expert edits",
  );
});

test("confidence legend control displays and sorts individual expert traces", () => {
  assert.equal(
    createTranslator("en")("modelCompare.showConfidenceInLegend"),
    "Show confidence rating in legend",
  );
  assert.match(componentSource, /showConfidenceInLegend/);
  assert.match(componentSource, /right\.meta\.sureness - left\.meta\.sureness/);
  assert.match(componentSource, /\$\{t\("modelCompare\.confidence"\)\}: \$\{confidence\}\/10/);
});

test("chart controls can reveal a plain MAE for every individual expert trace", () => {
  assert.equal(
    createTranslator("en")("modelCompare.showIndividualMae"),
    "Show individual MAE errors",
  );
  assert.match(componentSource, /const IndividualParticipantMaeList = \(/);
  assert.match(componentSource, /showIndividualParticipantMae/);
  assert.match(componentSource, /left_individual_edit_traces/);
  assert.match(componentSource, /right_individual_edit_traces/);
  assert.match(componentSource, /formatChartExpertLabel\(trace\.user_name\)/);
  assert.match(componentSource, /trace\.original_base_mae/);
});

test("chart centering is opt-in and subtracts each curve's mean", () => {
  assert.equal(
    createTranslator("en")("modelCompare.centerCharts"),
    "Center curves",
  );
  const centeringLogic = componentSource.match(
    /const centerChartTraces = \([\s\S]*?\n};/,
  )[0];
  const centeringContext = {};
  vm.runInNewContext(
    `${centeringLogic}\nglobalThis.centerChartTraces = centerChartTraces;`,
    centeringContext,
  );
  const traces = [{ name: "Expert1", y: [1, 3, 5] }];

  assert.strictEqual(centeringContext.centerChartTraces(traces, false), traces);
  assert.deepEqual(
    Array.from(centeringContext.centerChartTraces(traces, true)[0].y),
    [-2, 0, 2],
  );
  assert.deepEqual(traces[0].y, [1, 3, 5]);
});

test("exported comparison charts retain labeled ticks on both sides of zero", () => {
  const scaleLogic = componentSource.match(
    /const getEffectYAxisScale = \([\s\S]*?\n};/,
  );
  assert.ok(
    scaleLogic,
    "export charts need an explicit y-axis scale that extends beyond zero",
  );

  const scaleContext = {};
  vm.runInNewContext(
    `${scaleLogic[0]}\nglobalThis.getEffectYAxisScale = getEffectYAxisScale;`,
    scaleContext,
  );

  const scale = scaleContext.getEffectYAxisScale([
    { y: [-83, -5] },
    { y: [4] },
  ]);
  assert.deepEqual(JSON.parse(JSON.stringify(scale)), {
    range: [-100, 20],
    dtick: 20,
    tick0: 0,
  });
  assert.match(
    componentSource,
    /yaxis: \{[\s\S]*?\.\.\.getEffectYAxisScale\(chartData\)/,
  );
});

test("comparison page surfaces overall shape-function MAE above the charts", () => {
  assert.equal(
    createTranslator("en")("modelCompare.overallShapeFunctionMae"),
    "Overall shape-function MAE",
  );
  assert.match(componentSource, /const OverallShapeFunctionMaeSummary = \(/);
  assert.match(componentSource, /overall_shape_function_mae/);
  assert.match(componentSource, /modelCompare\.pooledShapeFunctionMae/);
  assert.match(componentSource, /modelCompare\.meanPerFeatureMae/);
});

test("superadmin can limit overall MAE to selected shape functions", () => {
  assert.match(componentSource, /modelCompare\.overallMaeFeatureSelection/);
  assert.match(componentSource, /overall_mae_feature_names/);
  assert.match(componentSource, /handleOverallMaeFeatureToggle/);
  assert.match(componentSource, /<details/);
});

test("overall MAE feature selection shows each feature's direction and MAE delta", () => {
  assert.equal(
    createTranslator("en")("modelCompare.featureMaeDelta", {
      delta: "-0.250",
    }),
    "MAE Δ: -0.250",
  );
  assert.match(componentSource, /const getOverallMaeFeatureTone = \(/);
  assert.match(componentSource, /edited_vs_original_base_status/);
  assert.match(componentSource, /baseline_edited_vs_original_base_mae/);
  assert.match(componentSource, /const maeDelta =/);
  assert.match(componentSource, /formatSignedMetricValue\(maeDelta, language\)/);
  assert.match(componentSource, /modelCompare\.featureMaeDelta/);
});

test("overall MAE feature cards toggle without checkboxes and mute deselected features", () => {
  const overallMaeSegment = componentSource.match(
    /const OverallShapeFunctionMaeSummary = \([\s\S]*?const IndividualParticipantMaeList/,
  )[0];
  assert.doesNotMatch(overallMaeSegment, /type="checkbox"/);
  assert.match(overallMaeSegment, /aria-pressed/);
  assert.match(overallMaeSegment, /grayscale/);
});

test("applied feature mean limit shows its shared edit scale factor", () => {
  assert.equal(
    createTranslator("en")("modelCompare.editScaleFactor", { factor: "0.850" }),
    "×0.850",
  );
  assert.match(componentSource, /feature_scale_factors/);
  assert.match(componentSource, /modelCompare\.editScaleFactor/);
  assert.match(componentSource, /mean_deviation_constraint\.scale_factor/);
});

test("average contribution accordion can filter features and show their live combined change", () => {
  assert.equal(
    createTranslator("en")("modelCompare.contributionFeatureSelection", {
      selected: 1,
      total: 2,
    }),
    "Choose contributions (1 of 2 selected)",
  );
  assert.match(componentSource, /contributionFeatureSelection/);
  assert.match(componentSource, /handleContributionFeatureToggle/);
  assert.match(componentSource, /selectedContributionScoreRows/);
  assert.match(componentSource, /selectedContributionDifference/);
});

test("average prediction comparison identifies the full-data artifact as its reference", () => {
  assert.equal(
    createTranslator("en")("modelCompare.averagePredictedScore"),
    "Average prediction comparison",
  );
  assert.equal(
    createTranslator("en")("modelCompare.editedAveragePredictedScore"),
    "Edited artifact on full-data reference",
  );
});

test("contribution feature selection recalculates the average-prediction cards", () => {
  assert.equal(
    createTranslator("en")("modelCompare.selectedContributionAverageView"),
    "Cards show the base offset plus selected feature contributions.",
  );
  assert.match(componentSource, /selectedContributionAverageScores/);
  assert.match(componentSource, /selectedContributionScoreRows/);
  assert.match(componentSource, /selectedContributionAverageView/);
});

test("average prediction difference card displays its relative change", () => {
  assert.equal(
    createTranslator("en")("modelCompare.averagePredictionRelativeChange", {
      change: "+77.0%",
    }),
    "Relative change: +77.0%",
  );
  assert.match(componentSource, /const averagePredictionRelativeChange =/);
  assert.match(componentSource, /modelCompare\.averagePredictionRelativeChange/);
});

test("mean-shift limiting uses the selected contribution features", () => {
  assert.match(componentSource, /mean_deviation_feature_names/);
  assert.match(componentSource, /handleContributionFeatureToggle = async/);
});

test("average prediction comparison opens one modal to apply an optimal feature factor", () => {
  assert.match(componentSource, /mean_deviation_scaled_feature_names/);
  assert.match(componentSource, /const MeanShiftFeatureSelectionModal = \(/);
  assert.match(componentSource, /setMeanShiftFeatureModalOpen\(true\)/);
  assert.match(componentSource, /handleApplyMeanShiftFeatureFactors/);
  assert.match(componentSource, /feature_scale_factors/);
  assert.doesNotMatch(componentSource, /handleMeanShiftFeatureFactorToggle/);
});

test("contribution rows can preview a positive or negative edited-feature shift", () => {
  assert.match(componentSource, /manual_feature_shifts/);
  assert.match(componentSource, /handleManualFeatureShift/);
  assert.match(componentSource, /modelCompare\.decreaseFeatureEditShift/);
  assert.match(componentSource, /modelCompare\.increaseFeatureEditShift/);
});

test("feature edit-shift number can be typed and committed from the breakdown row", () => {
  assert.match(componentSource, /handleManualFeatureShiftInputChange/);
  assert.match(componentSource, /handleManualFeatureShiftInputCommit/);
  assert.match(componentSource, /type="number"/);
  assert.match(componentSource, /onBlur=\{\(\) =>/);
});
