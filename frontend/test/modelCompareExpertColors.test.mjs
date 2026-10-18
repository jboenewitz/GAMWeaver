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

test("chart controls can reveal a per-feature MAE-change bar chart for experts", () => {
  assert.equal(
    createTranslator("en")("modelCompare.showIndividualMaeChangeChart"),
    "Show expert MAE-change chart",
  );
  assert.match(componentSource, /const buildIndividualMaeChangeBars = \(/);
  assert.match(componentSource, /const IndividualParticipantMaeChangeChart = \(/);
  assert.match(componentSource, /showIndividualParticipantMaeChangeChart/);
  assert.match(
    componentSource,
    /baseline_edited_vs_original_base_mae/,
  );
  assert.match(componentSource, /getIndividualEditColor\(bar\.userName\)/);

  const barLogic = componentSource.match(
    /const buildIndividualMaeChangeBars = \([\s\S]*?\n};/,
  );
  assert.ok(barLogic, "the chart must derive one signed MAE delta per expert");
  const chartContext = {};
  vm.runInNewContext(
    `${barLogic[0]}\nglobalThis.buildBars = buildIndividualMaeChangeBars;`,
    chartContext,
  );
  const bars = chartContext.buildBars(
    {
      comparison_metrics: { baseline_edited_vs_original_base_mae: 0.4 },
      right_individual_edit_traces: [
        { submission_id: "e1", user_name: "E1", original_base_mae: 0.1 },
        { submission_id: "e2", user_name: "E2", original_base_mae: 0.7 },
      ],
    },
    (userName) => `Expert ${userName}`,
  );
  assert.deepEqual(JSON.parse(JSON.stringify(bars)), [
    { expertLabel: "Expert E1", maeChange: -0.3, userName: "E1" },
    { expertLabel: "Expert E2", maeChange: 0.3, userName: "E2" },
  ]);
});

test("MAE-change chart can hide one expert without changing the preview data", () => {
  const barLogic = componentSource.match(
    /const buildIndividualMaeChangeBars = \([\s\S]*?\n};/,
  );
  assert.ok(barLogic);
  const chartContext = {};
  vm.runInNewContext(
    `${barLogic[0]}\nglobalThis.buildBars = buildIndividualMaeChangeBars;`,
    chartContext,
  );
  const featurePreview = {
    comparison_metrics: { baseline_edited_vs_original_base_mae: 0.4 },
    right_individual_edit_traces: [
      { submission_id: "e1", user_name: "E1", original_base_mae: 0.1 },
      { submission_id: "e2", user_name: "E2", original_base_mae: 0.7 },
    ],
  };
  const bars = chartContext.buildBars(
    featurePreview,
    (userName) => `Expert ${userName}`,
    (trace) => trace.user_name !== "E1",
  );

  assert.deepEqual(JSON.parse(JSON.stringify(bars)), [
    { expertLabel: "Expert E2", maeChange: 0.3, userName: "E2" },
  ]);
  assert.equal(featurePreview.right_individual_edit_traces.length, 2);
  assert.equal(featurePreview.comparison_metrics.baseline_edited_vs_original_base_mae, 0.4);
});

test("expert MAE chart adds confidence-rated bars with matching expert colors and pattern", () => {
  assert.equal(
    createTranslator("en")("modelCompare.showConfidenceRatedMae"),
    "Show confidence-rated expert MAE",
  );
  const barLogic = componentSource.match(
    /const buildIndividualMaeChangeBars = \([\s\S]*?\n};/,
  )?.[0];
  assert.ok(barLogic);
  const chartContext = {};
  vm.runInNewContext(
    `${barLogic}\nglobalThis.buildBars = buildIndividualMaeChangeBars;`,
    chartContext,
  );
  const featurePreview = {
    comparison_metrics: { baseline_edited_vs_original_base_mae: 0.5 },
    right_individual_edit_traces: [
      {
        user_name: "E1",
        original_base_mae_without_confidence: 0.2,
        original_base_mae_with_confidence: 0.3,
      },
      {
        user_name: "E2",
        original_base_mae_without_confidence: 0.7,
        original_base_mae_with_confidence: 0.6,
      },
    ],
  };
  const plain = chartContext.buildBars(featurePreview, (name) => name);
  const rated = chartContext.buildBars(featurePreview, (name) => name, undefined, true);
  assert.deepEqual(Array.from(plain, (bar) => bar.maeChange), [-0.3, 0.2]);
  assert.deepEqual(Array.from(rated, (bar) => bar.maeChange), [-0.2, 0.1]);
  assert.match(componentSource, /showConfidenceRatedMae=\{showConfidenceRatedMae\}/);
  const chart = componentSource.match(/const ExpertMaeChangeBarChart = \([\s\S]*?\n};/)?.[0];
  assert.ok(chart);
  assert.match(chart, /color: traceBars\.map\(\(bar\) => getIndividualEditColor\(bar\.userName\)\)/);
  assert.match(chart, /bgcolor: traceBars\.map\(\(bar\) => getIndividualEditColor\(bar\.userName\)\)/);
  assert.match(chart, /shape: "\/"/);
  assert.match(chart, /showlegend: false/);
  assert.match(chart, /aria-pressed=\{!hiddenModes\.includes\(mode\)\}/);
  assert.match(chart, /repeating-linear-gradient\(135deg, #94a3b8/);
});

test("comparison shows static MAE-change bars per feature with a patterned combined view", () => {
  assert.equal(
    createTranslator("en")("modelCompare.allFeatureMaeChange"),
    "Aggregated MAE change across all features",
  );
  assert.match(componentSource, /const AllFeatureMaeChangeCharts = \(/);
  assert.match(componentSource, /all_feature_mae_change/);
  const section = componentSource.match(
    /const AllFeatureMaeChangeCharts = \([\s\S]*?\n};/,
  )?.[0];
  assert.ok(section);
  assert.match(section, /without_confidence/);
  assert.match(section, /with_confidence/);
  assert.match(section, /setCombinedView/);
  assert.match(section, /aria-pressed=\{combinedView\}/);
  assert.match(section, /combinedView\s*\?/);
  assert.doesNotMatch(section, /hiddenExpertKeys|selectionState|useConfidence/);
  const barsLogic = componentSource.match(
    /const buildAllFeatureMaeChangeBars = \(features\) =>[\s\S]*?\n    \.filter\(Boolean\);/,
  )?.[0];
  assert.ok(barsLogic);
  const barsContext = {};
  vm.runInNewContext(
    `${barsLogic}\nglobalThis.buildBars = buildAllFeatureMaeChangeBars;`,
    barsContext,
  );
  assert.deepEqual(
    JSON.parse(
      JSON.stringify(
        barsContext.buildBars([
          { feature_name: "age", mae: 0.2, baseline_mae: 0.5 },
          { feature_name: "month", mae: 0.8, baseline_mae: 0.5 },
        ]),
      ),
    ),
    [
      { featureName: "age", maeChange: -0.3 },
      { featureName: "month", maeChange: 0.3 },
    ],
  );
  assert.match(componentSource, /barmode: "group"/);
  assert.match(componentSource, /pattern:\s*\{\s*shape: "\/"/);
  assert.ok(
    componentSource.indexOf("<AllFeatureMaeChangeCharts") <
      componentSource.indexOf('t("modelCompare.chartControls")'),
  );
  assert.ok(
    componentSource.indexOf("</section>\n\n            <AllFeatureMaeChangeCharts") >
      componentSource.indexOf('t("modelCompare.averagePredictedScore")'),
  );
});

test("aggregated MAE charts shorten feature labels while retaining full hover names", () => {
  const labelLogic = componentSource.match(
    /const formatMaeFeatureLabel = \(featureName\) =>[\s\S]*?;/,
  )?.[0];
  assert.ok(labelLogic);

  const labelContext = {};
  vm.runInNewContext(
    `${labelLogic}\nglobalThis.formatLabel = formatMaeFeatureLabel;`,
    labelContext,
  );
  assert.equal(
    labelContext.formatLabel("Digital Competencies – Computer- und Tabletkompetenz"),
    "Computer- und Tabletkompetenz",
  );
  assert.equal(labelContext.formatLabel("Category - Feature"), "Feature");
  assert.equal(
    labelContext.formatLabel("Computer- und Tabletkompetenz"),
    "Computer- und Tabletkompetenz",
  );

  const chartSection = componentSource.match(
    /const FeatureMaeChangeChart = \([\s\S]*?\n};/,
  )?.[0];
  assert.ok(chartSection);
  assert.match(
    chartSection,
    /y: item\.bars\.map\(\(bar\) => formatMaeFeatureLabel\(bar\.featureName\)\)/,
  );
  assert.match(
    chartSection,
    /customdata: item\.bars\.map\(\(bar\) => bar\.featureName\)/,
  );
  assert.match(chartSection, /%\{customdata\}<br>/);
});

test("expert MAE-change bars display signed deltas with a detailed x-axis", () => {
  const xAxisScaleLogic = componentSource.match(
    /const getDetailedMaeDeltaXAxis = \([\s\S]*?\n};/,
  );
  assert.ok(
    xAxisScaleLogic,
    "the MAE-change chart must calculate a deliberately detailed x-axis scale",
  );

  const scaleContext = {};
  vm.runInNewContext(
    `${xAxisScaleLogic[0]}\nglobalThis.getXAxis = getDetailedMaeDeltaXAxis;`,
    scaleContext,
  );
  assert.deepEqual(
    JSON.parse(
      JSON.stringify(
        scaleContext.getXAxis([
          { maeChange: -0.3 },
          { maeChange: 0.3 },
        ]),
      ),
    ),
    { range: [-0.4, 0.4], dtick: 0.05, tick0: 0, tickformat: "+.3~f" },
  );

  assert.match(componentSource, /\.\.\.getDetailedMaeDeltaXAxis\(\[\.\.\.bars, \.\.\.confidenceBars\]\)/);
  assert.match(
    componentSource,
    /text:\s*traceBars\.map\(\(bar\) => formatMaeDeltaLabel\(bar\.maeChange\)\)/,
  );
  assert.match(componentSource, /texttemplate:\s*"%\{text\}"/);
  assert.match(componentSource, /textposition:\s*"outside"/);
  assert.match(componentSource, /cliponaxis:\s*false/);
  assert.match(componentSource, /:\s*%\{x:\+\.3~f\}<extra><\/extra>/);
});

test("expert MAE-delta labels are rounded to at most three decimal places", () => {
  const labelLogic = componentSource.match(
    /const formatMaeDeltaLabel = \([\s\S]*?\n};/,
  );
  assert.ok(labelLogic, "the chart must explicitly format its bar labels");

  const labelContext = {};
  vm.runInNewContext(
    `${labelLogic[0]}\nglobalThis.formatLabel = formatMaeDeltaLabel;`,
    labelContext,
  );
  assert.equal(labelContext.formatLabel(0.12349), "+0.123");
  assert.equal(labelContext.formatLabel(-0.12351), "-0.124");
  assert.equal(labelContext.formatLabel(0.1), "+0.1");
});

test("comparison selects main-page training data when the original test set is absent", () => {
  assert.match(componentSource, /<option value="main_training"/);
  assert.doesNotMatch(componentSource, /<option value="right"/);
  const sourceLogic = componentSource.match(
    /const getInitialHeldOutTestSource = \([\s\S]*?\n};/,
  );
  assert.ok(
    sourceLogic,
    "comparison must select an available held-out test set after preparation",
  );

  const sourceContext = {};
  vm.runInNewContext(
    `${sourceLogic[0]}\nglobalThis.getSource = getInitialHeldOutTestSource;`,
    sourceContext,
  );
  assert.equal(
    sourceContext.getSource({}, true),
    "main_training",
  );
  assert.equal(
    sourceContext.getSource({
      held_out_test_data: {
        feature_rows: [{ age: 20 }],
        target_values: [1.2],
      },
    }, true),
    "left",
  );
  assert.equal(sourceContext.getSource({}, false), "left");
  assert.match(
    componentSource,
    /setHeldOutTestSource\(\s*getInitialHeldOutTestSource\(\s*result\.left_artifact,\s*result\.preview\?\.main_training_performance_available,\s*\),?\s*\)/,
  );
});

test("comparison charts label German-spoken-at-home responses instead of 1–3", () => {
  assert.match(
    componentSource,
    /const GERMAN_SPOKEN_AT_HOME_SCALE_TICKS = \[[\s\S]*?\{ value: 1, label: "Never" \}[\s\S]*?\{ value: 2, label: "Sometimes<br>\(bilingual\)" \}[\s\S]*?\{ value: 3, label: "Always" \}/,
  );
  assert.match(componentSource, /isGermanSpokenAtHomeFeature\(featurePreview\)/);
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

test("overall shape-function MAE can expand each expert's complete-model MAE", () => {
  assert.equal(
    createTranslator("en")("modelCompare.expertCompleteModelMae"),
    "Expert complete-model MAE",
  );
  assert.match(componentSource, /const ExpertCompleteModelMaeList = \(/);
  assert.match(componentSource, /expert_complete_model_mae/);
  assert.match(componentSource, /formatChartExpertLabel\(expert\.user_name\)/);
  assert.match(componentSource, /modelCompare\.expertCompleteModelMae/);
  const expertList = componentSource.match(
    /const ExpertCompleteModelMaeList = \([\s\S]*?\n};/,
  )[0];
  assert.match(expertList, /<FeatureMetricBadge/);
  assert.match(expertList, /baselineValue=\{expert\.baseline_mae\}/);
  assert.match(expertList, /status=\{expert\.status\}/);
});

test("comparison page reports three-model held-out accuracy below shape-function MAE", () => {
  assert.equal(
    createTranslator("en")("modelCompare.heldOutTestPerformance"),
    "Held-out test performance",
  );
  assert.equal(createTranslator("en")("modelCompare.effectOfExpertEdits"), "Effect of expert edits");
  assert.equal(createTranslator("en")("modelCompare.fullData"), "Full-data benchmark");
  assert.match(componentSource, /const HeldOutTestPerformanceSummary = \(/);
  assert.match(componentSource, /held_out_test_performance/);
  assert.match(componentSource, /modelCompare\.heldOutTestUnavailable/);
  assert.ok(
    componentSource.indexOf("<OverallShapeFunctionMaeSummary") <
      componentSource.indexOf("<HeldOutTestPerformanceSummary"),
    "the held-out segment belongs directly below overall shape-function MAE",
  );
});

test("held-out comparison descriptions name the models selected in the plot", () => {
  const t = createTranslator("en");
  assert.equal(t("modelCompare.reducedToEdited"), "Original reduced-data → edited reduced-data");
  assert.equal(t("modelCompare.fullToReduced"), "Full-data benchmark → original reduced-data model");
  assert.equal(t("modelCompare.fullToEdited"), "Full-data benchmark → edited reduced-data model");
  assert.equal(t("modelCompare.allThreeComparison"), "Full-data benchmark → original reduced-data model → edited reduced-data model");
});

test("held-out performance shows prediction standard deviation for all three models", () => {
  const summary = componentSource.match(/const HeldOutTestPerformanceSummary = \([\s\S]*?\n};/)[0];
  assert.equal(createTranslator("en")("modelCompare.metric.std_dev"), "Prediction standard deviation");
  assert.match(summary, /\["fullData", performance\.baseline\]/);
  assert.match(summary, /\["reducedData", performance\.reduced\]/);
  assert.match(summary, /\["editedReducedData", performance\.edited\]/);
  assert.match(summary, /"std_dev"/);
});

test("held-out performance shows R2 and signed mean error for all three models", () => {
  assert.equal(createTranslator("en")("modelCompare.heldOutMeanError"), "Mean error");
  const summary = componentSource.match(
    /const HeldOutTestPerformanceSummary = \([\s\S]*?\n};/,
  )[0];
  assert.match(summary, /"r2", "mean_error", "std_dev"/);
  assert.match(summary, /formatSignedMetricValue\(metrics\?\.\[key\], language\)/);
  assert.match(summary, /getHeldOutMetricComparison\(performance, selectedComparison, key\)/);
  assert.match(summary, /modelCompare\.meanErrorConvention/);
});

test("held-out performance displays actual-versus-predicted points below its errors", () => {
  assert.equal(
    createTranslator("en")("modelCompare.actualVsPredicted"),
    "Actual vs. predicted",
  );
  assert.match(componentSource, /const HeldOutActualVsPredictedChart = \(/);
  assert.match(componentSource, /actual_values/);
  assert.match(componentSource, /baseline_predictions/);
  assert.match(componentSource, /reduced_predictions/);
  assert.match(componentSource, /edited_predictions/);
  assert.match(componentSource, /modelCompare\.actualVsPredictedIdentityLine/);
  const chart = componentSource.match(/const HeldOutActualVsPredictedChart = \([\s\S]*?\n};/)[0];
  const summary = componentSource.match(/const HeldOutTestPerformanceSummary = \([\s\S]*?\n};/)[0];
  assert.match(summary, /useState\("reducedEdited"\)/);
  assert.match(summary, /Object\.keys\(HELD_OUT_COMPARISONS\)/);
  assert.match(summary, /onClick=\{\(\) => setSelectedComparison\(option\)\}/);
  assert.match(summary, /comparison\.descriptionKey/);
  assert.match(summary, /comparison\.headingKey/);
  assert.match(summary, /visibleModels=\{comparison\.modelKeys\}/);
  assert.match(chart, /visibleModels\.map\(\(modelKey\) =>/);
  assert.match(chart, /renderChart\(\s*\[identityTrace, modelTraces\[modelKey\]/);
  assert.match(chart, /lg:grid-cols-2/);
  assert.match(chart, /xl:grid-cols-3/);
  assert.match(chart, /pair\.predicted - pair\.actual/);
});

test("actual-versus-predicted chart uses a fixed x-axis and spans observed y values", () => {
  const rangeLogic = componentSource.match(
    /const getActualVsPredictedRange = \([\s\S]*?\n};/,
  );
  assert.ok(rangeLogic, "the plot needs a dedicated dynamic range calculation");
  const rangeContext = {};
  vm.runInNewContext(
    `${rangeLogic[0]}\nglobalThis.getRange = getActualVsPredictedRange;`,
    rangeContext,
  );

  assert.deepEqual(
    JSON.parse(JSON.stringify(rangeContext.getRange([-437, 942]))),
    { range: [-500, 1000], dtick: 500, tick0: 0 },
  );
  const actualVsPredictedChart = componentSource.match(
    /const HeldOutActualVsPredictedChart = \([\s\S]*?\n};/,
  )[0];
  assert.match(actualVsPredictedChart, /\.\.\.yAxisScale/);
  assert.match(actualVsPredictedChart, /const xAxisRange = \[0, 800\]/);
  assert.match(actualVsPredictedChart, /range: xAxisRange/);
  assert.doesNotMatch(actualVsPredictedChart, /scaleanchor|scaleratio|constrain/);
});

test("held-out evaluation can use the main-page model artifact test set", () => {
  assert.equal(
    createTranslator("en")("modelCompare.useOriginalArtifactTestSet"),
    "Full-data held-out test set",
  );
  assert.match(componentSource, /heldOutTestSource/);
  assert.match(componentSource, /held_out_test_source/);
  assert.match(componentSource, /modelCompare\.useOriginalArtifactTestSet/);
  assert.match(componentSource, /left_artifact\?\.held_out_test_data/);
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
    /const OverallShapeFunctionMaeSummary = \([\s\S]*?const getActualVsPredictedRange/,
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

test("average contribution accordion can filter features", () => {
  assert.match(componentSource, /contributionFeatureSelection/);
  assert.match(componentSource, /handleContributionFeatureToggle/);
  assert.match(componentSource, /selectedContributionScoreRows/);
});

test("average prediction comparison offers the original held-out test rows", () => {
  assert.equal(
    createTranslator("en")("modelCompare.averagePredictedScore"),
    "Average prediction comparison",
  );
  assert.equal(
    createTranslator("en")("modelCompare.editedAveragePredictedScore"),
    "Edited artifact on selected reference",
  );
  assert.equal(
    createTranslator("en")("modelCompare.testSetAverage"),
    "Original held-out test set",
  );
  assert.match(componentSource, /test_set_average_prediction_scores/);
  assert.match(componentSource, /setAverageScoreMethod\("test_set"\)/);
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

test("average prediction comparison shows edited minus unedited reduced model", () => {
  assert.equal(
    createTranslator("en")("modelCompare.editedMinusReducedAveragePrediction"),
    "Average prediction difference: edited reduced model − unedited reduced model",
  );
  assert.match(componentSource, /modelCompare\.editedMinusReducedAveragePrediction/);
  assert.match(componentSource, /selectedContributionAverageScores\?\.edited_minus_reduced/);
  assert.match(componentSource, /scores\.edited_minus_reduced \+ row\.edited - row\.reduced/);
});

test("average comparison toggles training means and intercept correction details", () => {
  assert.equal(
    createTranslator("en")("modelCompare.trainingMeanDetailsTitle"),
    "Training mean and intercept values",
  );
  const details = componentSource.match(/const TrainingMeanDetails = \([\s\S]*?\n};/)?.[0];
  assert.ok(details);
  assert.match(details, /<details /);
  assert.match(details, /<summary /);
  assert.match(details, /summary\.reduced_model_mean/);
  assert.match(details, /summary\.edited_before_correction_mean/);
  assert.match(details, /summary\.difference_before_correction/);
  assert.match(details, /summary\.edited_after_correction_mean/);
  assert.match(details, /summary\.suggested_matching_adjustment/);
  assert.match(details, /summary\.suggested_matching_intercept/);
  assert.match(details, /modelCompare\.trainingMeanUnavailable/);
  assert.match(componentSource, /summary=\{preview\?\.training_mean_prediction_summary\}/);
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

test("contribution breakdown shows feature names and prediction changes", () => {
  const breakdown = componentSource.match(
    /<div className="w-full rounded-xl border border-indigo-200 bg-white px-4 py-3">[\s\S]*?<\/table>/,
  )[0];
  assert.equal(
    createTranslator("en")("modelCompare.averageContributionsTitle"),
    "Change in average prediction by feature",
  );
  assert.match(breakdown, /modelCompare\.averageContributionsTitle/);
  assert.match(breakdown, /<h3 className="text-base font-semibold text-indigo-950">/);
  assert.doesNotMatch(breakdown, /<summary|<details|averageContributionsDescription/);
  assert.match(componentSource, /<div className="mt-4 grid gap-4 lg:grid-cols-2">/);
  assert.doesNotMatch(breakdown, /modelCompare\.contributionFeatureSelection|selectedContributionDifference/);
  assert.match(breakdown, /min-w-\[520px\]/);
  assert.match(breakdown, /<td className="whitespace-nowrap px-2 py-2 font-medium">/);
  assert.match(breakdown, /modelCompare\.contributionFeature/);
  assert.equal(
    createTranslator("en")("modelCompare.meanPredictionChangeColumn"),
    "Mean prediction change (edited − original)",
  );
  assert.match(breakdown, /modelCompare\.meanPredictionChangeColumn/);
  assert.doesNotMatch(breakdown, /modelCompare\.averagePredictedScoreDifference/);
  assert.doesNotMatch(breakdown, /modelCompare\.(originalAveragePredictedScore|editedAveragePredictedScore)/);
  assert.doesNotMatch(breakdown, /row\.(original|edited), language/);
  assert.doesNotMatch(breakdown, /modelCompare\.basePredictionOffset/);
  assert.match(componentSource, /const displayedContributionRows = selectedContributionRows\.filter\(\s*\(row\) => !row\.is_base_offset/);
  assert.match(componentSource, /selectedContributionScoreRows = selectedContributionRows\.filter\(\s*\(row\) =>\s*row\.is_base_offset/);
  assert.doesNotMatch(breakdown, /featureEditShift|manualFeatureShift|matchAveragePrediction/);
});

test("contribution breakdown footer shows the full-model mean deviation", () => {
  assert.equal(
    createTranslator("en")("modelCompare.totalMeanDeviationFromOriginal"),
    "Total mean deviation from original full-data model",
  );
  const breakdown = componentSource.match(
    /<div className="w-full rounded-xl border border-indigo-200 bg-white px-4 py-3">[\s\S]*?<\/table>/,
  )[0];
  const footer = breakdown.match(/<tfoot[\s\S]*?<\/tfoot>/)[0];
  assert.match(footer, /modelCompare\.totalMeanDeviationFromOriginal/);
  assert.match(footer, /averagePredictionScores\?\.difference/);
  assert.doesNotMatch(footer, /selectedContributionAverageScores/);
  assert.match(footer, /formatSignedMetricValue/);
});

test("reduced-model contribution breakdown shows edited reduced minus original reduced", () => {
  assert.equal(
    createTranslator("en")("modelCompare.reducedEditedContributionsTitle"),
    "Change in average prediction by feature: edited reduced − original reduced model",
  );
  assert.equal(
    createTranslator("en")("modelCompare.reducedEditedMeanPredictionChangeColumn"),
    "Mean prediction change (edited reduced − original reduced)",
  );
  const breakdown = componentSource.match(
    /<div className="w-full rounded-xl border border-teal-200 bg-white px-4 py-3">[\s\S]*?<\/table>/,
  )[0];
  assert.match(breakdown, /modelCompare\.reducedEditedContributionsTitle/);
  assert.match(breakdown, /displayedReducedEditedContributionRows\.map/);
  assert.match(breakdown, /row\.edited - row\.reduced/);
  assert.match(breakdown, /handleContributionFeatureToggle/);
  assert.match(breakdown, /modelCompare\.totalMeanDeviationFromOriginalReduced/);
  assert.match(breakdown, /averagePredictionScores\.edited - averagePredictionScores\.reduced/);
});

test("each contribution table orders features by its own absolute prediction shift", () => {
  assert.match(componentSource, /return Math\.abs\(right\.difference\) - Math\.abs\(left\.difference\);/);
  const orderingLogic = componentSource.match(
    /const displayedReducedEditedContributionRows = \[[\s\S]*?\n  \);/,
  );
  assert.ok(orderingLogic);
  const rows = [
    { feature_name: "small", reduced: 5, edited: 4 },
    { feature_name: "medium", reduced: 5, edited: 2 },
    { feature_name: "large", reduced: 2, edited: 8 },
  ];
  const orderingContext = { displayedContributionRows: rows };
  vm.runInNewContext(
    `${orderingLogic[0]}\nglobalThis.sortedRows = displayedReducedEditedContributionRows;`,
    orderingContext,
  );
  assert.deepEqual(
    Array.from(orderingContext.sortedRows, (row) => row.feature_name),
    ["large", "medium", "small"],
  );
  assert.deepEqual(rows.map((row) => row.feature_name), ["small", "medium", "large"]);
});

test("comparison exposes a preview-only numeric intercept adjustment", () => {
  assert.equal(
    createTranslator("en")("modelCompare.interceptAdjustmentLabel"),
    "Edited model intercept adjustment",
  );
  assert.match(componentSource, /onSubmit=\{handleInterceptAdjustmentSubmit\}/);
  assert.match(componentSource, /type="number"\s+step="any"\s+required/);
  assert.match(componentSource, /intercept_adjustment: nextInterceptAdjustment/);
  assert.match(componentSource, /modelCompare\.originalEditedBaseOffset/);
  assert.match(componentSource, /modelCompare\.adjustedEditedBaseOffset/);
});
