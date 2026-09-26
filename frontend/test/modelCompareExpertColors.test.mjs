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
