import assert from "node:assert/strict";
import { test } from "node:test";
import { getAveragePredictionScores } from "../src/utils/averagePredictionAdjustment.js";

test("average prediction comparison selects the requested reference scores", () => {
  const preview = {
    test_set_average_prediction_scores: {
      original: 6,
      edited: 4,
      difference: -2,
    },
    main_dataset_average_prediction_scores: {
      original: 8,
      edited: 9,
      difference: 1,
    },
  };

  assert.deepEqual(
    getAveragePredictionScores(preview, "test_set"),
    preview.test_set_average_prediction_scores,
  );
  assert.deepEqual(
    getAveragePredictionScores(preview, "main_dataset"),
    preview.main_dataset_average_prediction_scores,
  );
});
