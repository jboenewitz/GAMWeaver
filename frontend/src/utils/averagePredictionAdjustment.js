export const getAveragePredictionScores = (preview, method) =>
  method === "test_set"
    ? preview?.test_set_average_prediction_scores
    : preview?.main_dataset_average_prediction_scores;
