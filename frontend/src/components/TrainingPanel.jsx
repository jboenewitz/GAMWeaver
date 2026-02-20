import React, { useState } from "react";

const TrainingPanel = ({ onLoadData, onTrainModel, loading, modelStatus }) => {
  const [nEstimators, setNEstimators] = useState(100);
  const [trainingProgress, setTrainingProgress] = useState(null);

  const handleLoadData = async () => {
    setTrainingProgress("Loading data...");
    await onLoadData();
    setTrainingProgress(null);
  };

  const handleTrain = async () => {
    setTrainingProgress("Training model... This may take a moment.");
    await onTrainModel({ n_estimators: nEstimators });
    setTrainingProgress(null);
  };

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">
        Model Training
      </h3>

      <div className="space-y-4">
        {/* Step 1: Load Data */}
        <div className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                1
              </span>
              <span className="font-medium text-gray-700">Load Data</span>
            </div>
            {modelStatus?.data_loaded && (
              <span className="text-green-600 text-sm flex items-center gap-1">
                <svg
                  className="w-4 h-4"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                Loaded
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mb-3">
            Load a dataset for training.
          </p>
          <button
            onClick={handleLoadData}
            disabled={loading}
            className="btn-secondary text-sm"
          >
            {loading ? "Loading..." : "Load Dataset"}
          </button>
        </div>

        {/* Step 2: Configure & Train */}
        <div className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                2
              </span>
              <span className="font-medium text-gray-700">Train Model</span>
            </div>
            {modelStatus?.is_trained && (
              <span className="text-green-600 text-sm flex items-center gap-1">
                <svg
                  className="w-4 h-4"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                Trained
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mb-3">
            Configure and train the IGANN model.
          </p>

          <div className="mb-3">
            <label className="label">Number of Estimators</label>
            <input
              type="number"
              value={nEstimators}
              readOnly
              className="input-field bg-gray-100 cursor-not-allowed"
              min="10"
              max="500"
              step="10"
            />
            <p className="text-xs text-gray-400 mt-1">
              More estimators = better accuracy but slower training
            </p>
          </div>

          <button
            onClick={handleTrain}
            disabled={loading || !modelStatus?.data_loaded}
            className="btn-primary text-sm w-full disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Training..." : "Train IGANN Model"}
          </button>
        </div>

        {/* Progress indicator */}
        {trainingProgress && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <svg
                className="animate-spin h-4 w-4 text-blue-600"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              <span className="text-sm text-blue-700">{trainingProgress}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingPanel;
