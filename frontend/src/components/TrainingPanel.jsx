import React, { useState } from "react";

const TrainingPanel = ({
  onUploadDataset,
  onLoadData,
  onTrainModel,
  loading,
  modelStatus,
  isSuperadmin,
}) => {
  const [nEstimators, setNEstimators] = useState(100);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [modalError, setModalError] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [selectedFeatureColumns, setSelectedFeatureColumns] = useState([]);

  const resetModal = () => {
    setSelectedFile(null);
    setUploading(false);
    setModalError(null);
    setUploadPreview(null);
    setTargetColumn("");
    setSelectedFeatureColumns([]);
  };

  const openModal = () => {
    resetModal();
    setShowUploadModal(true);
  };

  const closeModal = () => {
    setShowUploadModal(false);
    resetModal();
  };

  const handleInspectUpload = async () => {
    if (!selectedFile) {
      setModalError("Please choose a CSV file first.");
      return;
    }

    try {
      setUploading(true);
      setModalError(null);
      const preview = await onUploadDataset(selectedFile);
      const defaultTarget =
        preview.default_target_column || preview.columns?.[0] || "";
      setUploadPreview(preview);
      setTargetColumn(defaultTarget);
      setSelectedFeatureColumns(
        (preview.columns || []).filter((column) => column !== defaultTarget),
      );
    } catch (err) {
      setModalError(err.message || "Failed to upload dataset");
    } finally {
      setUploading(false);
    }
  };

  const handleLoadUploadedDataset = async () => {
    if (!uploadPreview?.dataset_id || !targetColumn || !selectedFeatureColumns.length) {
      setModalError(
        "Please upload a CSV, choose a target column, and select at least one feature column.",
      );
      return;
    }

    try {
      setUploading(true);
      setTrainingProgress("Loading data...");
      await onLoadData({
        dataset_id: uploadPreview.dataset_id,
        target_column: targetColumn,
        feature_columns: selectedFeatureColumns,
      });
      setTrainingProgress(null);
      closeModal();
    } catch (err) {
      setModalError(err.message || "Failed to load dataset");
      setTrainingProgress(null);
    } finally {
      setUploading(false);
    }
  };

  const handleTargetColumnChange = (nextTarget) => {
    setTargetColumn(nextTarget);
    setSelectedFeatureColumns((prev) =>
      prev.filter((column) => column !== nextTarget),
    );
  };

  const toggleFeatureColumn = (columnName) => {
    if (columnName === targetColumn) return;
    setSelectedFeatureColumns((prev) => {
      if (prev.includes(columnName)) {
        return prev.filter((column) => column !== columnName);
      }
      return [...prev, columnName];
    });
  };

  const selectAllFeatureColumns = () => {
    setSelectedFeatureColumns(
      (uploadPreview?.columns || []).filter((column) => column !== targetColumn),
    );
  };

  const clearAllFeatureColumns = () => {
    setSelectedFeatureColumns([]);
  };

  const handleTrain = async () => {
    setTrainingProgress("Training model... This may take a moment.");
    await onTrainModel({ n_estimators: nEstimators });
    setTrainingProgress(null);
  };

  if (!isSuperadmin) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">Model Training</h3>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          Dataset upload/loading and training controls are restricted to the superadmin.
        </div>
        <div className="mt-4 space-y-2 text-sm text-gray-600">
          <div>Data loaded: {modelStatus?.data_loaded ? "Yes" : "No"}</div>
          <div>Model trained: {modelStatus?.is_trained ? "Yes" : "No"}</div>
          {modelStatus?.dataset_name && <div>Active dataset: {modelStatus.dataset_name}</div>}
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">Model Training</h3>

      <div className="space-y-4">
        <div className="border border-gray-200 rounded-xl p-4 bg-white/40">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                1
              </span>
              <span className="font-medium text-gray-700">Load Data</span>
            </div>
            {modelStatus?.data_loaded && (
              <span className="text-green-600 text-sm flex items-center gap-1">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
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
            Upload a CSV dataset, select the prediction target, and choose which columns to import.
          </p>
          <button onClick={openModal} disabled={loading} className="btn-secondary text-sm">
            {loading ? "Loading..." : "Upload & Load Dataset"}
          </button>
          {modelStatus?.dataset_name && (
            <p className="text-xs text-gray-500 mt-3">Active dataset: {modelStatus.dataset_name}</p>
          )}
        </div>

        <div className="border border-gray-200 rounded-xl p-4 bg-white/40">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                2
              </span>
              <span className="font-medium text-gray-700">Train Model</span>
            </div>
            {modelStatus?.is_trained && (
              <span className="text-green-600 text-sm flex items-center gap-1">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
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
              onChange={(event) => {
                const next = parseInt(event.target.value || "100", 10);
                setNEstimators(Number.isFinite(next) ? next : 100);
              }}
              className="input-field"
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

      {showUploadModal && (
        <div className="glass-overlay fixed inset-0 flex items-center justify-center z-50 p-4">
          <div className="glass-modal rounded-2xl w-full max-w-xl">
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <h4 className="text-lg font-semibold text-gray-800">Upload Dataset</h4>
              <button onClick={closeModal} className="text-gray-500 hover:text-gray-700">
                ✕
              </button>
            </div>

            <div className="px-6 py-5 space-y-4">
              {modalError && (
                <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {modalError}
                </div>
              )}

              <div>
                <label className="label">CSV File</label>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => {
                    const file = event.target.files?.[0] || null;
                    setSelectedFile(file);
                    setUploadPreview(null);
                    setTargetColumn("");
                    setModalError(null);
                  }}
                  className="block w-full text-sm text-gray-700 file:mr-4 file:px-4 file:py-2 file:rounded-xl file:border-0 file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                />
              </div>

              <button
                onClick={handleInspectUpload}
                disabled={!selectedFile || uploading}
                className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? "Uploading..." : "Upload & Inspect Columns"}
              </button>

              {uploadPreview && (
                <div className="space-y-3 rounded-xl border border-gray-200 bg-gray-50 p-3">
                  <div className="text-sm text-gray-700">
                    <span className="font-medium">Columns detected:</span> {uploadPreview.columns.length}
                  </div>
                  <div>
                    <label className="label">Target Column</label>
                    <select
                      value={targetColumn}
                      onChange={(event) => handleTargetColumnChange(event.target.value)}
                      className="select-field"
                    >
                      {uploadPreview.columns.map((column) => (
                        <option key={column} value={column}>
                          {column}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="label mb-0">Feature Columns</label>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={selectAllFeatureColumns}
                          className="text-xs text-primary-700 hover:underline"
                        >
                          Select all
                        </button>
                        <button
                          type="button"
                          onClick={clearAllFeatureColumns}
                          className="text-xs text-gray-600 hover:underline"
                        >
                          Clear all
                        </button>
                      </div>
                    </div>
                    <div className="max-h-40 overflow-y-auto rounded-xl border border-gray-200 bg-white p-2 space-y-1">
                      {uploadPreview.columns.map((column) => {
                        const isTarget = column === targetColumn;
                        const checked =
                          !isTarget && selectedFeatureColumns.includes(column);
                        return (
                          <label
                            key={column}
                            className={`flex items-center justify-between gap-2 px-2 py-1 rounded-lg ${
                              isTarget ? "bg-amber-50 text-amber-700" : "hover:bg-gray-50"
                            }`}
                          >
                            <span className="text-sm">{column}</span>
                            {isTarget ? (
                              <span className="text-xs font-medium">Target</span>
                            ) : (
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => toggleFeatureColumn(column)}
                              />
                            )}
                          </label>
                        );
                      })}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Selected features: {selectedFeatureColumns.length}
                    </p>
                  </div>
                  <div className="max-h-28 overflow-y-auto rounded-xl border border-gray-200 bg-white p-2 text-xs text-gray-600">
                    {uploadPreview.columns.join(", ")}
                  </div>
                </div>
              )}
            </div>

            <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
              <button
                onClick={closeModal}
                disabled={uploading}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleLoadUploadedDataset}
                disabled={
                  !uploadPreview ||
                  !targetColumn ||
                  !selectedFeatureColumns.length ||
                  uploading
                }
                className="px-4 py-2 btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? "Loading..." : "Load Dataset"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingPanel;
