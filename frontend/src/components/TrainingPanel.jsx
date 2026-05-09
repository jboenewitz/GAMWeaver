import React, { useMemo, useState } from "react";

const UploadDatasetModal = ({
  isOpen,
  onClose,
  onInspectUpload,
  onUploadDataset,
  currentUser,
}) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState(null);
  const [inspecting, setInspecting] = useState(false);
  const [uploading, setUploading] = useState(false);

  const numericColumns = useMemo(
    () => (inspection?.columns || []).filter((column) => column.is_numeric),
    [inspection],
  );

  if (!isOpen) {
    return null;
  }

  const resetModal = () => {
    setSelectedFile(null);
    setInspection(null);
    setTargetColumn("");
    setDisplayName("");
    setError(null);
    setInspecting(false);
    setUploading(false);
    onClose();
  };

  const handleInspect = async () => {
    if (!selectedFile) {
      setError("Choose a CSV file first.");
      return;
    }

    try {
      setInspecting(true);
      setError(null);
      const result = await onInspectUpload(selectedFile);
      setInspection(result);
      const firstNumeric = (result.columns || []).find((column) => column.is_numeric);
      setTargetColumn(firstNumeric?.name || "");
      setDisplayName(selectedFile.name.replace(/\.csv$/i, ""));
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to inspect dataset.");
    } finally {
      setInspecting(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Choose a CSV file first.");
      return;
    }
    if (!targetColumn) {
      setError("Choose a numeric target column before uploading.");
      return;
    }

    try {
      setUploading(true);
      setError(null);
      await onUploadDataset({
        file: selectedFile,
        targetColumn,
        displayName,
        uploadedByUserId: currentUser?.id,
      });
      resetModal();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Failed to upload dataset.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-2xl p-6 space-y-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">Upload Dataset</h3>
            <p className="text-sm text-gray-500">
              Step 1: inspect the CSV. Step 2: choose the numeric target column and upload it.
            </p>
          </div>
          <button
            onClick={resetModal}
            className="text-gray-500 hover:text-gray-700 text-xl leading-none"
          >
            ×
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="label">CSV File</label>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setSelectedFile(file);
                setInspection(null);
                setTargetColumn("");
                setError(null);
              }}
              className="input-field"
            />
          </div>

          <div className="flex justify-end">
            <button
              onClick={handleInspect}
              disabled={!selectedFile || inspecting || uploading}
              className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {inspecting ? "Inspecting..." : "Inspect File"}
            </button>
          </div>

          {inspection ? (
            <div className="border border-gray-200 rounded-lg p-4 space-y-4">
              <div className="text-sm text-gray-600">
                Detected {inspection.columns?.length || 0} columns across{" "}
                {inspection.row_count?.toLocaleString()} rows.
              </div>

              <div>
                <label className="label">Dataset Name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  className="input-field"
                  placeholder="Dataset name"
                />
              </div>

              <div>
                <label className="label">Target Column</label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="select-field"
                >
                  <option value="">Choose target column</option>
                  {numericColumns.map((column) => (
                    <option key={column.name} value={column.name}>
                      {column.name}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Only numeric columns can be used as the regression target.
                </p>
              </div>

              <div className="max-h-56 overflow-y-auto border rounded-lg">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-600 sticky top-0">
                    <tr>
                      <th className="text-left px-3 py-2">Column</th>
                      <th className="text-left px-3 py-2">Type</th>
                      <th className="text-left px-3 py-2">Sample Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(inspection.columns || []).map((column) => (
                      <tr key={column.name} className="border-t">
                        <td className="px-3 py-2 font-medium text-gray-800">
                          {column.name}
                        </td>
                        <td className="px-3 py-2 text-gray-600">
                          {column.is_numeric ? "Numeric" : "Categorical"}
                        </td>
                        <td className="px-3 py-2 text-gray-500">
                          {(column.sample_values || []).join(", ") || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          {error ? (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
              {error}
            </div>
          ) : null}
        </div>

        <div className="flex items-center justify-end gap-3">
          <button
            onClick={resetModal}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={!inspection || !targetColumn || uploading}
            className="btn-primary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? "Uploading..." : "Upload Dataset"}
          </button>
        </div>
      </div>
    </div>
  );
};

const TrainingPanel = ({
  currentUser,
  datasets = [],
  activeDatasetId,
  activeDatasetName,
  onSelectDataset,
  onInspectUpload,
  onUploadDataset,
  onTrainModel,
  loading,
  modelStatus,
}) => {
  const [nEstimators] = useState(100);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [chooseOpen, setChooseOpen] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);

  const handleSelectDataset = async (datasetId) => {
    setTrainingProgress("Loading selected dataset...");
    await onSelectDataset(datasetId);
    setTrainingProgress(null);
    setChooseOpen(false);
  };

  const handleTrain = async () => {
    setTrainingProgress("Training model... This may take a moment.");
    await onTrainModel({ n_estimators: nEstimators });
    setTrainingProgress(null);
  };

  return (
    <>
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Dataset and Model
        </h3>

        <div className="space-y-4">
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                  1
                </span>
                <span className="font-medium text-gray-700">Dataset</span>
              </div>
              {modelStatus?.data_loaded && (
                <span className="text-green-600 text-sm flex items-center gap-1">
                  Loaded
                </span>
              )}
            </div>
            <p className="text-sm text-gray-500 mb-3">
              Choose an uploaded dataset for your session. Superadmins can also upload new CSV datasets.
            </p>

            <div className="space-y-3">
              <div className="text-sm text-gray-600">
                Active dataset:{" "}
                <span className="font-medium text-gray-800">
                  {activeDatasetName || "None selected"}
                </span>
              </div>

              <div className="flex flex-wrap gap-2">
                {currentUser?.is_superadmin ? (
                  <button
                    onClick={() => setShowUploadModal(true)}
                    disabled={loading}
                    className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Upload Dataset
                  </button>
                ) : null}

                <button
                  onClick={() => setChooseOpen((prev) => !prev)}
                  disabled={loading || datasets.length === 0}
                  className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Choose Dataset
                </button>
              </div>

              {chooseOpen ? (
                <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                  {datasets.length === 0 ? (
                    <p className="text-sm text-gray-500">No uploaded datasets are available yet.</p>
                  ) : (
                    <select
                      value={activeDatasetId || ""}
                      onChange={(e) => handleSelectDataset(Number(e.target.value))}
                      className="select-field"
                    >
                      {datasets.map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>
                          {dataset.display_name}
                          {dataset.latest_model_version_number
                            ? ` (v${dataset.latest_model_version_number})`
                            : " (untrained)"}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              ) : null}
            </div>
          </div>

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
                  Trained
                </span>
              )}
            </div>
            <p className="text-sm text-gray-500 mb-3">
              Train a new IGANN model version for the currently selected dataset.
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
                More estimators can improve fit but also increase training time.
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

          {trainingProgress ? (
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
          ) : null}
        </div>
      </div>

      <UploadDatasetModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onInspectUpload={onInspectUpload}
        onUploadDataset={onUploadDataset}
        currentUser={currentUser}
      />
    </>
  );
};

export default TrainingPanel;
