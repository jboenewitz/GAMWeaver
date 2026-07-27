import React, { useState } from "react";
import { createPortal } from "react-dom";
import { createTranslator } from "../i18n";

const TrainingPanel = ({
  onUploadDataset,
  onLoadData,
  onTrainModel,
  loading,
  modelStatus,
  isSuperadmin,
  language = "en",
}) => {
  const [nEstimators, setNEstimators] = useState(100);
  const [trainingProgressKey, setTrainingProgressKey] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [modalError, setModalError] = useState(null);
  const [uploadPreview, setUploadPreview] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [selectedFeatureColumns, setSelectedFeatureColumns] = useState([]);
  const [isFeatureSelectorExpanded, setIsFeatureSelectorExpanded] =
    useState(false);
  const t = createTranslator(language);

  const resetModal = () => {
    setSelectedFile(null);
    setUploading(false);
    setModalError(null);
    setUploadPreview(null);
    setTargetColumn("");
    setSelectedFeatureColumns([]);
    setIsFeatureSelectorExpanded(false);
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
      setModalError(t("training.error.chooseCsvFirst"));
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
      setModalError(err.message || t("training.error.uploadFailed"));
    } finally {
      setUploading(false);
    }
  };

  const handleLoadUploadedDataset = async () => {
    if (
      !uploadPreview?.dataset_id ||
      !targetColumn ||
      !selectedFeatureColumns.length
    ) {
      setModalError(t("training.error.uploadNeedsSelections"));
      return;
    }

    try {
      setUploading(true);
      setTrainingProgressKey("training.progress.loadingData");
      await onLoadData({
        dataset_id: uploadPreview.dataset_id,
        dataset_name: uploadPreview.original_filename,
        target_column: targetColumn,
        feature_columns: selectedFeatureColumns,
      });
      setTrainingProgressKey(null);
      closeModal();
    } catch (err) {
      setModalError(err.message || t("training.error.loadFailed"));
      setTrainingProgressKey(null);
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
      (uploadPreview?.columns || []).filter(
        (column) => column !== targetColumn,
      ),
    );
  };

  const clearAllFeatureColumns = () => {
    setSelectedFeatureColumns([]);
  };

  const toggleFeatureSelectorExpanded = () => {
    setIsFeatureSelectorExpanded((prev) => !prev);
  };

  const handleTrain = async () => {
    setTrainingProgressKey("training.progress.training");
    await onTrainModel({ n_estimators: nEstimators });
    setTrainingProgressKey(null);
  };

  if (!isSuperadmin) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          {t("training.title")}
        </h3>
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
          {t("training.restrictedNotice")}
        </div>
        <div className="mt-4 space-y-2 text-sm text-gray-600">
          <div>
            {t("training.statusDataLoaded")}:{" "}
            {modelStatus?.data_loaded ? t("common.yes") : t("common.no")}
          </div>
          <div>
            {t("training.statusModelTrained")}:{" "}
            {modelStatus?.is_trained ? t("common.yes") : t("common.no")}
          </div>
          <div>
            {t("training.statusModelSource")}:{" "}
            {modelStatus?.model_source === "imported"
              ? t("training.modelSourceImported")
              : t("training.modelSourceTrained")}
          </div>
          {modelStatus?.dataset_name && (
            <div>
              {t("training.activeDataset")}: {modelStatus.dataset_name}
            </div>
          )}
          {modelStatus?.model_source === "imported" &&
            !modelStatus?.analytics_available && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-amber-800">
                {t("training.importedAnalyticsUnavailable")}
              </div>
            )}
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">
        {t("training.title")}
      </h3>

      <div className="space-y-4">
        <div className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                1
              </span>
              <span className="font-medium text-gray-700">
                {t("training.stepLoadData")}
              </span>
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
                {t("training.stepLoaded")}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mb-3">
            {t("training.loadDescription")}
          </p>
          <button
            onClick={openModal}
            disabled={loading}
            className="btn-secondary text-sm"
          >
            {loading ? t("common.loading") : t("training.uploadAndLoad")}
          </button>
          {modelStatus?.dataset_name && (
            <p className="text-xs text-gray-500 mt-3">
              Active dataset: {modelStatus.dataset_name}
            </p>
          )}
        </div>

        <div className="border border-gray-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                2
              </span>
              <span className="font-medium text-gray-700">
                {t("training.stepTrainModel")}
              </span>
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
                {t("training.stepTrained")}
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mb-3">
            {t("training.trainDescription")}
          </p>

          <div className="mb-3">
            <label className="label">{t("training.estimators")}</label>
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
              {t("training.estimatorHint")}
            </p>
          </div>

          <button
            onClick={handleTrain}
            disabled={loading || !modelStatus?.data_loaded}
            className="btn-primary text-sm w-full disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? t("training.training") : t("training.trainButton")}
          </button>
        </div>

        {trainingProgressKey && (
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
              <span className="text-sm text-blue-700">
                {t(trainingProgressKey)}
              </span>
            </div>
          </div>
        )}
      </div>

      {showUploadModal &&
        createPortal(
          <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/50 p-4">
            <div
              className={`flex w-full flex-col overflow-hidden rounded-xl bg-white shadow-xl transition-all duration-200 ${
                isFeatureSelectorExpanded
                  ? "max-h-[calc(100vh-2rem)] max-w-[calc(100vw-2rem)]"
                  : "max-w-xl"
              }`}
            >
              <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
                <h4 className="text-lg font-semibold text-gray-800">
                  {t("training.uploadModalTitle")}
                </h4>
                <button
                  onClick={closeModal}
                  className="text-gray-500 hover:text-gray-700"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4 overflow-y-auto px-6 py-5">
                {modalError && (
                  <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {modalError}
                  </div>
                )}

                <div>
                  <label className="label">{t("training.csvFile")}</label>
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
                    className="block w-full text-sm text-gray-700 file:mr-4 file:px-4 file:py-2 file:rounded-lg file:border-0 file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                  />
                </div>

                <button
                  onClick={handleInspectUpload}
                  disabled={!selectedFile || uploading}
                  className="btn-secondary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploading
                    ? t("training.uploading")
                    : t("training.uploadInspect")}
                </button>

                {uploadPreview && (
                  <div
                    className={`rounded-lg border border-gray-200 bg-gray-50 p-3 ${
                      isFeatureSelectorExpanded
                        ? "grid gap-4 xl:grid-cols-[minmax(16rem,20rem)_minmax(0,1fr)]"
                        : "space-y-3"
                    }`}
                  >
                    <div className="text-sm text-gray-700">
                      <span className="font-medium">
                        {t("training.columnsDetected")}:
                      </span>{" "}
                      {uploadPreview.columns.length}
                    </div>
                    <div className={isFeatureSelectorExpanded ? "xl:col-start-1" : ""}>
                      <label className="label">
                        {t("training.targetColumn")}
                      </label>
                      <select
                        value={targetColumn}
                        onChange={(event) =>
                          handleTargetColumnChange(event.target.value)
                        }
                        className="select-field"
                      >
                        {uploadPreview.columns.map((column) => (
                          <option key={column} value={column}>
                            {column}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div
                      className={
                        isFeatureSelectorExpanded
                          ? "xl:col-start-2 xl:row-span-3 xl:row-start-1"
                          : ""
                      }
                    >
                      <div className="mb-2 flex items-center justify-between">
                        <label className="label mb-0">
                          {t("training.featureColumns")}
                        </label>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={toggleFeatureSelectorExpanded}
                            className="rounded p-1 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
                            title={t(
                              isFeatureSelectorExpanded
                                ? "training.collapseFeatureColumns"
                                : "training.expandFeatureColumns",
                            )}
                            aria-label={t(
                              isFeatureSelectorExpanded
                                ? "training.collapseFeatureColumns"
                                : "training.expandFeatureColumns",
                            )}
                          >
                            {isFeatureSelectorExpanded ? (
                              <svg
                                className="h-4 w-4"
                                viewBox="0 0 20 20"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                              >
                                <path
                                  d="M7.5 3.75H3.75V7.5M12.5 3.75H16.25V7.5M16.25 12.5V16.25H12.5M7.5 16.25H3.75V12.5"
                                  stroke="currentColor"
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                                <path
                                  d="M8.75 8.75L3.75 3.75M11.25 8.75L16.25 3.75M11.25 11.25L16.25 16.25M8.75 11.25L3.75 16.25"
                                  stroke="currentColor"
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            ) : (
                              <svg
                                className="h-4 w-4"
                                viewBox="0 0 20 20"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                              >
                                <path
                                  d="M8.75 3.75H3.75V8.75M11.25 3.75H16.25V8.75M16.25 11.25V16.25H11.25M8.75 16.25H3.75V11.25"
                                  stroke="currentColor"
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                                <path
                                  d="M6.25 6.25L3.75 3.75M13.75 6.25L16.25 3.75M13.75 13.75L16.25 16.25M6.25 13.75L3.75 16.25"
                                  stroke="currentColor"
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            )}
                          </button>
                          <button
                            type="button"
                            onClick={selectAllFeatureColumns}
                            className="text-xs text-primary-700 hover:underline"
                          >
                            {t("training.selectAll")}
                          </button>
                          <button
                            type="button"
                            onClick={clearAllFeatureColumns}
                            className="text-xs text-gray-600 hover:underline"
                          >
                            {t("training.clearAll")}
                          </button>
                        </div>
                      </div>
                      <div
                        className={`overflow-y-auto rounded border border-gray-200 bg-white p-2 ${
                          isFeatureSelectorExpanded
                            ? "max-h-[calc(100vh-16rem)]"
                            : "max-h-40"
                        }`}
                      >
                        <div
                          className={`space-y-1 ${
                            isFeatureSelectorExpanded
                              ? "md:columns-2 xl:columns-3"
                              : ""
                          }`}
                        >
                        {uploadPreview.columns.map((column) => {
                          const isTarget = column === targetColumn;
                          const checked =
                            !isTarget &&
                            selectedFeatureColumns.includes(column);
                          return (
                            <label
                              key={column}
                              className={`mb-1 flex break-inside-avoid items-center justify-between gap-2 rounded px-2 py-1 ${
                                isTarget
                                  ? "bg-amber-50 text-amber-700"
                                  : "hover:bg-gray-50"
                              }`}
                            >
                              <span className="text-sm">{column}</span>
                              {isTarget ? (
                                <span className="text-xs font-medium">
                                  {t("training.target")}
                                </span>
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
                      </div>
                      <p className="mt-2 text-xs text-gray-500">
                        {t("training.selectedFeatures")}:{" "}
                        {selectedFeatureColumns.length}
                      </p>
                    </div>
                    <div
                      className={`max-h-28 overflow-y-auto rounded border border-gray-200 bg-white p-2 text-xs text-gray-600 ${
                        isFeatureSelectorExpanded ? "xl:col-start-1" : ""
                      }`}
                    >
                      {uploadPreview.columns.join(", ")}
                    </div>
                  </div>
                )}
              </div>

              <div className="flex justify-end gap-3 border-t border-gray-200 px-6 py-4">
                <button
                  onClick={closeModal}
                  disabled={uploading}
                  className="rounded-lg border border-gray-300 px-4 py-2 text-gray-700 hover:bg-gray-50"
                >
                  {t("common.cancel")}
                </button>
                <button
                  onClick={handleLoadUploadedDataset}
                  disabled={
                    !uploadPreview ||
                    !targetColumn ||
                    !selectedFeatureColumns.length ||
                    uploading
                  }
                  className="rounded-lg bg-primary-600 px-4 py-2 text-white hover:bg-primary-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {uploading
                    ? t("training.loadingDataset")
                    : t("training.loadDataset")}
                </button>
              </div>
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
};

export default TrainingPanel;
