import React, { useCallback, useEffect, useRef, useState } from "react";
import apiService from "./api/apiService";
import Header from "./components/Header";
import TrainingPanel from "./components/TrainingPanel";
import MetricsCard from "./components/MetricsCard";
import PredictionForm from "./components/PredictionForm";
import EditableShapeFunctionsGrid from "./components/EditableShapeFunctionsGrid";
import PredictionChart from "./components/PredictionChart";
import PredictionComparisonChart from "./components/PredictionComparisonChart";
import HourlyPatternChart from "./components/HourlyPatternChart";
import DataSummaryCard from "./components/DataSummaryCard";
import UserLogin from "./components/UserLogin";
import CombinedResultsPage from "./components/CombinedResultsPage";
import SuperadminPage from "./components/SuperadminPage";

const EMPTY_STATUS = {
  dataset_id: null,
  model_version_id: null,
  dataset_name: null,
  version_number: null,
  data_loaded: false,
  is_trained: false,
  train_size: 0,
  test_size: 0,
  features: [],
};

const convertEditsListToMap = (edits = []) => {
  const editsMap = {};
  for (const sf of edits) {
    editsMap[sf.feature_name] = (sf.edited_points || []).map((point) => ({
      x_value: point.x_value,
      y_value: point.y_value,
    }));
  }
  return editsMap;
};

function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [currentPage, setCurrentPage] = useState("login");

  const [datasetContext, setDatasetContext] = useState(null);
  const [modelStatus, setModelStatus] = useState(EMPTY_STATUS);
  const [dataSummary, setDataSummary] = useState(null);
  const [predictionFields, setPredictionFields] = useState([]);
  const [capabilities, setCapabilities] = useState({});

  const [metrics, setMetrics] = useState(null);
  const [shapeFunctions, setShapeFunctions] = useState([]);
  const [predictionsData, setPredictionsData] = useState(null);
  const [hourlyPattern, setHourlyPattern] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [userSavedEdits, setUserSavedEdits] = useState({});

  const [deletionNotifications, setDeletionNotifications] = useState([]);
  const [showNotificationPopup, setShowNotificationPopup] = useState(false);

  const [contextLoading, setContextLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [predictLoading, setPredictLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  const [error, setError] = useState(null);

  const previewTimerRef = useRef(null);
  const isApplyingSavedEditsRef = useRef(false);

  const clearModelState = useCallback(() => {
    setMetrics(null);
    setShapeFunctions([]);
    setPredictionsData(null);
    setComparisonData(null);
    setUserSavedEdits({});
  }, []);

  const fetchNotifications = useCallback(async (userId, modelVersionId) => {
    if (!userId || !modelVersionId) {
      setDeletionNotifications([]);
      setShowNotificationPopup(false);
      return;
    }

    try {
      const data = await apiService.getUserNotifications(userId, modelVersionId);
      if (data.notifications?.length) {
        setDeletionNotifications(data.notifications);
        setShowNotificationPopup(true);
      } else {
        setDeletionNotifications([]);
        setShowNotificationPopup(false);
      }
    } catch (err) {
      console.log("Failed to check notifications:", err);
    }
  }, []);

  const loadModelVersionData = useCallback(
    async (modelVersionId, userId) => {
      if (!modelVersionId || !userId) {
        clearModelState();
        return;
      }

      try {
        setChartLoading(true);
        setComparisonLoading(true);

        const [metricsData, shapeData, predictionsVsActual, comparison, userEdits] =
          await Promise.all([
            apiService.getModelMetrics(modelVersionId),
            apiService.getShapeFunctions(modelVersionId),
            apiService.getPredictionsVsActual(modelVersionId),
            apiService.getPredictionsComparison(modelVersionId, userId),
            apiService.getUserEdits(userId, modelVersionId),
          ]);

        setMetrics(metricsData);
        setShapeFunctions(shapeData.shape_functions || []);
        setPredictionsData(predictionsVsActual);
        setComparisonData(comparison);
        setUserSavedEdits(convertEditsListToMap(userEdits.edits || []));
      } catch (err) {
        console.error("Failed to load model version data:", err);
        clearModelState();
      } finally {
        setChartLoading(false);
        setComparisonLoading(false);
      }
    },
    [clearModelState],
  );

  const refreshContext = useCallback(
    async (userOverride = null) => {
      const activeUser = userOverride || currentUser;
      if (!activeUser?.id) {
        return;
      }

      try {
        setContextLoading(true);
        const context = await apiService.getUserContext(activeUser.id);
        setDatasetContext(context);
        setModelStatus(context.model_status || EMPTY_STATUS);
        setDataSummary(context.data_summary || null);
        setPredictionFields(context.prediction_fields || []);
        setCapabilities(context.capabilities || {});

        if (
          context.model_status?.dataset_id &&
          context.capabilities?.hourly_pattern
        ) {
          try {
            const pattern = await apiService.getHourlyPattern(
              context.model_status.dataset_id,
            );
            setHourlyPattern(pattern);
          } catch (err) {
            console.log("Failed to fetch hourly pattern:", err);
            setHourlyPattern(null);
          }
        } else {
          setHourlyPattern(null);
        }

        if (context.model_status?.model_version_id) {
          await loadModelVersionData(
            context.model_status.model_version_id,
            activeUser.id,
          );
          await fetchNotifications(
            activeUser.id,
            context.model_status.model_version_id,
          );
        } else {
          clearModelState();
          setDeletionNotifications([]);
          setShowNotificationPopup(false);
        }
      } catch (err) {
        console.error("Failed to fetch active context:", err);
      } finally {
        setContextLoading(false);
      }
    },
    [
      clearModelState,
      currentUser,
      fetchNotifications,
      loadModelVersionData,
    ],
  );

  useEffect(() => {
    const validateAndLoadUser = async () => {
      const savedUser = localStorage.getItem("currentUser");
      if (!savedUser) {
        return;
      }

      try {
        const user = JSON.parse(savedUser);
        const validatedUser = await apiService.getUser(user.id);
        if (!validatedUser) {
          localStorage.removeItem("currentUser");
          return;
        }
        setCurrentUser({ ...user, ...validatedUser });
        setCurrentPage("main");
      } catch (err) {
        localStorage.removeItem("currentUser");
      }
    };

    validateAndLoadUser();
  }, []);

  useEffect(() => {
    if (currentPage === "main" && currentUser) {
      refreshContext();
    }
  }, [currentPage, currentUser, refreshContext]);

  useEffect(() => {
    if (currentPage === "superadmin" && !currentUser?.is_superadmin) {
      setCurrentPage("main");
    }
  }, [currentPage, currentUser]);

  useEffect(() => {
    return () => {
      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
      }
    };
  }, []);

  const handleDismissNotifications = async () => {
    if (!currentUser?.id || !modelStatus?.model_version_id) {
      return;
    }
    try {
      await apiService.markNotificationsSeen(
        currentUser.id,
        modelStatus.model_version_id,
      );
    } catch (err) {
      console.log("Failed to mark notifications as seen:", err);
    }
    setShowNotificationPopup(false);
    setDeletionNotifications([]);
  };

  const handleLogin = async (username, password) => {
    const user = await apiService.loginUser(username, password);
    setCurrentUser(user);
    localStorage.setItem("currentUser", JSON.stringify(user));
    if (user.access_token) {
      localStorage.setItem("superadminToken", user.access_token);
    } else {
      localStorage.removeItem("superadminToken");
    }
    setCurrentPage("main");
    await refreshContext(user);
  };

  const handleRegister = async (username, password, inviteToken) => {
    const user = await apiService.registerUser(username, password, inviteToken);
    setCurrentUser(user);
    localStorage.setItem("currentUser", JSON.stringify(user));
    localStorage.removeItem("superadminToken");
    setCurrentPage("main");
    await refreshContext(user);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem("currentUser");
    localStorage.removeItem("superadminToken");
    setCurrentPage("login");
    setDatasetContext(null);
    setModelStatus(EMPTY_STATUS);
    setDataSummary(null);
    setPredictionFields([]);
    setCapabilities({});
    clearModelState();
    setHourlyPattern(null);
    setDeletionNotifications([]);
    setShowNotificationPopup(false);
    setError(null);
  };

  const handleResetDatabase = async () => {
    await apiService.resetDatabase();
    handleLogout();
  };

  const handleSelectDataset = async (datasetId) => {
    if (!currentUser?.id) {
      return;
    }
    setError(null);
    try {
      await apiService.selectDataset(currentUser.id, datasetId);
      await refreshContext();
    } catch (err) {
      setError(
        "Failed to select dataset: " +
          (err.response?.data?.detail || err.message),
      );
    }
  };

  const handleInspectUpload = async (file) => apiService.inspectDatasetUpload(file);

  const handleUploadDataset = async (payload) => {
    if (!currentUser?.id) {
      return;
    }
    setError(null);
    try {
      const result = await apiService.uploadDataset(payload);
      const datasetId = result?.dataset?.id;
      if (datasetId) {
        await apiService.selectDataset(currentUser.id, datasetId);
      }
      await refreshContext();
    } catch (err) {
      setError(
        "Failed to upload dataset: " +
          (err.response?.data?.detail || err.message),
      );
      throw err;
    }
  };

  const handleTrainModel = async (params) => {
    if (!currentUser?.id || !modelStatus?.dataset_id) {
      return;
    }
    try {
      setTrainingLoading(true);
      setError(null);
      await apiService.trainDataset(modelStatus.dataset_id, {
        user_id: currentUser.id,
        ...params,
      });
      await refreshContext();
    } catch (err) {
      setError(
        "Failed to train model: " + (err.response?.data?.detail || err.message),
      );
    } finally {
      setTrainingLoading(false);
    }
  };

  const handlePredict = async (inputData) => {
    if (!modelStatus?.model_version_id) {
      return null;
    }

    try {
      setPredictLoading(true);
      setError(null);
      return await apiService.predict(modelStatus.model_version_id, inputData);
    } catch (err) {
      setError(
        "Prediction failed: " + (err.response?.data?.detail || err.message),
      );
      return null;
    } finally {
      setPredictLoading(false);
    }
  };

  const buildMergedPreviewEdits = useCallback(
    (unsavedEdits = {}) => {
      if (!shapeFunctions?.length) {
        return [];
      }

      return shapeFunctions
        .map((sf) => {
          const saved = userSavedEdits[sf.feature_name] || [];
          const unsaved = unsavedEdits[sf.feature_name] || [];

          const mergedMap = new Map();
          saved.forEach((point) => mergedMap.set(String(point.x_value), point));
          unsaved.forEach((point) =>
            mergedMap.set(String(point.x_value), point),
          );

          const mergedPoints = Array.from(mergedMap.values()).map((point) => ({
            x_value: point.x_value,
            y_value: point.y_value,
          }));

          if (mergedPoints.length === 0) {
            return null;
          }

          return {
            feature_name: sf.feature_name,
            feature_type: sf.feature_type || "numeric",
            edited_points: mergedPoints,
          };
        })
        .filter(Boolean);
    },
    [shapeFunctions, userSavedEdits],
  );

  const handleUnsavedEditsChange = useCallback(
    (unsavedEdits) => {
      if (
        !modelStatus?.is_trained ||
        !currentUser?.id ||
        !modelStatus?.model_version_id ||
        isApplyingSavedEditsRef.current
      ) {
        return;
      }

      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
      }

      previewTimerRef.current = setTimeout(async () => {
        try {
          const mergedPreviewEdits = buildMergedPreviewEdits(unsavedEdits);
          const comparison = await apiService.previewPredictionsComparison(
            modelStatus.model_version_id,
            currentUser.id,
            mergedPreviewEdits,
          );
          setComparisonData(comparison);
        } catch (err) {
          console.error("Failed to refresh live comparison preview:", err);
        }
      }, 180);
    },
    [buildMergedPreviewEdits, currentUser?.id, modelStatus],
  );

  const fetchComparisonData = useCallback(async () => {
    if (!currentUser?.id || !modelStatus?.model_version_id) {
      return;
    }

    try {
      setComparisonLoading(true);
      const data = await apiService.getPredictionsComparison(
        modelStatus.model_version_id,
        currentUser.id,
      );
      setComparisonData(data);
    } catch (err) {
      console.error("Failed to fetch comparison data:", err);
    } finally {
      setComparisonLoading(false);
    }
  }, [currentUser?.id, modelStatus?.model_version_id]);

  const handleShapeFunctionsEdit = async (editedShapeFunctions) => {
    if (!currentUser?.id || !modelStatus?.model_version_id) {
      return;
    }

    try {
      isApplyingSavedEditsRef.current = true;
      setComparisonLoading(true);
      setError(null);

      await apiService.saveUserEdits(
        currentUser.id,
        modelStatus.model_version_id,
        editedShapeFunctions,
      );

      setUserSavedEdits((prevEdits) => {
        const nextEdits = { ...prevEdits };
        for (const sf of editedShapeFunctions) {
          nextEdits[sf.feature_name] = sf.edited_points;
        }
        return nextEdits;
      });

      await fetchComparisonData();
    } catch (err) {
      setError(
        "Failed to apply shape function edits: " +
          (err.response?.data?.detail || err.message),
      );
    } finally {
      isApplyingSavedEditsRef.current = false;
      setComparisonLoading(false);
    }
  };

  const handleResetShapeFunctions = async () => {
    if (!currentUser?.id || !modelStatus?.model_version_id) {
      return;
    }

    try {
      setComparisonLoading(true);
      setError(null);
      await apiService.clearUserEdits(
        currentUser.id,
        modelStatus.model_version_id,
      );
      setUserSavedEdits({});
      await fetchComparisonData();
    } catch (err) {
      setError(
        "Failed to reset shape functions: " +
          (err.response?.data?.detail || err.message),
      );
    } finally {
      setComparisonLoading(false);
    }
  };

  const handleResetFeature = async (featureName) => {
    if (!currentUser?.id || !modelStatus?.model_version_id) {
      return;
    }

    try {
      setComparisonLoading(true);
      setError(null);
      await apiService.clearUserFeatureEdits(
        currentUser.id,
        modelStatus.model_version_id,
        featureName,
      );
      setUserSavedEdits((prevEdits) => {
        const nextEdits = { ...prevEdits };
        delete nextEdits[featureName];
        return nextEdits;
      });
      await fetchComparisonData();
    } catch (err) {
      setError(
        "Failed to reset feature: " +
          (err.response?.data?.detail || err.message),
      );
    } finally {
      setComparisonLoading(false);
    }
  };

  if (currentPage === "login") {
    return <UserLogin onLogin={handleLogin} onRegister={handleRegister} />;
  }

  if (currentPage === "combined") {
    return (
      <CombinedResultsPage
        onBack={() => setCurrentPage("main")}
        onResetDatabase={handleResetDatabase}
        currentUser={currentUser}
      />
    );
  }

  if (currentPage === "superadmin" && currentUser?.is_superadmin) {
    return (
      <SuperadminPage
        onBack={() => setCurrentPage("main")}
        onOpenCombined={() => setCurrentPage("combined")}
      />
    );
  }

  const targetLabel =
    dataSummary?.target_label ||
    datasetContext?.active_dataset?.target_column ||
    "target value";
  const predictionContextKey = `main-${modelStatus?.model_version_id || modelStatus?.dataset_id || "none"}`;

  return (
    <div className="min-h-screen bg-gray-50">
      <Header modelStatus={modelStatus} />

      <div className="bg-white border-b">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                <span className="text-primary-600 font-medium text-sm">
                  {currentUser?.name?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-sm text-gray-600">Logged in as </span>
                <span className="font-medium text-gray-800">
                  {currentUser?.name}
                </span>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => setCurrentPage("combined")}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center space-x-2"
              >
                <span>View Combined Results</span>
              </button>
              {currentUser?.is_superadmin ? (
                <button
                  onClick={() => setCurrentPage("superadmin")}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  Superadmin
                </button>
              ) : null}
              <button
                onClick={handleLogout}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      <main className="container mx-auto px-4 py-8">
        {error ? (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center justify-between">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
        ) : null}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-6">
            <TrainingPanel
              currentUser={currentUser}
              datasets={datasetContext?.datasets || []}
              activeDatasetId={modelStatus?.dataset_id}
              activeDatasetName={datasetContext?.active_dataset?.display_name}
              onSelectDataset={handleSelectDataset}
              onInspectUpload={handleInspectUpload}
              onUploadDataset={handleUploadDataset}
              onTrainModel={handleTrainModel}
              loading={contextLoading || trainingLoading}
              modelStatus={modelStatus}
            />

            <DataSummaryCard summary={dataSummary} loading={contextLoading} />

            <MetricsCard metrics={metrics} loading={trainingLoading} />
          </div>

          <div className="lg:col-span-2 space-y-6">
            <PredictionForm
              title="Make Prediction"
              description="Set the input feature values and run a prediction for the active IGANN model version."
              onPredict={handlePredict}
              loading={predictLoading}
              modelTrained={modelStatus?.is_trained}
              predictionFields={predictionFields}
              targetLabel={targetLabel}
              submitLabel="Predict Value"
              contextKey={predictionContextKey}
            />

            {capabilities?.hourly_pattern ? (
              <HourlyPatternChart
                patternData={hourlyPattern}
                loading={contextLoading}
              />
            ) : null}

            <PredictionChart
              predictionsData={predictionsData}
              loading={chartLoading}
            />
          </div>
        </div>

        <div className="mt-6">
          <EditableShapeFunctionsGrid
            shapeFunctions={shapeFunctions}
            loading={chartLoading}
            onShapeFunctionsEdit={handleShapeFunctionsEdit}
            onReset={handleResetShapeFunctions}
            onFeatureReset={handleResetFeature}
            initialEditedPoints={userSavedEdits}
            onUnsavedEditsChange={handleUnsavedEditsChange}
          />
        </div>

        <div className="mt-6">
          <PredictionComparisonChart
            comparisonData={comparisonData}
            loading={comparisonLoading}
            currentUser={currentUser}
            targetLabel={targetLabel}
          />
        </div>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>
            Interactive IGANN Workspace
            <span className="mx-2">•</span>
            <a
              href="https://github.com/MathiasKraus/igann"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 hover:underline"
            >
              IGANN GitHub
            </a>
          </p>
        </footer>
      </main>

      {showNotificationPopup && deletionNotifications.length > 0 ? (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-lg w-full mx-4">
            <div className="flex items-center space-x-2 mb-4">
              <svg
                className="w-6 h-6 text-amber-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                />
              </svg>
              <h3 className="text-lg font-bold text-gray-800">
                Edit{deletionNotifications.length > 1 ? "s" : ""} Removed
              </h3>
            </div>
            <p className="text-gray-600 mb-4 text-sm">
              {deletionNotifications.length === 1
                ? "One of your edits was removed by another user:"
                : `${deletionNotifications.length} of your edits were removed by other users:`}
            </p>
            <div className="space-y-3 max-h-60 overflow-y-auto mb-4">
              {deletionNotifications.map((notification) => (
                <div
                  key={notification.id}
                  className="bg-amber-50 border border-amber-200 rounded-lg p-3"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      Feature:{" "}
                      <span className="font-mono">
                        {notification.feature_name}
                      </span>
                    </span>
                    <span className="text-xs text-gray-500">
                      by {notification.deleted_by}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500 mb-1">
                    X Value:{" "}
                    <span className="font-mono">{notification.x_value}</span>
                  </div>
                  <div className="text-sm text-gray-700 mt-1">
                    <span className="font-medium">Reason:</span>{" "}
                    {notification.reason}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex justify-end">
              <button
                onClick={handleDismissNotifications}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
