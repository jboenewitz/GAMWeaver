import React, { useState, useEffect, useRef, useCallback } from "react";
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
import LanguageToggleButton from "./components/LanguageToggleButton";
import {
  createTranslator,
  getInitialLanguage,
  LANGUAGE_STORAGE_KEY,
} from "./i18n";

const formatApiError = (err, fallback = "Request failed") => {
  const responseData = err?.response?.data;
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    try {
      return detail
        .map((item) => {
          if (typeof item === "string") return item;
          if (item?.msg) return item.msg;
          return JSON.stringify(item);
        })
        .join("; ");
    } catch (_) {
      return fallback;
    }
  }
  if (detail && typeof detail === "object") {
    if (detail.msg) return detail.msg;
    try {
      return JSON.stringify(detail);
    } catch (_) {
      return fallback;
    }
  }
  if (
    typeof responseData?.message === "string" &&
    responseData.message.trim()
  ) {
    return responseData.message;
  }
  if (typeof err?.message === "string" && err.message.trim())
    return err.message;
  return fallback;
};

function App() {
  const showPredictionComparisonOnMain = false;
  const [language, setLanguage] = useState(getInitialLanguage);
  const t = createTranslator(language);

  // User state
  const [currentUser, setCurrentUser] = useState(null);
  const [currentPage, setCurrentPage] = useState("login"); // 'login', 'main', 'combined', 'superadmin'

  // State
  const [modelStatus, setModelStatus] = useState(null);
  const [dataSummary, setDataSummary] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [shapeFunctions, setShapeFunctions] = useState([]);
  const [comparisonShapeFunctions, setComparisonShapeFunctions] = useState([]);
  const [predictionsData, setPredictionsData] = useState(null);
  const [hourlyPattern, setHourlyPattern] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [userSavedEdits, setUserSavedEdits] = useState({}); // User's saved edits for display
  const previewTimerRef = useRef(null);
  const isApplyingSavedEditsRef = useRef(false);

  // Notification state
  const [deletionNotifications, setDeletionNotifications] = useState([]);
  const [showNotificationPopup, setShowNotificationPopup] = useState(false);

  // Loading states
  const [loading, setLoading] = useState(false);
  const [dataLoading, setDataLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  // Error state
  const [error, setError] = useState(null);

  // Check for saved user on mount
  useEffect(() => {
    document.documentElement.lang = language;
    localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  }, [language]);

  useEffect(() => {
    const validateAndLoadUser = async () => {
      const savedUser = localStorage.getItem("currentUser");
      if (savedUser) {
        try {
          const user = JSON.parse(savedUser);
          // Verify user still exists in database by trying to get their info
          try {
            const validatedUser = await apiService.getUser(user.id);
            if (validatedUser) {
              setCurrentUser({ ...user, ...validatedUser });
              setCurrentPage("main");
            } else {
              // User no longer exists, clear localStorage
              localStorage.removeItem("currentUser");
            }
          } catch (err) {
            // User not found in DB, clear localStorage
            console.log("Saved user no longer exists, logging out");
            localStorage.removeItem("currentUser");
          }
        } catch (e) {
          localStorage.removeItem("currentUser");
        }
      }
    };
    validateAndLoadUser();
  }, []);

  // Fetch initial status when on main page
  useEffect(() => {
    if (currentPage === "main" && currentUser) {
      fetchModelStatus();
      checkForNotifications();
    }
  }, [currentPage, currentUser]);

  useEffect(() => {
    if (currentPage === "superadmin" && !currentUser?.is_superadmin) {
      setCurrentPage("main");
    }
  }, [currentPage, currentUser]);

  const checkForNotifications = async () => {
    if (!currentUser) return;
    try {
      const data = await apiService.getUserNotifications(currentUser.id);
      if (data.notifications && data.notifications.length > 0) {
        setDeletionNotifications(data.notifications);
        setShowNotificationPopup(true);
      }
    } catch (e) {
      console.log("Failed to check notifications:", e);
    }
  };

  const handleDismissNotifications = async () => {
    if (!currentUser) return;
    try {
      await apiService.markNotificationsSeen(currentUser.id);
    } catch (e) {
      console.log("Failed to mark notifications as seen:", e);
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

    // Load user's previous edits if they exist
    if (!user.is_new) {
      try {
        await apiService.loadUserEditsToModel(user.id);
      } catch (e) {
        console.log("No previous edits to load");
      }
    }

    // Ensure all user-specific data (including comparison chart) is refreshed
    // against the just-logged-in user context.
    await fetchModelStatus(user);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    localStorage.removeItem("currentUser");
    localStorage.removeItem("superadminToken");
    setCurrentPage("login");
    // Reset all state
    setModelStatus(null);
    setDataSummary(null);
    setMetrics(null);
    setShapeFunctions([]);
    setPredictionsData(null);
    setHourlyPattern(null);
    setComparisonData(null);
    setUserSavedEdits({});
  };

  const handleResetDatabase = async () => {
    await apiService.resetDatabase();
    handleLogout();
  };

  const handleRegister = async (username, password, inviteToken, profession) => {
    const user = await apiService.registerUser(
      username,
      password,
      inviteToken,
      profession,
    );
    setCurrentUser(user);
    localStorage.setItem("currentUser", JSON.stringify(user));
    localStorage.removeItem("superadminToken");
    setCurrentPage("main");
    await fetchModelStatus(user);
  };

  const clearTrainedModelState = () => {
    setMetrics(null);
    setShapeFunctions([]);
    setComparisonShapeFunctions([]);
    setPredictionsData(null);
    setComparisonData(null);
    setUserSavedEdits({});
  };

  const fetchModelStatus = async (userOverride = null) => {
    const activeUser = userOverride || currentUser;
    try {
      const status = await apiService.getModelStatus();
      setModelStatus(status);

      // If data is already loaded, fetch summary
      if (status.data_loaded) {
        fetchDataSummary();
        fetchHourlyPattern();
      } else {
        setDataSummary(null);
        setHourlyPattern(null);
      }

      // If model is trained, fetch metrics and visualizations
      if (status.is_trained) {
        await fetchShapeFunctions({ refreshComparison: false });

        // Load user's saved edits FIRST, then fetch comparison data
        if (activeUser) {
          try {
            const result = await apiService.loadUserEditsToModel(activeUser.id);
            // Convert the edits to the format expected by EditableShapeFunctionsGrid
            if (result.edits && result.edits.length > 0) {
              const editsMap = {};
              for (const sf of result.edits) {
                editsMap[sf.feature_name] = sf.edited_points.map((p) => ({
                  x_value: p.x_value,
                  y_value: p.y_value,
                }));
              }
              setUserSavedEdits(editsMap);
            } else {
              setUserSavedEdits({});
            }
          } catch (e) {
            console.log("No previous edits to load");
            setUserSavedEdits({});
          }
        }

        if (status.analytics_available) {
          fetchMetrics(status.analytics_available);
          fetchPredictionsVsActual(status.analytics_available);
          await fetchComparisonData(status.analytics_available);
        } else {
          setMetrics(null);
          setPredictionsData(null);
          setComparisonData(null);
        }
      } else {
        clearTrainedModelState();
      }
    } catch (err) {
      console.error("Failed to fetch model status:", err);
    }
  };

  const fetchDataSummary = async () => {
    try {
      const summary = await apiService.getDataSummary();
      setDataSummary(summary);
    } catch (err) {
      console.error("Failed to fetch data summary:", err);
    }
  };

  const fetchHourlyPattern = async () => {
    try {
      const pattern = await apiService.getHourlyPattern();
      setHourlyPattern(pattern);
    } catch (err) {
      console.error("Failed to fetch hourly pattern:", err);
    }
  };

  const fetchMetrics = async (
    analyticsAvailable = modelStatus?.analytics_available,
  ) => {
    if (!analyticsAvailable) {
      setMetrics(null);
      return;
    }
    try {
      const metrics = await apiService.getModelMetrics();
      setMetrics(metrics);
    } catch (err) {
      console.error("Failed to fetch metrics:", err);
    }
  };

  const fetchShapeFunctions = async ({ refreshComparison = true } = {}) => {
    try {
      setChartLoading(true);
      const response = await apiService.getShapeFunctions();
      setShapeFunctions(response.shape_functions || []);
      setComparisonShapeFunctions(response.comparison_shape_functions || []);
      // Also fetch initial comparison data after shape functions are loaded
      if (refreshComparison && modelStatus?.analytics_available) {
        await fetchComparisonData();
      }
    } catch (err) {
      console.error("Failed to fetch shape functions:", err);
      setComparisonShapeFunctions([]);
    } finally {
      setChartLoading(false);
    }
  };

  const fetchPredictionsVsActual = async (
    analyticsAvailable = modelStatus?.analytics_available,
  ) => {
    if (!analyticsAvailable) {
      setPredictionsData(null);
      return;
    }
    try {
      const data = await apiService.getPredictionsVsActual();
      setPredictionsData(data);
    } catch (err) {
      console.error("Failed to fetch predictions vs actual:", err);
    }
  };

  const fetchComparisonData = async (
    analyticsAvailable = modelStatus?.analytics_available,
  ) => {
    if (!analyticsAvailable) {
      setComparisonData(null);
      return;
    }
    try {
      setComparisonLoading(true);
      const data = await apiService.getPredictionsComparison();
      setComparisonData(data);
    } catch (err) {
      console.error("Failed to fetch comparison data:", err);
    } finally {
      setComparisonLoading(false);
    }
  };

  const buildMergedPreviewEdits = useCallback(
    (unsavedEdits = {}) => {
      if (!shapeFunctions || shapeFunctions.length === 0) return [];

      return shapeFunctions
        .map((sf) => {
          const saved = userSavedEdits[sf.feature_name] || [];
          const unsaved = unsavedEdits[sf.feature_name] || [];

          const mergedMap = new Map();
          saved.forEach((p) => mergedMap.set(String(p.x_value), p));
          unsaved.forEach((p) => mergedMap.set(String(p.x_value), p));

          const mergedPoints = Array.from(mergedMap.values()).map((p) => ({
            x_value: p.x_value,
            y_value: p.y_value,
          }));

          if (mergedPoints.length === 0) return null;

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
      if (!modelStatus?.is_trained || isApplyingSavedEditsRef.current) return;

      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
      }

      previewTimerRef.current = setTimeout(async () => {
        try {
          const mergedPreviewEdits = buildMergedPreviewEdits(unsavedEdits);
          await apiService.updateShapeFunctions(mergedPreviewEdits);
          if (modelStatus?.analytics_available) {
            const data = await apiService.getPredictionsComparison();
            setComparisonData(data);
          } else {
            setComparisonData(null);
          }
        } catch (err) {
          console.error("Failed to refresh live comparison preview:", err);
        }
      }, 180);
    },
    [
      modelStatus?.is_trained,
      modelStatus?.analytics_available,
      buildMergedPreviewEdits,
    ],
  );

  useEffect(() => {
    return () => {
      if (previewTimerRef.current) {
        clearTimeout(previewTimerRef.current);
      }
    };
  }, []);

  const handleShapeFunctionsEdit = async (editedShapeFunctions) => {
    try {
      isApplyingSavedEditsRef.current = true;
      setComparisonLoading(true);
      setError(null);

      // Send edited shape functions to backend (temporary update)
      await apiService.updateShapeFunctions(editedShapeFunctions);

      // Save to user's database record
      if (currentUser) {
        await apiService.saveUserEdits(currentUser.id, editedShapeFunctions);

        // Update local state with saved edits (merge with existing)
        setUserSavedEdits((prevEdits) => {
          const newEdits = { ...prevEdits };
          for (const sf of editedShapeFunctions) {
            // Replace all points for this feature with the new ones
            newEdits[sf.feature_name] = sf.edited_points;
          }
          return newEdits;
        });
      }

      if (modelStatus?.analytics_available) {
        await fetchComparisonData();
      } else {
        setComparisonData(null);
      }
    } catch (err) {
      setError(
        `${t("app.error.applyShapeEdits")}: ` +
          formatApiError(err, t("app.error.applyShapeEdits")),
      );
    } finally {
      isApplyingSavedEditsRef.current = false;
      setComparisonLoading(false);
    }
  };

  const handleResetShapeFunctions = async () => {
    try {
      setComparisonLoading(true);
      setError(null);

      // Reset shape functions on backend
      await apiService.resetShapeFunctions();

      // Clear user's edits in database
      if (currentUser) {
        await apiService.clearUserEdits(currentUser.id);
      }

      // Clear local saved edits
      setUserSavedEdits({});

      if (modelStatus?.analytics_available) {
        await fetchComparisonData();
      } else {
        setComparisonData(null);
      }
    } catch (err) {
      setError(
        `${t("app.error.resetShapeFunctions")}: ` +
          formatApiError(err, t("app.error.resetShapeFunctions")),
      );
    } finally {
      setComparisonLoading(false);
    }
  };

  const handleResetFeature = async (featureName) => {
    try {
      setComparisonLoading(true);
      setError(null);

      // Clear this feature's edits in database
      if (currentUser) {
        await apiService.clearUserFeatureEdits(currentUser.id, featureName);
      }

      // Remove from local saved edits
      setUserSavedEdits((prevEdits) => {
        const newEdits = { ...prevEdits };
        delete newEdits[featureName];
        return newEdits;
      });

      // Reload user's remaining edits to model
      if (currentUser) {
        await apiService.loadUserEditsToModel(currentUser.id);
      }

      if (modelStatus?.analytics_available) {
        await fetchComparisonData();
      } else {
        setComparisonData(null);
      }
    } catch (err) {
      setError(
        `${t("app.error.resetFeature")}: ` +
          formatApiError(err, t("app.error.resetFeature")),
      );
    } finally {
      setComparisonLoading(false);
    }
  };

  const handleUpdateFeatureChartSettings = async (featureName, settings) => {
    try {
      setError(null);
      await apiService.updateFeatureChartSettings(featureName, settings);
      await fetchShapeFunctions({
        refreshComparison: Boolean(modelStatus?.analytics_available),
      });
    } catch (err) {
      const message = formatApiError(
        err,
        t("app.error.updateFeatureChartSettings"),
      );
      setError(`${t("app.error.updateFeatureChartSettings")}: ${message}`);
      throw new Error(message);
    }
  };

  const handleUploadDataset = async (file) => {
    try {
      setError(null);
      const result = await apiService.uploadDataset(file);
      return result;
    } catch (err) {
      throw new Error(formatApiError(err, t("app.error.uploadDataset")));
    }
  };

  const handleLoadData = async (loadRequest) => {
    try {
      setDataLoading(true);
      setError(null);
      await apiService.loadData(loadRequest);
      clearTrainedModelState();
      await fetchModelStatus();
    } catch (err) {
      const message = formatApiError(err, t("app.error.loadData"));
      setError(`${t("app.error.loadData")}: ${message}`);
      throw new Error(message);
    } finally {
      setDataLoading(false);
    }
  };

  const handleTrainModel = async (params) => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.trainModel(params);

      if (response.success) {
        setMetrics(response.metrics);
        await fetchModelStatus();
      }
    } catch (err) {
      setError(
        `${t("app.error.trainModel")}: ` +
          formatApiError(err, t("app.error.trainModel")),
      );
    } finally {
      setLoading(false);
    }
  };

  const handleUploadComparisonDataset = async (file) => {
    try {
      setError(null);
      return await apiService.uploadComparisonDataset(file);
    } catch (err) {
      throw new Error(formatApiError(err, t("app.error.uploadDataset")));
    }
  };

  const handleLoadComparisonData = async (loadRequest) => {
    try {
      setDataLoading(true);
      setError(null);
      await apiService.loadComparisonData(loadRequest);
      await fetchShapeFunctions({
        refreshComparison: Boolean(modelStatus?.analytics_available),
      });
      await fetchModelStatus();
    } catch (err) {
      const message = formatApiError(err, t("app.error.loadData"));
      setError(`${t("app.error.loadData")}: ${message}`);
      throw new Error(message);
    } finally {
      setDataLoading(false);
    }
  };

  const handleTrainComparisonModel = async () => {
    try {
      setLoading(true);
      setError(null);
      await apiService.trainComparisonModel();
      await fetchShapeFunctions({
        refreshComparison: Boolean(modelStatus?.analytics_available),
      });
      await fetchModelStatus();
    } catch (err) {
      const message = formatApiError(err, t("app.error.trainModel"));
      setError(`${t("app.error.trainModel")}: ${message}`);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (inputData) => {
    try {
      setError(null);
      const result = await apiService.predict(inputData);
      return result;
    } catch (err) {
      setError(
        `${t("app.error.predictionFailed")}: ` +
          formatApiError(err, t("app.error.predictionFailed")),
      );
      return null;
    }
  };

  const handleExportModelArtifact = async (includeEdits = false) => {
    setError(null);
    const { blob, filename } = await apiService.exportModelArtifact(includeEdits);
    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(objectUrl);
    return { filename, includeEdits };
  };

  const handleImportModelArtifact = async (file) => {
    setError(null);
    const result = await apiService.importModelArtifact(file);
    clearTrainedModelState();
    await fetchModelStatus();
    return result;
  };

  // Render login page
  if (currentPage === "login") {
    return (
      <UserLogin
        onLogin={handleLogin}
        onRegister={handleRegister}
        language={language}
      />
    );
  }

  // Render combined results page
  if (currentPage === "combined") {
    return (
      <CombinedResultsPage
        onBack={() => setCurrentPage("main")}
        currentUser={currentUser}
        language={language}
      />
    );
  }

  if (currentPage === "superadmin" && currentUser?.is_superadmin) {
    return (
      <SuperadminPage
        onBack={() => setCurrentPage("main")}
        onOpenCombined={() => setCurrentPage("combined")}
        onResetDatabase={handleResetDatabase}
        onExportModel={handleExportModelArtifact}
        onImportModel={handleImportModelArtifact}
        onUploadComparisonDataset={handleUploadComparisonDataset}
        onLoadComparisonData={handleLoadComparisonData}
        onTrainComparisonModel={handleTrainComparisonModel}
        modelStatus={modelStatus}
        busy={loading || dataLoading}
        language={language}
      />
    );
  }

  // Render main app
  return (
    <div className="app-ambient-bg relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute -top-20 left-8 h-64 w-64 rounded-full bg-primary-200/40 blur-3xl" />
      <div className="pointer-events-none absolute top-64 right-0 h-72 w-72 rounded-full bg-cyan-200/35 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-20 left-1/3 h-72 w-72 rounded-full bg-sky-200/25 blur-3xl" />
      <Header modelStatus={modelStatus} language={language} />

      {/* User Bar */}
      <div className="relative z-10 pt-2">
        <div className="container mx-auto px-4">
          <div className="glass-surface-strong flex flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-5">
            <div className="flex items-center space-x-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary-100">
                <span className="text-primary-600 font-medium text-sm">
                  {currentUser?.name?.charAt(0).toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-sm text-slate-600">
                  {t("app.loggedInAs")}{" "}
                </span>
                <span className="font-medium text-slate-800">
                  {currentUser?.name}
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <LanguageToggleButton
                language={language}
                onToggle={() =>
                  setLanguage((currentLanguage) =>
                    currentLanguage === "en" ? "de" : "en",
                  )
                }
                title={
                  language === "en"
                    ? t("language.switchToGerman")
                    : t("language.switchToEnglish")
                }
              />
              <button
                onClick={() => setCurrentPage("combined")}
                className="flex items-center space-x-2 rounded-lg bg-gradient-to-r from-emerald-600 to-cyan-600 px-4 py-2 text-white shadow-lg shadow-emerald-800/20 transition-all hover:from-emerald-500 hover:to-cyan-500"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                  />
                </svg>
                <span>{t("app.viewCombinedResults")}</span>
              </button>
              {currentUser?.is_superadmin && (
                <button
                  onClick={() => setCurrentPage("superadmin")}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-white transition-colors hover:bg-indigo-700"
                >
                  {t("app.superadmin")}
                </button>
              )}
              <button
                onClick={handleLogout}
                className="rounded-lg border border-slate-300 bg-white/70 px-4 py-2 text-slate-700 transition-colors hover:bg-white"
              >
                {t("app.logout")}
              </button>
            </div>
          </div>
        </div>
      </div>

      <main className="relative z-10 container mx-auto px-4 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 flex items-center justify-between rounded-xl border border-red-200/80 bg-red-50/90 p-4 text-red-700 shadow-md">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
        )}

        {modelStatus?.is_trained &&
          modelStatus?.model_source === "imported" &&
          !modelStatus?.analytics_available && (
            <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50/95 p-4 text-amber-900 shadow-sm">
              Imported model is active for predictions and shape-function
              editing. Load a compatible dataset as superadmin to re-enable
              metrics and dataset-derived analytics.
            </div>
          )}

        {/* Top Row: Training + Data Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TrainingPanel
            onUploadDataset={handleUploadDataset}
            onLoadData={handleLoadData}
            onTrainModel={handleTrainModel}
            loading={loading || dataLoading}
            modelStatus={modelStatus}
            isSuperadmin={Boolean(currentUser?.is_superadmin)}
            language={language}
          />
          <DataSummaryCard
            summary={dataSummary}
            loading={dataLoading}
            language={language}
          />
        </div>

        {/* Remaining Dashboard Cards/Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="space-y-6">
            <MetricsCard metrics={metrics} loading={loading} />
          </div>
          <div className="lg:col-span-2 space-y-6">
            <HourlyPatternChart
              patternData={hourlyPattern}
              loading={dataLoading}
            />

            <PredictionChart
              predictionsData={predictionsData}
              loading={chartLoading}
            />
          </div>
        </div>

        {/* Full Width Interactive Shape Functions */}
        <div className="mt-6">
          <EditableShapeFunctionsGrid
            shapeFunctions={shapeFunctions}
            comparisonShapeFunctions={
              currentUser?.is_superadmin ? comparisonShapeFunctions : []
            }
            loading={chartLoading}
            onShapeFunctionsEdit={handleShapeFunctionsEdit}
            onReset={handleResetShapeFunctions}
            onFeatureReset={handleResetFeature}
            initialEditedPoints={userSavedEdits}
            onUnsavedEditsChange={handleUnsavedEditsChange}
            isSuperadmin={Boolean(currentUser?.is_superadmin)}
            onUpdateFeatureChartSettings={handleUpdateFeatureChartSettings}
            language={language}
          />
        </div>

        {/* Prediction Comparison Section stays available for combined view but is hidden on the main dashboard */}
        {showPredictionComparisonOnMain && (
          <div className="mt-6">
            <PredictionComparisonChart
              comparisonData={comparisonData}
              loading={comparisonLoading}
              currentUser={currentUser}
              language={language}
            />
          </div>
        )}

        {/* Full Width Prediction */}
        <div className="mt-6">
          <PredictionForm
            onPredict={handlePredict}
            loading={loading}
            modelTrained={modelStatus?.is_trained}
            featureSchema={modelStatus?.feature_schema || []}
            targetColumn={modelStatus?.target_column}
            language={language}
          />
        </div>

        {/* Footer */}
        <footer className="glass-surface mt-12 px-4 py-5 text-center text-sm text-slate-600">
          <p>
            {t("app.footerDescription", { name: "Johann Boenewitz" })}
            <span className="mx-2">•</span>
            <a
              href="https://github.com/jboenewitz/GAMWeaver"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary-600 hover:underline"
            >
              {t("app.githubLink")}
            </a>
          </p>
        </footer>
      </main>

      {/* Deletion Notification Popup */}
      {showNotificationPopup && deletionNotifications.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm">
          <div className="glass-surface-strong mx-4 w-full max-w-lg p-6">
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
              <h3 className="text-lg font-bold text-slate-800">
                {deletionNotifications.length > 1
                  ? t("app.notification.editsRemoved")
                  : t("app.notification.editRemoved")}
              </h3>
            </div>
            <p className="mb-4 text-sm text-slate-600">
              {deletionNotifications.length === 1
                ? t("app.notification.singleMessage")
                : t("app.notification.multipleMessage", {
                    count: deletionNotifications.length,
                  })}
            </p>
            <div className="space-y-3 max-h-60 overflow-y-auto mb-4">
              {deletionNotifications.map((notification) => (
                <div
                  key={notification.id}
                  className="rounded-lg border border-amber-200 bg-amber-50/90 p-3"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">
                      {t("app.notification.feature")}:{" "}
                      <span className="font-mono">
                        {notification.feature_name}
                      </span>
                    </span>
                    <span className="text-xs text-gray-500">
                      {t("app.notification.by")} {notification.deleted_by}
                    </span>
                  </div>
                  {notification.point_count ? (
                    <>
                      <div className="text-xs text-gray-500 mb-1">
                        {t("app.notification.curveEdit")}:{" "}
                        <span className="font-medium">
                          {notification.point_count}{" "}
                          {t("app.notification.pointCount").toLowerCase()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 mb-1">
                        {t("app.notification.xSummary")}:{" "}
                        <span className="font-mono">
                          {notification.x_summary || notification.x_value}
                        </span>
                      </div>
                    </>
                  ) : (
                    <div className="text-xs text-gray-500 mb-1">
                      {t("app.notification.xValue")}:{" "}
                      <span className="font-mono">{notification.x_value}</span>
                    </div>
                  )}
                  <div className="text-sm text-gray-700 mt-1">
                    <span className="font-medium">
                      {t("app.notification.reason")}:
                    </span>{" "}
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
                {t("app.notification.gotIt")}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
