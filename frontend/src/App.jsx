import React, { useState, useEffect } from "react";
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

function App() {
  // User state
  const [currentUser, setCurrentUser] = useState(null);
  const [currentPage, setCurrentPage] = useState("login"); // 'login', 'main', 'combined'

  // State
  const [modelStatus, setModelStatus] = useState(null);
  const [dataSummary, setDataSummary] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [shapeFunctions, setShapeFunctions] = useState([]);
  const [predictionsData, setPredictionsData] = useState(null);
  const [hourlyPattern, setHourlyPattern] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [userSavedEdits, setUserSavedEdits] = useState({}); // User's saved edits for display

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
    const validateAndLoadUser = async () => {
      const savedUser = localStorage.getItem("currentUser");
      if (savedUser) {
        try {
          const user = JSON.parse(savedUser);
          // Verify user still exists in database by trying to get their info
          try {
            const validatedUser = await apiService.getUser(user.id);
            if (validatedUser) {
              setCurrentUser(user);
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

  const handleLogin = async (name) => {
    const user = await apiService.loginOrCreateUser(name);
    setCurrentUser(user);
    localStorage.setItem("currentUser", JSON.stringify(user));
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

  const fetchModelStatus = async (userOverride = null) => {
    const activeUser = userOverride || currentUser;
    try {
      const status = await apiService.getModelStatus();
      setModelStatus(status);

      // If data is already loaded, fetch summary
      if (status.data_loaded) {
        fetchDataSummary();
        fetchHourlyPattern();
      }

      // If model is trained, fetch metrics and visualizations
      if (status.is_trained) {
        fetchMetrics();
        fetchShapeFunctions({ refreshComparison: false });

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

        // Now fetch predictions data (with user edits already applied)
        fetchPredictionsVsActual();
        await fetchComparisonData();
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

  const fetchMetrics = async () => {
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
      // Also fetch initial comparison data after shape functions are loaded
      if (refreshComparison) {
        await fetchComparisonData();
      }
    } catch (err) {
      console.error("Failed to fetch shape functions:", err);
    } finally {
      setChartLoading(false);
    }
  };

  const fetchPredictionsVsActual = async () => {
    try {
      const data = await apiService.getPredictionsVsActual();
      setPredictionsData(data);
    } catch (err) {
      console.error("Failed to fetch predictions vs actual:", err);
    }
  };

  const fetchComparisonData = async () => {
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

  const handleShapeFunctionsEdit = async (editedShapeFunctions) => {
    try {
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

      // Fetch updated comparison data
      await fetchComparisonData();
    } catch (err) {
      setError(
        "Failed to apply shape function edits: " +
          (err.response?.data?.detail || err.message),
      );
    } finally {
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

      // Fetch updated comparison data (should show same values for both)
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

      // Fetch updated comparison data
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

  const handleLoadData = async () => {
    try {
      setDataLoading(true);
      setError(null);
      await apiService.loadData();
      await fetchModelStatus();
      await fetchDataSummary();
      await fetchHourlyPattern();
    } catch (err) {
      setError(
        "Failed to load data: " + (err.response?.data?.detail || err.message),
      );
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
        await fetchShapeFunctions();
        await fetchPredictionsVsActual();
      }
    } catch (err) {
      setError(
        "Failed to train model: " + (err.response?.data?.detail || err.message),
      );
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
        "Prediction failed: " + (err.response?.data?.detail || err.message),
      );
      return null;
    }
  };

  // Render login page
  if (currentPage === "login") {
    return <UserLogin onLogin={handleLogin} />;
  }

  // Render combined results page
  if (currentPage === "combined") {
    return (
      <CombinedResultsPage
        onBack={() => setCurrentPage("main")}
        onResetDatabase={handleResetDatabase}
        currentUser={currentUser}
      />
    );
  }

  // Render main app
  return (
    <div className="min-h-screen bg-gray-50">
      <Header modelStatus={modelStatus} />

      {/* User Bar */}
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
                <span>View Combined Results</span>
              </button>
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
        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg flex items-center justify-between">
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
        )}

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Controls */}
          <div className="space-y-6">
            <TrainingPanel
              onLoadData={handleLoadData}
              onTrainModel={handleTrainModel}
              loading={loading || dataLoading}
              modelStatus={modelStatus}
            />

            <DataSummaryCard summary={dataSummary} loading={dataLoading} />

            <MetricsCard metrics={metrics} loading={loading} />
          </div>

          {/* Right Column - Visualizations */}
          <div className="lg:col-span-2 space-y-6">
            <PredictionForm
              onPredict={handlePredict}
              loading={loading}
              modelTrained={modelStatus?.is_trained}
            />

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
            loading={chartLoading}
            onShapeFunctionsEdit={handleShapeFunctionsEdit}
            onReset={handleResetShapeFunctions}
            onFeatureReset={handleResetFeature}
            initialEditedPoints={userSavedEdits}
          />
        </div>

        {/* Prediction Comparison Section */}
        <div className="mt-6">
          <PredictionComparisonChart
            comparisonData={comparisonData}
            loading={comparisonLoading}
            currentUser={currentUser}
          />
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>
            Bike Rental Prediction using IGANN
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

      {/* Deletion Notification Popup */}
      {showNotificationPopup && deletionNotifications.length > 0 && (
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
      )}
    </div>
  );
}

export default App;
