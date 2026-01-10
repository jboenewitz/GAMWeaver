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

function App() {
  // State
  const [modelStatus, setModelStatus] = useState(null);
  const [dataSummary, setDataSummary] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [shapeFunctions, setShapeFunctions] = useState([]);
  const [predictionsData, setPredictionsData] = useState(null);
  const [hourlyPattern, setHourlyPattern] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);

  // Loading states
  const [loading, setLoading] = useState(false);
  const [dataLoading, setDataLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  // Error state
  const [error, setError] = useState(null);

  // Fetch initial status
  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
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
        fetchShapeFunctions();
        fetchPredictionsVsActual();
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

  const fetchShapeFunctions = async () => {
    try {
      setChartLoading(true);
      const response = await apiService.getShapeFunctions();
      setShapeFunctions(response.shape_functions || []);
      // Also fetch initial comparison data after shape functions are loaded
      await fetchComparisonData();
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

      // Send edited shape functions to backend
      await apiService.updateShapeFunctions(editedShapeFunctions);

      // Fetch updated comparison data
      await fetchComparisonData();
    } catch (err) {
      setError(
        "Failed to apply shape function edits: " +
          (err.response?.data?.detail || err.message)
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

      // Fetch updated comparison data (should show same values for both)
      await fetchComparisonData();
    } catch (err) {
      setError(
        "Failed to reset shape functions: " +
          (err.response?.data?.detail || err.message)
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
        "Failed to load data: " + (err.response?.data?.detail || err.message)
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
        "Failed to train model: " + (err.response?.data?.detail || err.message)
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
        "Prediction failed: " + (err.response?.data?.detail || err.message)
      );
      return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header modelStatus={modelStatus} />

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
          />
        </div>

        {/* Prediction Comparison Section */}
        <div className="mt-6">
          <PredictionComparisonChart
            comparisonData={comparisonData}
            loading={comparisonLoading}
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
    </div>
  );
}

export default App;
