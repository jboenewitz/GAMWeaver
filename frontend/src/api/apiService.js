import axios from "axios";

const API_BASE_URL = "/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const apiService = {
  // Health check
  healthCheck: async () => {
    const response = await api.get("/health");
    return response.data;
  },

  // Data endpoints
  loadData: async () => {
    const response = await api.post("/data/load");
    return response.data;
  },

  getDataSummary: async () => {
    const response = await api.get("/data/summary");
    return response.data;
  },

  getDistributions: async () => {
    const response = await api.get("/data/distributions");
    return response.data;
  },

  getHourlyPattern: async () => {
    const response = await api.get("/data/hourly-pattern");
    return response.data;
  },

  // Model endpoints
  trainModel: async (params = {}) => {
    const response = await api.post("/model/train", params);
    return response.data;
  },

  getModelStatus: async () => {
    const response = await api.get("/model/status");
    return response.data;
  },

  getModelMetrics: async () => {
    const response = await api.get("/model/metrics");
    return response.data;
  },

  getShapeFunctions: async () => {
    const response = await api.get("/model/shape-functions");
    return response.data;
  },

  getPredictionsVsActual: async () => {
    const response = await api.get("/model/predictions-vs-actual");
    return response.data;
  },

  // Interactive shape function endpoints
  updateShapeFunctions: async (editedShapeFunctions) => {
    const response = await api.post("/model/update-shape-functions", {
      edited_shape_functions: editedShapeFunctions,
    });
    return response.data;
  },

  getPredictionsComparison: async () => {
    const response = await api.get("/model/predictions-comparison");
    return response.data;
  },

  resetShapeFunctions: async () => {
    const response = await api.post("/model/reset-shape-functions");
    return response.data;
  },

  // Prediction endpoints
  predict: async (inputData) => {
    const response = await api.post("/predict", inputData);
    return response.data;
  },

  batchPredict: async (predictions) => {
    const response = await api.post("/predict/batch", { predictions });
    return response.data;
  },
};

export default apiService;
