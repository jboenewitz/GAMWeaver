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

  // ==================== User endpoints ====================

  loginOrCreateUser: async (name) => {
    const response = await api.post("/users/login", { name });
    return response.data;
  },

  getAllUsers: async () => {
    const response = await api.get("/users");
    return response.data;
  },

  getUser: async (userId) => {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },

  getUserEdits: async (userId) => {
    const response = await api.get(`/users/${userId}/edits`);
    return response.data;
  },

  saveUserEdits: async (userId, editedShapeFunctions) => {
    const response = await api.post(`/users/${userId}/edits`, {
      edited_shape_functions: editedShapeFunctions,
    });
    return response.data;
  },

  clearUserEdits: async (userId) => {
    const response = await api.delete(`/users/${userId}/edits`);
    return response.data;
  },

  clearUserFeatureEdits: async (userId, featureName) => {
    const response = await api.delete(
      `/users/${userId}/edits/${encodeURIComponent(featureName)}`,
    );
    return response.data;
  },

  loadUserEditsToModel: async (userId) => {
    const response = await api.post(`/users/${userId}/load-edits`);
    return response.data;
  },

  // ==================== Combined results endpoints ====================

  getCombinedEdits: async () => {
    const response = await api.get("/combined/edits");
    return response.data;
  },

  getEditLogs: async () => {
    const response = await api.get("/combined/edit-logs");
    return response.data;
  },

  getUsersWithEdits: async () => {
    const response = await api.get("/combined/users");
    return response.data;
  },

  getCombinedPredictionsComparison: async (weighted = true) => {
    const response = await api.get("/combined/predictions-comparison", {
      params: { weighted },
    });
    return response.data;
  },

  getPerUserShapeFunctions: async (weighted = true) => {
    const response = await api.get("/combined/per-user-shape-functions", {
      params: { weighted },
    });
    return response.data;
  },

  predictCombined: async (inputData) => {
    const response = await api.post("/combined/predict", inputData);
    return response.data;
  },

  // ==================== Database management ====================

  resetDatabase: async () => {
    const response = await api.post("/database/reset");
    return response.data;
  },

  // ==================== Edit deletion ====================

  deleteEdit: async (editId, deletedByUserId, reason) => {
    const response = await api.post("/edits/delete", {
      edit_id: editId,
      deleted_by_user_id: deletedByUserId,
      reason: reason,
    });
    return response.data;
  },

  // ==================== Notifications ====================

  getUserNotifications: async (userId) => {
    const response = await api.get(`/users/${userId}/notifications`);
    return response.data;
  },

  markNotificationsSeen: async (userId) => {
    const response = await api.post(`/users/${userId}/notifications/mark-seen`);
    return response.data;
  },

  // ==================== User preferences ====================

  getUserPreferences: async (userId) => {
    const response = await api.get(`/users/${userId}/preferences`);
    return response.data.preferences;
  },

  updateUserPreferences: async (userId, preferences) => {
    const response = await api.put(`/users/${userId}/preferences`, {
      preferences,
    });
    return response.data.preferences;
  },
};

export default apiService;
