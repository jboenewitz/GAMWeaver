import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

const getSuperadminToken = () => localStorage.getItem("superadminToken") || "";

const adminHeaders = () => {
  const token = getSuperadminToken();
  return token ? { "x-superadmin-token": token } : {};
};

export const apiService = {
  // Health check
  healthCheck: async () => {
    const response = await api.get("/health");
    return response.data;
  },

  // Data endpoints
  uploadDataset: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/data/upload", formData, {
      headers: {
        ...adminHeaders(),
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  loadData: async (params = {}) => {
    const response = await api.post("/data/load", params, {
      headers: adminHeaders(),
    });
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
    const response = await api.post("/model/train", params, {
      headers: adminHeaders(),
    });
    return response.data;
  },

  exportModelArtifact: async (includeEdits = false) => {
    const response = await api.get("/model/export", {
      headers: adminHeaders(),
      ...(includeEdits ? { params: { include_edits: true } } : {}),
      responseType: "blob",
    });
    const disposition = response.headers["content-disposition"] || "";
    const match = disposition.match(/filename="([^"]+)"/);
    return {
      blob: response.data,
      filename: match?.[1] || "igann-model.json",
    };
  },

  importModelArtifact: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/model/import", formData, {
      headers: {
        ...adminHeaders(),
        "Content-Type": "multipart/form-data",
      },
    });
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

  getFeatureChartSettings: async (featureName) => {
    const response = await api.get(
      `/model/feature-chart-settings/${encodeURIComponent(featureName)}`,
    );
    return response.data;
  },

  updateFeatureChartSettings: async (featureName, settings) => {
    const response = await api.put(
      `/model/feature-chart-settings/${encodeURIComponent(featureName)}`,
      settings,
      { headers: adminHeaders() },
    );
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

  loginUser: async (username, password) => {
    const response = await api.post("/auth/login", { username, password });
    return response.data;
  },

  registerUser: async (username, password, inviteToken, profession) => {
    const response = await api.post("/auth/register", {
      username,
      password,
      invite_token: inviteToken,
      profession,
    });
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
    const response = await api.post("/database/reset", null, {
      headers: adminHeaders(),
    });
    return response.data;
  },

  // ==================== Edit deletion ====================

  deleteEdit: async (editId, deletedByUserId, reason) => {
    const response = await api.post(
      "/edits/delete",
      {
        edit_id: editId,
        deleted_by_user_id: deletedByUserId,
        reason: reason,
      },
      { headers: adminHeaders() },
    );
    return response.data;
  },

  deleteSubmission: async (submissionId, deletedByUserId, reason) => {
    const response = await api.post(
      "/edits/delete-submission",
      {
        submission_id: submissionId,
        deleted_by_user_id: deletedByUserId,
        reason,
      },
      { headers: adminHeaders() },
    );
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

  // ==================== Superadmin endpoints ====================

  getAdminUsers: async () => {
    const response = await api.get("/admin/users", {
      headers: adminHeaders(),
    });
    return response.data;
  },

  createAdminUser: async (username, password, profession) => {
    const response = await api.post(
      "/admin/users",
      { username, password, profession },
      { headers: adminHeaders() },
    );
    return response.data;
  },

  createInvite: async () => {
    const response = await api.post("/admin/invites", null, {
      headers: adminHeaders(),
    });
    return response.data;
  },
};

export default apiService;
