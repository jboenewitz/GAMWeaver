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
  healthCheck: async () => {
    const response = await api.get("/health");
    return response.data;
  },

  getUserContext: async (userId) => {
    const response = await api.get(`/users/${userId}/context`);
    return response.data;
  },

  listDatasets: async () => {
    const response = await api.get("/datasets");
    return response.data.datasets || [];
  },

  selectDataset: async (userId, datasetId) => {
    const response = await api.post(`/users/${userId}/active-dataset`, {
      dataset_id: datasetId,
    });
    return response.data;
  },

  inspectDatasetUpload: async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await api.post("/admin/datasets/inspect", formData, {
      headers: {
        ...adminHeaders(),
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  uploadDataset: async ({ file, targetColumn, displayName, uploadedByUserId }) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_column", targetColumn);
    if (displayName) {
      formData.append("display_name", displayName);
    }
    if (uploadedByUserId !== undefined && uploadedByUserId !== null) {
      formData.append("uploaded_by_user_id", String(uploadedByUserId));
    }

    const response = await api.post("/admin/datasets", formData, {
      headers: {
        ...adminHeaders(),
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  },

  getHourlyPattern: async (datasetId) => {
    const response = await api.get(`/datasets/${datasetId}/hourly-pattern`);
    return response.data;
  },

  trainDataset: async (datasetId, params = {}) => {
    const response = await api.post(`/datasets/${datasetId}/train`, params);
    return response.data;
  },

  getModelMetrics: async (modelVersionId) => {
    const response = await api.get(`/model-versions/${modelVersionId}/metrics`);
    return response.data;
  },

  getShapeFunctions: async (modelVersionId) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/shape-functions`,
    );
    return response.data;
  },

  getPredictionsVsActual: async (modelVersionId) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/predictions-vs-actual`,
    );
    return response.data;
  },

  getPredictionsComparison: async (modelVersionId, userId) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/predictions-comparison`,
      {
        params: { user_id: userId },
      },
    );
    return response.data;
  },

  previewPredictionsComparison: async (
    modelVersionId,
    userId,
    editedShapeFunctions,
  ) => {
    const response = await api.post(
      `/model-versions/${modelVersionId}/preview-comparison`,
      {
        edited_shape_functions: editedShapeFunctions,
      },
      {
        params: { user_id: userId },
      },
    );
    return response.data;
  },

  predict: async (modelVersionId, inputFeatures) => {
    const response = await api.post(`/model-versions/${modelVersionId}/predict`, {
      input_features: inputFeatures,
    });
    return response.data;
  },

  getUserEdits: async (userId, modelVersionId) => {
    const response = await api.get(
      `/users/${userId}/model-versions/${modelVersionId}/edits`,
    );
    return response.data;
  },

  saveUserEdits: async (userId, modelVersionId, editedShapeFunctions) => {
    const response = await api.post(
      `/users/${userId}/model-versions/${modelVersionId}/edits`,
      {
        edited_shape_functions: editedShapeFunctions,
      },
    );
    return response.data;
  },

  clearUserEdits: async (userId, modelVersionId) => {
    const response = await api.delete(
      `/users/${userId}/model-versions/${modelVersionId}/edits`,
    );
    return response.data;
  },

  clearUserFeatureEdits: async (userId, modelVersionId, featureName) => {
    const response = await api.delete(
      `/users/${userId}/model-versions/${modelVersionId}/edits/${encodeURIComponent(featureName)}`,
    );
    return response.data;
  },

  getCombinedEdits: async (modelVersionId) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/combined/edits`,
    );
    return response.data;
  },

  getEditLogs: async (modelVersionId) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/combined/edit-logs`,
    );
    return response.data;
  },

  getUsersWithEdits: async (modelVersionId) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/combined/users`,
    );
    return response.data;
  },

  getCombinedPredictionsComparison: async (modelVersionId, weighted = true) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/combined/predictions-comparison`,
      {
        params: { weighted },
      },
    );
    return response.data;
  },

  getPerUserShapeFunctions: async (modelVersionId, weighted = true) => {
    const response = await api.get(
      `/model-versions/${modelVersionId}/combined/per-user-shape-functions`,
      {
        params: { weighted },
      },
    );
    return response.data;
  },

  predictCombined: async (modelVersionId, inputFeatures) => {
    const response = await api.post(
      `/model-versions/${modelVersionId}/combined/predict`,
      {
        input_features: inputFeatures,
      },
    );
    return response.data;
  },

  loginUser: async (username, password) => {
    const response = await api.post("/auth/login", { username, password });
    return response.data;
  },

  registerUser: async (username, password, inviteToken) => {
    const response = await api.post("/auth/register", {
      username,
      password,
      invite_token: inviteToken,
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

  resetDatabase: async () => {
    const response = await api.post("/database/reset", null, {
      headers: adminHeaders(),
    });
    return response.data;
  },

  deleteEdit: async (editId, deletedByUserId, reason) => {
    const response = await api.post(
      "/edits/delete",
      {
        edit_id: editId,
        deleted_by_user_id: deletedByUserId,
        reason,
      },
      { headers: adminHeaders() },
    );
    return response.data;
  },

  getUserNotifications: async (userId, modelVersionId = null) => {
    const response = await api.get(`/users/${userId}/notifications`, {
      params: modelVersionId ? { model_version_id: modelVersionId } : {},
    });
    return response.data;
  },

  markNotificationsSeen: async (userId, modelVersionId = null) => {
    const response = await api.post(
      `/users/${userId}/notifications/mark-seen`,
      null,
      {
        params: modelVersionId ? { model_version_id: modelVersionId } : {},
      },
    );
    return response.data;
  },

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

  getAdminUsers: async () => {
    const response = await api.get("/admin/users", {
      headers: adminHeaders(),
    });
    return response.data;
  },

  createAdminUser: async (username, password) => {
    const response = await api.post(
      "/admin/users",
      { username, password },
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
