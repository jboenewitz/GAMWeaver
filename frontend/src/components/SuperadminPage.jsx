import React, { useEffect, useMemo, useRef, useState } from "react";
import apiService from "../api/apiService";
import { createTranslator, getDateLocale } from "../i18n";

const SuperadminPage = ({
  onBack,
  onOpenCombined,
  onResetDatabase,
  onExportModel,
  onImportModel,
  language = "en",
}) => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [creating, setCreating] = useState(false);
  const [inviteLink, setInviteLink] = useState("");
  const [inviteExpires, setInviteExpires] = useState("");
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  const appBaseUrl = useMemo(() => {
    const configured = import.meta.env.VITE_PUBLIC_APP_URL;
    if (configured) {
      return configured.endsWith("/") ? configured : `${configured}/`;
    }
    const base = import.meta.env.BASE_URL || "/";
    return `${window.location.origin}${base}`;
  }, []);

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getAdminUsers();
      setUsers(data.users || []);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          t("superadmin.error.loadUsers"),
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  const handleCreateUser = async (e) => {
    e.preventDefault();
    if (!newUsername.trim() || !newPassword) {
      setError(t("superadmin.error.usernamePasswordRequired"));
      return;
    }
    setCreating(true);
    setError(null);
    try {
      await apiService.createAdminUser(newUsername.trim(), newPassword);
      setNewUsername("");
      setNewPassword("");
      setShowCreateForm(false);
      await fetchUsers();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          t("superadmin.error.createUser"),
      );
    } finally {
      setCreating(false);
    }
  };

  const handleCreateInvite = async () => {
    setError(null);
    try {
      const data = await apiService.createInvite();
      const link = `${appBaseUrl}?invite=${encodeURIComponent(data.token)}`;
      setInviteLink(link);
      setInviteExpires(data.expires_at || "");
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(link);
      }
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          t("superadmin.error.createInvite"),
      );
    }
  };

  const handleResetDatabase = async () => {
    if (!onResetDatabase) return;
    setResetting(true);
    setError(null);
    try {
      await onResetDatabase();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          t("superadmin.error.resetDatabase"),
      );
      setResetting(false);
    }
  };

  const handleExportModel = async () => {
    if (!onExportModel) return;
    setExportingModel(true);
    setError(null);
    setModelTransferMessage("");
    try {
      const filename = await onExportModel();
      setModelTransferMessage(`Model exported as ${filename}.`);
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Failed to export model",
      );
    } finally {
      setExportingModel(false);
    }
  };

  const handleOpenImportDialog = () => {
    setError(null);
    setModelTransferMessage("");
    importInputRef.current?.click();
  };

  const handleImportFileChange = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !onImportModel) return;

    setImportingModel(true);
    setError(null);
    setModelTransferMessage("");
    try {
      const result = await onImportModel(file);
      setModelTransferMessage(
        result?.message || "Model imported successfully.",
      );
    } catch (err) {
      setError(
        err.response?.data?.detail || err.message || "Failed to import model",
      );
    } finally {
      setImportingModel(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg
                className="w-6 h-6 text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
            </button>
            <h1 className="text-xl font-bold text-gray-800">
              {t("superadmin.title")}
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onOpenCombined}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              {t("superadmin.openCombined")}
            </button>
            <button
              onClick={() => setShowCreateForm((prev) => !prev)}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
            >
              <span className="text-lg">+</span>
              <span>{t("superadmin.createUser")}</span>
            </button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-6">
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {modelTransferMessage && (
          <div className="p-4 bg-emerald-50 border border-emerald-200 text-emerald-700 rounded-lg">
            {modelTransferMessage}
          </div>
        )}

        <section className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <h2 className="text-lg font-semibold text-gray-800">
                Model Transfer
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                Export the active trained model to JSON or import a previously
                exported model artifact.
              </p>
            </div>
            <input
              ref={importInputRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={handleImportFileChange}
            />
            <div className="flex items-center gap-3">
              <button
                onClick={handleExportModel}
                disabled={exportingModel || importingModel}
                className="px-4 py-2 bg-sky-600 text-white rounded-lg hover:bg-sky-700 transition-colors disabled:opacity-50"
              >
                {exportingModel ? "Exporting..." : "Export Model"}
              </button>
              <button
                onClick={handleOpenImportDialog}
                disabled={importingModel || exportingModel}
                className="px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition-colors disabled:opacity-50"
              >
                {importingModel ? "Importing..." : "Import Model"}
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500">
            Importing a model replaces the current base model and clears saved
            user edits tied to the previous one.
          </p>
        </section>

        <section className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              {t("superadmin.inviteLink")}
            </h2>
            <button
              onClick={handleCreateInvite}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              {t("superadmin.generateInvite")}
            </button>
          </div>
          {inviteLink ? (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-2">
                {t("superadmin.inviteCopied")}
              </div>
              <div className="break-all text-sm text-gray-800 font-mono">
                {inviteLink}
              </div>
              {inviteExpires && (
                <div className="text-xs text-gray-500 mt-2">
                  {t("superadmin.expires")}:{" "}
                  {new Date(inviteExpires).toLocaleString(dateLocale)}
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">
              {t("superadmin.inviteDescription")}
            </p>
          )}
        </section>

        <section className="bg-white rounded-xl shadow-md p-6 border border-red-100">
          <h2 className="text-lg font-semibold text-gray-800 mb-2">
            {t("superadmin.systemReset")}
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            {t("superadmin.systemResetDescription")}
          </p>
          <button
            onClick={() => setShowResetConfirm(true)}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            {t("superadmin.resetDatabase")}
          </button>
        </section>

        {showCreateForm && (
          <section className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              {t("superadmin.createUser")}
            </h2>
            <form onSubmit={handleCreateUser} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t("common.username")}
                </label>
                <input
                  type="text"
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {t("common.password")}
                </label>
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
                />
              </div>
              <button
                type="submit"
                disabled={creating}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50"
              >
                {creating
                  ? t("superadmin.creating")
                  : t("superadmin.createUser")}
              </button>
            </form>
          </section>
        )}

        <section className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              {t("superadmin.users")} ({users.length})
            </h2>
            <button
              onClick={fetchUsers}
              disabled={loading}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              {t("common.refresh")}
            </button>
          </div>
          {loading ? (
            <p className="text-gray-500">{t("superadmin.loadingUsers")}</p>
          ) : users.length === 0 ? (
            <p className="text-gray-500">{t("superadmin.noUsers")}</p>
          ) : (
            <div className="space-y-2">
              {users.map((user) => (
                <div
                  key={user.id}
                  className="flex items-center justify-between border rounded-lg px-4 py-3"
                >
                  <div>
                    <div className="font-medium text-gray-800">{user.name}</div>
                    <div className="text-xs text-gray-500">
                      {t("superadmin.created")}:{" "}
                      {new Date(user.created_at).toLocaleString(dateLocale)}
                    </div>
                  </div>
                  {user.is_superadmin && (
                    <span className="text-xs px-2 py-1 rounded-full bg-amber-100 text-amber-700">
                      {t("superadmin.badge")}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      {showResetConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm">
          <div className="bg-white rounded-xl shadow-xl mx-4 w-full max-w-md p-6">
            <h3 className="mb-3 text-lg font-bold text-slate-800">
              {t("superadmin.resetConfirmTitle")}
            </h3>
            <p className="mb-6 text-sm text-slate-600">
              {t("superadmin.resetConfirmBody")}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowResetConfirm(false)}
                disabled={resetting}
                className="px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-100 transition-colors"
              >
                {t("common.cancel")}
              </button>
              <button
                onClick={handleResetDatabase}
                disabled={resetting}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:opacity-50 flex items-center"
              >
                {resetting ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2 h-4 w-4"
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
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    {t("superadmin.resetting")}
                  </>
                ) : (
                  t("superadmin.resetEverything")
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SuperadminPage;
