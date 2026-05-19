import React, { useEffect, useMemo, useState } from "react";
import apiService from "../api/apiService";

const SuperadminPage = ({ onBack, onOpenCombined, onResetDatabase }) => {
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
        err.response?.data?.detail || err.message || "Failed to load users",
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
      setError("Username and password are required");
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
        err.response?.data?.detail || err.message || "Failed to create user",
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
        err.response?.data?.detail || err.message || "Failed to create invite",
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
        err.response?.data?.detail || err.message || "Failed to reset database",
      );
      setResetting(false);
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
              Superadmin Overview
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onOpenCombined}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              Open Combined Results
            </button>
            <button
              onClick={() => setShowCreateForm((prev) => !prev)}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
            >
              <span className="text-lg">+</span>
              <span>Create User</span>
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

        <section className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">Invite Link</h2>
            <button
              onClick={handleCreateInvite}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
            >
              Generate Invite
            </button>
          </div>
          {inviteLink ? (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <div className="text-sm text-gray-600 mb-2">
                Invite link copied to clipboard.
              </div>
              <div className="break-all text-sm text-gray-800 font-mono">
                {inviteLink}
              </div>
              {inviteExpires && (
                <div className="text-xs text-gray-500 mt-2">
                  Expires: {new Date(inviteExpires).toLocaleString()}
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">
              Generate an invite link to allow new registrations.
            </p>
          )}
        </section>

        <section className="bg-white rounded-xl shadow-md p-6 border border-red-100">
          <h2 className="text-lg font-semibold text-gray-800 mb-2">
            System Reset
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Permanently removes all users, all saved edits, uploaded CSV files,
            and extracted feature state for the current backend environment.
          </p>
          <button
            onClick={() => setShowResetConfirm(true)}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Reset Database
          </button>
        </section>

        {showCreateForm && (
          <section className="bg-white rounded-xl shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              Create User
            </h2>
            <form onSubmit={handleCreateUser} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Username
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
                  Password
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
                {creating ? "Creating..." : "Create User"}
              </button>
            </form>
          </section>
        )}

        <section className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              Users ({users.length})
            </h2>
            <button
              onClick={fetchUsers}
              disabled={loading}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Refresh
            </button>
          </div>
          {loading ? (
            <p className="text-gray-500">Loading users...</p>
          ) : users.length === 0 ? (
            <p className="text-gray-500">No users yet.</p>
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
                      Created: {new Date(user.created_at).toLocaleString()}
                    </div>
                  </div>
                  {user.is_superadmin && (
                    <span className="text-xs px-2 py-1 rounded-full bg-amber-100 text-amber-700">
                      Superadmin
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
              Reset Database?
            </h3>
            <p className="mb-6 text-sm text-slate-600">
              This permanently deletes users, edits, uploaded CSV files, and
              extracted feature state. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowResetConfirm(false)}
                disabled={resetting}
                className="px-4 py-2 border border-slate-300 rounded-lg hover:bg-slate-100 transition-colors"
              >
                Cancel
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
                    Resetting...
                  </>
                ) : (
                  "Yes, Reset Everything"
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
