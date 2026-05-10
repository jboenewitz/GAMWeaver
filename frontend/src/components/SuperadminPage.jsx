import React, { useEffect, useMemo, useState } from "react";
import apiService from "../api/apiService";

const SuperadminPage = ({ onBack, onOpenCombined }) => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [creating, setCreating] = useState(false);
  const [inviteLink, setInviteLink] = useState("");
  const [inviteExpires, setInviteExpires] = useState("");

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

  return (
    <div className="glass-root min-h-screen">
      <header className="glass-toolbar">
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
              className="px-4 py-2 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors shadow-md"
            >
              Open Combined Results
            </button>
            <button
              onClick={() => setShowCreateForm((prev) => !prev)}
              className="px-4 py-2 bg-primary-600 text-white rounded-xl hover:bg-primary-700 transition-colors flex items-center gap-2 shadow-md"
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

        <section className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">Invite Link</h2>
            <button
              onClick={handleCreateInvite}
              className="px-4 py-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition-colors shadow-md"
            >
              Generate Invite
            </button>
          </div>
          {inviteLink ? (
            <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
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

        {showCreateForm && (
          <section className="card">
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
                  className="input-field"
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
                  className="input-field"
                />
              </div>
              <button
                type="submit"
                disabled={creating}
                className="px-4 py-2 btn-primary disabled:opacity-50"
              >
                {creating ? "Creating..." : "Create User"}
              </button>
            </form>
          </section>
        )}

        <section className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-800">
              Users ({users.length})
            </h2>
            <button
              onClick={fetchUsers}
              disabled={loading}
              className="px-4 py-2 border border-gray-300 rounded-xl hover:bg-gray-50 transition-colors"
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
                  className="flex items-center justify-between border rounded-xl px-4 py-3 bg-white/45"
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
    </div>
  );
};

export default SuperadminPage;
