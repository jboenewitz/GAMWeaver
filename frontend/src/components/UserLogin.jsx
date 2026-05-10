import React, { useMemo, useState } from "react";

const formatAuthError = (err, fallback) => {
  const detail = err?.response?.data?.detail;
  const message = err?.response?.data?.message;

  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const joined = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (typeof item?.msg === "string") return item.msg;
        return "";
      })
      .filter(Boolean)
      .join("; ");
    if (joined) return joined;
  }
  if (typeof message === "string" && message.trim()) return message;
  if (typeof err?.message === "string" && err.message === "Network Error") {
    return "Unable to reach the server. Please try again.";
  }
  if (typeof err?.message === "string" && err.message.trim()) {
    return err.message;
  }
  return fallback;
};

function UserLogin({ onLogin, onRegister }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const inviteToken = useMemo(() => {
    try {
      return new URLSearchParams(window.location.search).get("invite") || "";
    } catch {
      return "";
    }
  }, []);
  const [mode, setMode] = useState(inviteToken ? "register" : "login");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("Please enter username and password");
      return;
    }
    if (mode === "register" && !inviteToken) {
      setError("Registration requires a valid invite link");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (mode === "login") {
        await onLogin(username.trim(), password);
      } else {
        await onRegister(username.trim(), password, inviteToken);
      }
    } catch (err) {
      const fallback =
        mode === "login" ? "Failed to login" : "Failed to register";
      setError(formatAuthError(err, fallback));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-900 via-primary-900 to-primary-700 flex items-center justify-center p-4 sm:p-6">
      <div className="pointer-events-none absolute -top-24 -left-24 h-72 w-72 rounded-full bg-cyan-300/30 blur-3xl" />
      <div className="pointer-events-none absolute top-1/2 -right-24 h-80 w-80 -translate-y-1/2 rounded-full bg-primary-300/25 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 left-1/3 h-64 w-64 rounded-full bg-sky-200/20 blur-3xl" />

      <div className="relative w-full max-w-md rounded-3xl border border-white/30 bg-white/15 p-7 shadow-2xl shadow-slate-900/40 backdrop-blur-2xl sm:p-8">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 inline-flex h-16 w-16 items-center justify-center rounded-2xl border border-white/40 bg-white/20 shadow-lg shadow-slate-900/20 backdrop-blur-md">
            <svg
              className="h-8 w-8 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
              />
            </svg>
          </div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary-100/90">
            GAMWeaver
          </p>
          <h1 className="mt-2 text-3xl font-bold text-white">
            The Interactive GAM Editor
          </h1>
          <p className="mt-2 text-sm text-slate-100/85">
            {mode === "login"
              ? "Sign in with your credentials"
              : "Register with an invite link"}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label
              htmlFor="username"
              className="mb-2 block text-sm font-medium text-slate-100"
            >
              Username
            </label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Your username"
              className="w-full rounded-xl border border-white/30 bg-white/10 px-4 py-3 text-white placeholder:text-slate-200/75 transition-all duration-200 focus:border-primary-200 focus:bg-white/15 focus:outline-none focus:ring-2 focus:ring-primary-300/60"
              disabled={loading}
              autoFocus
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="mb-2 block text-sm font-medium text-slate-100"
            >
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Your password"
              className="w-full rounded-xl border border-white/30 bg-white/10 px-4 py-3 text-white placeholder:text-slate-200/75 transition-all duration-200 focus:border-primary-200 focus:bg-white/15 focus:outline-none focus:ring-2 focus:ring-primary-300/60"
              disabled={loading}
            />
          </div>

          {mode === "register" && !inviteToken && (
            <div className="rounded-xl border border-amber-200/40 bg-amber-100/20 p-3 text-sm text-amber-100 backdrop-blur-sm">
              Registration is only possible via an invite link.
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-rose-200/45 bg-rose-200/20 p-3 text-sm text-rose-100 backdrop-blur-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !username.trim() || !password}
            className="flex w-full items-center justify-center rounded-xl border border-white/30 bg-gradient-to-r from-primary-500 to-cyan-500 px-4 py-3 font-semibold text-white shadow-lg shadow-primary-900/30 transition-all duration-200 hover:from-primary-400 hover:to-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (
              <>
                <svg
                  className="-ml-1 mr-3 h-5 w-5 animate-spin text-white"
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
                {mode === "login" ? "Signing in..." : "Registering..."}
              </>
            ) : mode === "login" ? (
              "Sign In"
            ) : (
              "Register"
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

export default UserLogin;
