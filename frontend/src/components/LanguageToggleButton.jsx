import React from "react";

const LanguageToggleButton = ({ language, onToggle, title }) => {
  const isEnglish = language === "en";

  return (
    <button
      type="button"
      onClick={onToggle}
      title={title}
      aria-label={title}
      className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white/70 px-3 py-2 text-sm text-slate-700 transition-colors hover:bg-white"
    >
      <svg
        className="h-4 w-4 text-slate-500"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.5 9m6.548 5.5A18.022 18.022 0 0016.5 9m-6.452 5.5L12 21l1.952-6.5M12 21H6m6 0h6"
        />
      </svg>
      <span className={isEnglish ? "font-semibold text-slate-900" : "text-slate-500"}>
        EN
      </span>
      <span className="text-slate-300">/</span>
      <span className={!isEnglish ? "font-semibold text-slate-900" : "text-slate-500"}>
        DE
      </span>
    </button>
  );
};

export default LanguageToggleButton;
