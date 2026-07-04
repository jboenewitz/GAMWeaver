import React from "react";
import { createTranslator, getNumberLocale } from "../i18n";

const Header = ({ modelStatus, language = "en" }) => {
  const totalRecords =
    (modelStatus?.train_size || 0) + (modelStatus?.test_size || 0);
  const t = createTranslator(language);

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/80 bg-white/88 backdrop-blur-xl">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary-400 to-transparent" />
      <div className="container mx-auto px-4">
        <div className="flex min-h-[76px] items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              GAMWeaver
            </h1>
            <p className="text-sm text-slate-600">
              {t("header.subtitle")}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium ${
                modelStatus?.is_trained
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-amber-200 bg-amber-50 text-amber-700"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  modelStatus?.is_trained ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              {modelStatus?.is_trained
                ? t("header.modelTrained")
                : t("header.modelNotTrained")}
            </div>

            {modelStatus?.data_loaded && (
              <div className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm font-medium text-slate-700">
                {t("header.recordsLoaded", {
                  count: totalRecords.toLocaleString(getNumberLocale(language)),
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
