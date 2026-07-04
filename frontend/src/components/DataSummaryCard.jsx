import React from "react";
import { createTranslator, getNumberLocale } from "../i18n";

const DataSummaryCard = ({ summary, loading, language = "en" }) => {
  const t = createTranslator(language);
  const numberLocale = getNumberLocale(language);

  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/2 mb-4"></div>
        <div className="space-y-2">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 rounded w-2/3"></div>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          {t("dataSummary.title")}
        </h3>
        <p className="text-gray-500 text-center py-4">
          {t("dataSummary.empty")}
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">
        {t("dataSummary.title")}
      </h3>

      <div className="space-y-4">
        <div className="flex justify-between items-center py-2 border-b border-gray-100">
          <span className="text-gray-600">{t("dataSummary.totalRecords")}</span>
          <span className="font-semibold text-primary-600">
            {summary.total_records?.toLocaleString(numberLocale)}
          </span>
        </div>

        <div>
          <div className="text-sm text-gray-500 mb-2">
            {t("dataSummary.targetVariable")} (
            {summary.target_column || t("dataSummary.targetFallback")})
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500">{t("dataSummary.mean")}</div>
              <div className="font-medium">
                {summary.target_stats?.mean?.toFixed(1)}
              </div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500">{t("dataSummary.stdDev")}</div>
              <div className="font-medium">
                {summary.target_stats?.std?.toFixed(1)}
              </div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500">{t("dataSummary.min")}</div>
              <div className="font-medium">
                {summary.target_stats?.min?.toFixed(0)}
              </div>
            </div>
            <div className="bg-gray-50 rounded p-2">
              <div className="text-gray-500">{t("dataSummary.max")}</div>
              <div className="font-medium">
                {summary.target_stats?.max?.toFixed(0)}
              </div>
            </div>
          </div>
        </div>

        <div>
          <div className="text-sm text-gray-500 mb-2">
            {t("dataSummary.features")} ({summary.features?.length})
          </div>
          <div className="flex flex-wrap gap-1">
            {summary.numeric_features?.map((f) => (
              <span
                key={f}
                className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full"
              >
                {f}
              </span>
            ))}
            {summary.categorical_features?.map((f) => (
              <span
                key={f}
                className="px-2 py-1 bg-green-100 text-green-700 text-xs rounded-full"
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataSummaryCard;
