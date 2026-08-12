import React from "react";
import { createTranslator } from "../i18n";

const COMPETENCE_LEVELS = {
  en: [
    {
      level: "V",
      score: "from 615 points",
      assignment: "Optimal standard",
    },
    {
      level: "IV",
      score: "540-614 points",
      assignment: "Standard Plus",
    },
    {
      level: "III",
      score: "465-539 points",
      assignment: "Standard",
    },
    {
      level: "II",
      score: "390-464 points",
      assignment: "Minimum standard",
    },
    {
      level: "I",
      score: "below 390 points",
      assignment: "Minimum standard not achieved",
    },
  ],
  de: [
    {
      level: "V",
      score: "ab 615 Punkten",
      assignment: "Optimalstandard",
    },
    {
      level: "IV",
      score: "540-614 Punkte",
      assignment: "Regelstandard Plus",
    },
    {
      level: "III",
      score: "465-539 Punkte",
      assignment: "Regelstandard",
    },
    {
      level: "II",
      score: "390-464 Punkte",
      assignment: "Mindeststandard",
    },
    {
      level: "I",
      score: "unter 390 Punkten",
      assignment: "Mindeststandard nicht erreicht",
    },
  ],
};

const CompetenceLevelsTable = ({ language = "en" }) => {
  const t = createTranslator(language);
  const rows = COMPETENCE_LEVELS[language] || COMPETENCE_LEVELS.en;

  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50/80">
            <tr>
              <th className="px-5 py-4 text-left text-sm font-semibold text-slate-800">
                {t("prediction.competenceLevel")}
              </th>
              <th className="px-5 py-4 text-right text-sm font-semibold text-slate-800">
                {t("prediction.scoreValue")}
              </th>
              <th className="px-5 py-4 text-left text-sm font-semibold text-slate-800">
                {t("prediction.assignment")}
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 bg-white">
            {rows.map((row) => (
              <tr key={row.level}>
                <td className="px-5 py-4 text-2xl font-semibold text-slate-900">
                  {row.level}
                </td>
                <td className="px-5 py-4 text-right text-sm font-semibold text-slate-900">
                  {row.score}
                </td>
                <td className="px-5 py-4 text-sm text-slate-800">
                  {row.assignment}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CompetenceLevelsTable;
