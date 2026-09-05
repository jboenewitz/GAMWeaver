import React, { useEffect, useMemo, useState } from "react";
import { createTranslator } from "../i18n";
import CompetenceLevelsTable from "./CompetenceLevelsTable";
import { getFeatureDisplayName } from "../utils/featureDisplay";
import {
  buildInitialFormData,
  coerceFeatureInputValue,
  featureUsesChoiceInput,
  getFeatureChoiceOptions,
  getFeatureResponseOptions,
} from "../utils/predictionInputs";

const SCALE_TOOLTIP_WIDTH_CLASS = "w-72";

const getFeatureInputId = (featureName) =>
  `prediction-field-${String(featureName || "")
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase()}`;

const formatTooltipNumber = (value) => {
  if (!Number.isFinite(value)) return "";
  return Number.isInteger(value)
    ? String(value)
    : String(Math.round(value * 100) / 100);
};

const buildScaleTooltipText = (feature, t) => {
  const responseOptions = getFeatureResponseOptions(feature)
    .map((option) => ({
      value: String(option?.value ?? "").trim(),
      label: String(option?.label ?? "").trim(),
    }))
    .filter((option) => option.value);

  if (responseOptions.length > 0) {
    return [
      t("prediction.scaleTooltipOptionsIntro"),
      ...responseOptions.map((option) =>
        option.label ? `${option.value} = ${option.label}` : option.value,
      ),
    ].join("\n");
  }

  if (feature.feature_type === "categorical") {
    const values = (feature.categorical_options || [])
      .map((option) => String(option ?? "").trim())
      .filter(Boolean);
    if (values.length > 0) {
      return t("prediction.scaleTooltipAvailableValues", {
        values: values.join(", "),
      });
    }
    return "";
  }

  const min = Number.isFinite(feature.min_value) ? feature.min_value : undefined;
  const max = Number.isFinite(feature.max_value) ? feature.max_value : undefined;

  if (min !== undefined && max !== undefined && min === max) {
    return t("prediction.scaleTooltipSingleValue", {
      value: formatTooltipNumber(min),
    });
  }

  if (min !== undefined && max !== undefined) {
    return t("prediction.scaleTooltipRange", {
      min: formatTooltipNumber(min),
      max: formatTooltipNumber(max),
    });
  }

  return "";
};

const FeatureScaleTooltip = ({ feature, label, t }) => {
  const tooltipText = buildScaleTooltipText(feature, t);

  if (!tooltipText) return null;

  return (
    <div className="group relative inline-flex">
      <button
        type="button"
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-slate-300 bg-white text-[11px] font-semibold text-slate-500 transition-colors hover:border-primary-500 hover:text-primary-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
        aria-label={t("prediction.scaleHelpLabel", { feature: label })}
        title={tooltipText}
      >
        i
      </button>
      <span
        role="tooltip"
        className={`pointer-events-none absolute left-0 top-full z-20 mt-2 hidden ${SCALE_TOOLTIP_WIDTH_CLASS} whitespace-pre-line rounded-lg bg-slate-900 px-3 py-2 text-xs leading-5 text-white shadow-xl group-hover:block group-focus-within:block`}
      >
        {tooltipText}
      </span>
    </div>
  );
};

const PredictionForm = ({
  onPredict,
  loading,
  modelTrained,
  featureSchema = [],
  targetColumn,
  language = "en",
}) => {
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const hasFeatures = featureSchema && featureSchema.length > 0;
  const t = createTranslator(language);

  const title = useMemo(
    () =>
      t("prediction.predictTarget", {
        target: targetColumn || t("common.target"),
      }),
    [t, targetColumn],
  );

  useEffect(() => {
    setFormData(buildInitialFormData(featureSchema));
    setPrediction(null);
  }, [featureSchema]);

  const handleChange = (feature, value) => {
    setFormData((prev) => ({
      ...prev,
      [feature.name]: coerceFeatureInputValue(feature, value),
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await onPredict(formData);
    if (result) {
      setPrediction(result.predicted_count);
    }
  };

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-2">
        {t("prediction.title")}
      </h3>
      <p className="text-sm text-gray-500 mb-4">
        {t("prediction.description")}
      </p>

      {!modelTrained && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
          {t("prediction.trainFirst")}
        </div>
      )}

      {!hasFeatures && (
        <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded-lg text-gray-600 text-sm">
          {t("prediction.noSchema")}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {featureSchema.map((feature) => {
            const featureLabel = getFeatureDisplayName(feature);
            const inputId = getFeatureInputId(feature.name);
            const choiceOptions = getFeatureChoiceOptions(feature);
            const usesChoiceInput = featureUsesChoiceInput(feature);

            return (
              <div key={feature.name}>
                <div className="mb-1 flex items-center gap-2">
                  <label htmlFor={inputId} className="label mb-0">
                    {featureLabel}
                  </label>
                  <FeatureScaleTooltip
                    feature={feature}
                    label={featureLabel}
                    t={t}
                  />
                </div>
                {usesChoiceInput ? (
                  <select
                    id={inputId}
                    value={String(formData[feature.name] ?? "")}
                    onChange={(e) => handleChange(feature, e.target.value)}
                    className="select-field"
                  >
                    {choiceOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id={inputId}
                    type="number"
                    value={formData[feature.name] ?? ""}
                    onChange={(e) => handleChange(feature, e.target.value)}
                    className="input-field"
                    min={
                      Number.isFinite(feature.min_value)
                        ? feature.min_value
                        : undefined
                    }
                    max={
                      Number.isFinite(feature.max_value)
                        ? feature.max_value
                        : undefined
                    }
                    step="any"
                  />
                )}
              </div>
            );
          })}
        </div>

        <button
          type="submit"
          disabled={!modelTrained || loading || !hasFeatures}
          className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? t("prediction.predicting") : title}
        </button>
      </form>

      {prediction !== null && (
        <>
          <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
            <div className="text-center">
              <div className="text-sm text-gray-600">
                {t("prediction.predictedTarget", {
                  target: targetColumn || t("common.target"),
                })}
              </div>
              <div className="text-4xl font-bold text-primary-600 mt-1">
                {Math.round(prediction)}
              </div>
            </div>
          </div>

          <CompetenceLevelsTable language={language} />
        </>
      )}
    </div>
  );
};

export default PredictionForm;
