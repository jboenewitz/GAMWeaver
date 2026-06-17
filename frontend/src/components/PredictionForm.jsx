import React, { useEffect, useMemo, useState } from "react";
import { createTranslator } from "../i18n";

const roundToTwoDecimals = (value) => Math.round(value * 100) / 100;

const getValidNumericValue = (feature) => {
  const min = Number.isFinite(feature.min_value) ? feature.min_value : undefined;
  const max = Number.isFinite(feature.max_value) ? feature.max_value : undefined;
  let value = Number.isFinite(feature.default_value) ? feature.default_value : 0;

  if (min !== undefined && value < min) {
    value = min;
  }
  if (max !== undefined && value > max) {
    value = max;
  }

  return roundToTwoDecimals(value);
};

const getValidCategoricalValue = (feature) => {
  const options = feature.categorical_options || [];
  if (!options.length) {
    return "";
  }
  return options.includes(feature.default_value)
    ? feature.default_value
    : options[0];
};

const buildInitialFormData = (featureSchema = []) => {
  const initial = {};
  for (const feature of featureSchema) {
    if (feature.feature_type === "numeric") {
      initial[feature.name] = getValidNumericValue(feature);
    } else {
      initial[feature.name] = getValidCategoricalValue(feature);
    }
  }
  return initial;
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
        target: targetColumn || "Target",
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
      [feature.name]:
        feature.feature_type === "numeric"
          ? value === ""
            ? ""
            : Number(value)
          : value,
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
          {featureSchema.map((feature) => (
            <div key={feature.name}>
              <label className="label">{feature.name}</label>
              {feature.feature_type === "categorical" ? (
                <select
                  value={formData[feature.name] ?? ""}
                  onChange={(e) => handleChange(feature, e.target.value)}
                  className="select-field"
                >
                  {(feature.categorical_options || []).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={formData[feature.name] ?? ""}
                  onChange={(e) => handleChange(feature, e.target.value)}
                  className="input-field"
                  min={
                    Number.isFinite(feature.min_value) ? feature.min_value : undefined
                  }
                  max={
                    Number.isFinite(feature.max_value) ? feature.max_value : undefined
                  }
                  step="any"
                />
              )}
            </div>
          ))}
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
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600">
              {t("prediction.predictedTarget", {
                target: targetColumn || "Target",
              })}
            </div>
            <div className="text-4xl font-bold text-primary-600 mt-1">
              {Math.round(prediction)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
