import React, { useEffect, useMemo, useState } from "react";

const buildDefaultFormData = (predictionFields = []) => {
  const defaults = {};
  for (const field of predictionFields) {
    defaults[field.name] = field.default ?? "";
  }
  return defaults;
};

const PredictionForm = ({
  title = "Make Prediction",
  description = "Set the input feature values and run a prediction.",
  onPredict,
  loading,
  modelTrained,
  predictionFields = [],
  targetLabel = "prediction target",
  submitLabel = "Run Prediction",
  contextKey = "default",
  resultLabel,
  resultUnit = "",
}) => {
  const [formData, setFormData] = useState(() =>
    buildDefaultFormData(predictionFields),
  );
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    setFormData(buildDefaultFormData(predictionFields));
    setPrediction(null);
  }, [predictionFields, contextKey]);

  const hasFields = predictionFields.length > 0;
  const resolvedResultLabel = useMemo(
    () => resultLabel || `Predicted ${targetLabel}`,
    [resultLabel, targetLabel],
  );

  const handleChange = (field, rawValue) => {
    setFormData((prev) => ({
      ...prev,
      [field.name]:
        field.feature_type === "numeric" && rawValue !== ""
          ? Number(rawValue)
          : rawValue,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await onPredict(formData);
    if (result) {
      setPrediction(result.predicted_value);
    }
  };

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-2">{title}</h3>
      <p className="text-sm text-gray-500 mb-4">{description}</p>

      {!modelTrained && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
          Train the selected model version before making predictions.
        </div>
      )}

      {modelTrained && !hasFields && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
          No prediction fields are available for the selected dataset.
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {predictionFields.map((field) => (
            <div key={field.name}>
              <label className="label">{field.label}</label>
              {field.feature_type === "categorical" ? (
                <select
                  name={field.name}
                  value={formData[field.name] ?? ""}
                  onChange={(e) => handleChange(field, e.target.value)}
                  className="select-field"
                >
                  {(field.options || []).map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  name={field.name}
                  value={formData[field.name] ?? ""}
                  onChange={(e) => handleChange(field, e.target.value)}
                  className="input-field"
                  min={field.min}
                  max={field.max}
                  step={field.step || (field.is_integer ? 1 : 0.1)}
                />
              )}
            </div>
          ))}
        </div>

        <button
          type="submit"
          disabled={!modelTrained || !hasFields || loading}
          className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Predicting..." : submitLabel}
        </button>
      </form>

      {prediction !== null && (
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600">{resolvedResultLabel}</div>
            <div className="text-4xl font-bold text-primary-600 mt-1">
              {Math.round(prediction * 1000) / 1000}
            </div>
            {resultUnit ? (
              <div className="text-sm text-gray-500 mt-1">{resultUnit}</div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
