import React, { useEffect, useMemo, useState } from "react";

const buildInitialFormData = (featureSchema = []) => {
  const initial = {};
  for (const feature of featureSchema) {
    if (feature.feature_type === "numeric") {
      initial[feature.name] = Number.isFinite(feature.default_value)
        ? feature.default_value
        : 0;
    } else {
      const options = feature.categorical_options || [];
      initial[feature.name] = feature.default_value || options[0] || "";
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
}) => {
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const hasFeatures = featureSchema && featureSchema.length > 0;

  const title = useMemo(
    () => `Predict ${targetColumn || "Target"}`,
    [targetColumn],
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
      <h3 className="text-lg font-semibold text-gray-700 mb-2">Make Prediction</h3>
      <p className="text-sm text-gray-500 mb-4">
        Set dataset feature values and run a prediction. Inputs are generated
        from the active dataset schema.
      </p>

      {!modelTrained && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
          Please train the model first before making predictions.
        </div>
      )}

      {!hasFeatures && (
        <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded-lg text-gray-600 text-sm">
          No feature schema available. Ask the superadmin to load a dataset.
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
                  step="0.01"
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
          {loading ? "Predicting..." : title}
        </button>
      </form>

      {prediction !== null && (
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600">Predicted {targetColumn || "Target"}</div>
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
