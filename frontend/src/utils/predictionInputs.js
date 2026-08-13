const roundToTwoDecimals = (value) => Math.round(value * 100) / 100;

const normalizeChoiceValue = (value, isNumericFeature = false) => {
  const rawValue = String(value ?? "").trim();
  if (!rawValue) return "";
  if (!isNumericFeature) return rawValue;

  const numericValue = Number(rawValue);
  if (!Number.isFinite(numericValue)) return rawValue;
  return Number.isInteger(numericValue)
    ? String(numericValue)
    : String(numericValue);
};

const formatChoiceLabel = (value, label) => {
  const normalizedLabel = String(label ?? "").trim();
  if (!normalizedLabel || normalizedLabel === value) {
    return value;
  }
  return `${value} = ${normalizedLabel}`;
};

export const getValidNumericValue = (feature) => {
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

export const getValidCategoricalValue = (feature) => {
  const options = feature.categorical_options || [];
  if (!options.length) {
    return "";
  }
  return options.includes(feature.default_value)
    ? feature.default_value
    : options[0];
};

export const getFeatureResponseOptions = (feature) => {
  const sharedResponseOptions = Array.isArray(
    feature?.feature_provenance?.response_options,
  )
    ? feature.feature_provenance.response_options
    : [];
  if (sharedResponseOptions.length > 0) {
    return sharedResponseOptions;
  }

  const sourceDetails = Array.isArray(
    feature?.feature_provenance?.source_details,
  )
    ? feature.feature_provenance.source_details
    : [];
  const firstSourceWithOptions = sourceDetails.find(
    (detail) =>
      Array.isArray(detail?.response_options) && detail.response_options.length > 0,
  );
  return firstSourceWithOptions?.response_options || [];
};

export const getFeatureChoiceOptions = (feature) => {
  if (feature?.feature_type === "categorical") {
    const normalizedResponseOptions = getFeatureResponseOptions(feature)
      .map((option) => ({
        value: normalizeChoiceValue(option?.value, false),
        label: String(option?.label ?? "").trim(),
      }))
      .filter((option) => option.value);
    const labelByValue = new Map(
      normalizedResponseOptions.map((option) => [option.value, option.label]),
    );
    const rawOptions = (feature.categorical_options || [])
      .map((option) => normalizeChoiceValue(option, false))
      .filter(Boolean);
    const optionValues =
      rawOptions.length > 0
        ? rawOptions
        : normalizedResponseOptions.map((option) => option.value);

    return Array.from(new Set(optionValues)).map((value) => ({
      value,
      label: formatChoiceLabel(value, labelByValue.get(value)),
    }));
  }

  return [];
};

export const featureUsesChoiceInput = (feature) =>
  feature?.feature_type === "categorical" &&
  getFeatureChoiceOptions(feature).length > 0;

export const coerceFeatureInputValue = (feature, value) => {
  if (value === "") return "";
  return feature?.feature_type === "numeric" ? Number(value) : value;
};

const getValidChoiceValue = (feature) => {
  const options = getFeatureChoiceOptions(feature);
  if (!options.length) {
    return feature?.feature_type === "numeric"
      ? getValidNumericValue(feature)
      : getValidCategoricalValue(feature);
  }

  const normalizedDefaultValue = normalizeChoiceValue(
    feature?.default_value,
    feature?.feature_type === "numeric",
  );
  const selectedValue = options.some(
    (option) => option.value === normalizedDefaultValue,
  )
    ? normalizedDefaultValue
    : options[0].value;

  return coerceFeatureInputValue(feature, selectedValue);
};

export const buildInitialFormData = (featureSchema = []) => {
  const initial = {};
  for (const feature of featureSchema) {
    initial[feature.name] = featureUsesChoiceInput(feature)
      ? getValidChoiceValue(feature)
      : feature.feature_type === "numeric"
        ? getValidNumericValue(feature)
        : getValidCategoricalValue(feature);
  }
  return initial;
};
