import React from "react";
import Plot from "react-plotly.js";
import { createTranslator } from "../i18n";
import { getShapeFunctionDisplayName } from "../utils/featureDisplay";

const ShapeFunctionChart = ({ shapeFunction, t }) => {
  if (!shapeFunction) return null;

  const { feature_name, x_values, y_values, feature_type, x_tick_labels } =
    shapeFunction;
  const displayName = getShapeFunctionDisplayName(shapeFunction);

  const isNumeric = feature_type === "numeric";
  const xTickLabels =
    Array.isArray(x_tick_labels) && x_tick_labels.length === x_values.length
      ? x_tick_labels
      : null;

  const data = [
    {
      x: x_values,
      y: y_values,
      type: isNumeric ? "scatter" : "bar",
      mode: isNumeric ? "lines+markers" : undefined,
      customdata: !isNumeric ? xTickLabels || x_values : undefined,
      marker: {
        color: "#3b82f6",
        size: isNumeric ? 4 : undefined,
      },
      line: isNumeric
        ? {
            color: "#3b82f6",
            width: 2,
          }
        : undefined,
      fill: isNumeric ? "tozeroy" : undefined,
      fillcolor: "rgba(59, 130, 246, 0.1)",
      hovertemplate: !isNumeric
        ? `<b>%{customdata}</b><br>${t("shapeFunctions.effect")}: %{y:.3f}<extra></extra>`
        : undefined,
    },
  ];

  const layout = {
    title: {
      text: displayName,
      font: { size: 14, color: "#374151" },
    },
    xaxis: {
      title: displayName,
      gridcolor: "#e5e7eb",
      tickfont: { size: 10 },
      ...(!isNumeric && xTickLabels
        ? {
            tickmode: "array",
            tickvals: x_values,
            ticktext: xTickLabels,
            tickangle: -20,
          }
        : {}),
    },
    yaxis: {
      title: t("shapeFunctions.effectOnPrediction"),
      gridcolor: "#e5e7eb",
      zeroline: true,
      zerolinecolor: "#9ca3af",
      zerolinewidth: 1,
      tickfont: { size: 10 },
    },
    margin: { l: 50, r: 20, t: 40, b: 50 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    height: 250,
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <Plot data={data} layout={layout} config={config} className="w-full" />
    </div>
  );
};

const ShapeFunctionsGrid = ({
  shapeFunctions,
  loading,
  language = "en",
}) => {
  const t = createTranslator(language);

  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          {t("shapeFunctions.title")}
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div
              key={i}
              className="h-64 bg-gray-100 rounded-lg animate-pulse"
            ></div>
          ))}
        </div>
      </div>
    );
  }

  if (!shapeFunctions || shapeFunctions.length === 0) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          {t("shapeFunctions.title")}
        </h3>
        <p className="text-gray-500 text-center py-8">
          {t("shapeFunctions.emptyDescription")}
          <br />
          <span className="text-sm">{t("shapeFunctions.sharedDescription")}</span>
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-700">
          {t("shapeFunctions.title")}
        </h3>
        <p className="text-sm text-gray-500">{t("shapeFunctions.gridDescription")}</p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {shapeFunctions.map((sf, index) => (
          <ShapeFunctionChart key={index} shapeFunction={sf} t={t} />
        ))}
      </div>
    </div>
  );
};

export default ShapeFunctionsGrid;
