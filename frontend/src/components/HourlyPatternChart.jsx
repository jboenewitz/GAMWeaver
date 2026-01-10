import React from "react";
import Plot from "react-plotly.js";

const HourlyPatternChart = ({ patternData, loading }) => {
  if (loading) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Hourly Rental Pattern
        </h3>
        <div className="h-64 bg-gray-100 rounded-lg animate-pulse"></div>
      </div>
    );
  }

  if (!patternData) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          🕐 Hourly Rental Pattern
        </h3>
        <p className="text-gray-500 text-center py-8">
          Load the data to see the hourly rental pattern.
        </p>
      </div>
    );
  }

  const { hours, avg_rentals } = patternData;

  const data = [
    {
      x: hours,
      y: avg_rentals,
      type: "bar",
      marker: {
        color: hours.map((h) => {
          // Color code by time of day
          if (h >= 7 && h <= 9) return "#ef4444"; // Morning rush
          if (h >= 17 && h <= 19) return "#ef4444"; // Evening rush
          if (h >= 10 && h <= 16) return "#22c55e"; // Midday
          return "#3b82f6"; // Other times
        }),
      },
    },
  ];

  const layout = {
    title: {
      text: "Average Bike Rentals by Hour",
      font: { size: 16, color: "#374151" },
    },
    xaxis: {
      title: "Hour of Day",
      tickmode: "linear",
      dtick: 2,
      gridcolor: "#e5e7eb",
    },
    yaxis: {
      title: "Average Rentals",
      gridcolor: "#e5e7eb",
    },
    margin: { l: 60, r: 30, t: 50, b: 60 },
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    height: 300,
    bargap: 0.1,
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  return (
    <div className="card">
      <Plot data={data} layout={layout} config={config} className="w-full" />
      <div className="mt-2 flex justify-center gap-4 text-sm">
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 bg-red-500 rounded"></span>
          <span className="text-gray-600">Rush Hours</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 bg-green-500 rounded"></span>
          <span className="text-gray-600">Midday</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-3 bg-blue-500 rounded"></span>
          <span className="text-gray-600">Off-Peak</span>
        </div>
      </div>
    </div>
  );
};

export default HourlyPatternChart;
