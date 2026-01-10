import React from "react";

const MetricsCard = ({ metrics, loading }) => {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/2 mb-4"></div>
        <div className="grid grid-cols-2 gap-4">
          <div className="h-20 bg-gray-200 rounded"></div>
          <div className="h-20 bg-gray-200 rounded"></div>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-700 mb-4">
          Model Metrics
        </h3>
        <p className="text-gray-500 text-center py-4">
          Train the model to see performance metrics
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-lg font-semibold text-gray-700 mb-4">
        Model Metrics
      </h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="metric-card">
          <div className="text-sm opacity-80">RMSE</div>
          <div className="text-2xl font-bold">{metrics.rmse?.toFixed(2)}</div>
          <div className="text-xs opacity-70 mt-1">Root Mean Squared Error</div>
        </div>
        <div className="metric-card bg-gradient-to-br from-green-500 to-green-600">
          <div className="text-sm opacity-80">MAE</div>
          <div className="text-2xl font-bold">{metrics.mae?.toFixed(2)}</div>
          <div className="text-xs opacity-70 mt-1">Mean Absolute Error</div>
        </div>
      </div>
      <div className="mt-4 text-center text-sm text-gray-500">
        Model: {metrics.model_type}
      </div>
    </div>
  );
};

export default MetricsCard;
