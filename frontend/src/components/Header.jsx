import React from "react";

const Header = ({ modelStatus }) => {
  return (
    <header className="glass-header text-white">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Bike Rental Prediction</h1>
            <p className="text-blue-100/90 mt-1">
              IGANN - Interpretable Generalized Additive Neural Networks
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-sm text-blue-100/70">Model Status</div>
              <div
                className={`flex items-center gap-2 ${
                  modelStatus?.is_trained ? "text-green-300" : "text-yellow-300"
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    modelStatus?.is_trained ? "bg-green-400" : "bg-yellow-400"
                  }`}
                ></span>
                {modelStatus?.is_trained ? "Trained" : "Not Trained"}
              </div>
            </div>
            {modelStatus?.data_loaded && (
              <div className="text-right border-l border-white/30 pl-4">
                <div className="text-sm text-blue-100/70">Data Loaded</div>
                <div className="text-white">
                  {modelStatus?.train_size + modelStatus?.test_size} records
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
