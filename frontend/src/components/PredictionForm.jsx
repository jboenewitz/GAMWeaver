import React, { useState } from "react";

const PredictionForm = ({ onPredict, loading, modelTrained }) => {
  const [formData, setFormData] = useState({
    temperature: 20,
    humidity: 50,
    windspeed: 10,
    time_of_day: 12,
    type_of_day: "Working Day",
    weathersituation: "Clear",
  });

  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "number" ? parseFloat(value) : value,
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
      <h3 className="text-lg font-semibold text-gray-700 mb-4">
        Make Prediction
      </h3>

      {!modelTrained && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-yellow-700 text-sm">
          Please train the model first before making predictions
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Temperature (°C)</label>
            <input
              type="number"
              name="temperature"
              value={formData.temperature}
              onChange={handleChange}
              className="input-field"
              min="-10"
              max="40"
              step="0.5"
            />
          </div>

          <div>
            <label className="label">Humidity (%)</label>
            <input
              type="number"
              name="humidity"
              value={formData.humidity}
              onChange={handleChange}
              className="input-field"
              min="0"
              max="100"
            />
          </div>

          <div>
            <label className="label">Windspeed (km/h)</label>
            <input
              type="number"
              name="windspeed"
              value={formData.windspeed}
              onChange={handleChange}
              className="input-field"
              min="0"
              max="70"
            />
          </div>

          <div>
            <label className="label">Time of Day (Hour)</label>
            <input
              type="number"
              name="time_of_day"
              value={formData.time_of_day}
              onChange={handleChange}
              className="input-field"
              min="0"
              max="23"
            />
          </div>

          <div>
            <label className="label">Type of Day</label>
            <select
              name="type_of_day"
              value={formData.type_of_day}
              onChange={handleChange}
              className="select-field"
            >
              <option value="Working Day">Working Day</option>
              <option value="Weekend">Weekend</option>
              <option value="Holiday">Holiday</option>
            </select>
          </div>

          <div>
            <label className="label">Weather</label>
            <select
              name="weathersituation"
              value={formData.weathersituation}
              onChange={handleChange}
              className="select-field"
            >
              <option value="Clear">Clear</option>
              <option value="Cloudy">Cloudy</option>
              <option value="Light Rain">Light Rain</option>
              <option value="Heavy Rain">Heavy Rain</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={!modelTrained || loading}
          className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Predicting..." : "Predict Bike Rentals"}
        </button>
      </form>

      {prediction !== null && (
        <div className="mt-6 p-4 bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-600">Predicted Bike Rentals</div>
            <div className="text-4xl font-bold text-primary-600 mt-1">
              {Math.round(prediction)}
            </div>
            <div className="text-sm text-gray-500 mt-1">bikes per hour</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
