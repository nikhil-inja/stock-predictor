// frontend/src/StockChart.js
import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const StockChart = ({ historicalData, prediction }) => {
  // Defensive: ensure historicalData has expected shape
  if (!historicalData || !Array.isArray(historicalData.values) || !Array.isArray(historicalData.labels)) {
    console.warn('StockChart: historicalData malformed', historicalData);
    return <div>No chart data available</div>;
  }

  const labels = [...historicalData.labels, 'Prediction'];
  const histValues = historicalData.values || [];

  // pad historical values with null to align with labels
  const closeData = [...histValues, null];

  // Build prediction series: nulls up to last historical value, then prediction
  const predictionSeries = Array(Math.max(0, histValues.length - 1)).fill(null);
  if (histValues.length > 0) predictionSeries.push(histValues.slice(-1)[0]);
  predictionSeries.push(prediction);

  const data = {
    labels,
    datasets: [
      {
        label: 'Close Price',
        data: closeData,
        borderColor: 'rgb(53, 162, 235)',
        tension: 0.1,
      },
      {
        label: 'Prediction',
        data: predictionSeries,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        pointRadius: 5,
        pointBackgroundColor: 'rgb(255, 99, 132)',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: '90-Day Price History & Next-Day Prediction' },
    },
  };

  return <Line options={options} data={data} />;
};

export default StockChart;