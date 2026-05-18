export const AQI_COLOR = (aqi) => {
  if (aqi <= 50)  return "#3B6D11";
  if (aqi <= 100) return "#0F6E56";
  if (aqi <= 200) return "#EF9F27";
  if (aqi <= 300) return "#D85A30";
  if (aqi <= 400) return "#993C1D";
  return "#534AB7";
};

export const AQI_CAT = (aqi) => {
  if (aqi <= 50)  return { cat: "Good",         style: "background:#EAF3DE;color:#3B6D11" };
  if (aqi <= 100) return { cat: "Satisfactory", style: "background:#E1F5EE;color:#0F6E56" };
  if (aqi <= 200) return { cat: "Moderate",     style: "background:#FAEEDA;color:#854F0B" };
  if (aqi <= 300) return { cat: "Poor",         style: "background:#FAECE7;color:#993C1D" };
  if (aqi <= 400) return { cat: "Very Poor",    style: "background:#F7C1C1;color:#791F1F" };
  return { cat: "Severe", style: "background:#CECBF6;color:#3C3489" };
};

export const pointsToSvg = (arr, height = 110, width = 620) => {
  if (!arr || arr.length === 0) return "";
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = (max - min) || 1;
  return arr.map((val, i) => {
    const x = (i / (arr.length - 1)) * width;
    const y = height - 15 - ((val - min) / range) * (height - 30);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
};

// Helper to extract [MAE, RMSE, R2, MAPE] whether input is an Object or an Array
export const getMetricVals = (val, fallbackArr) => {
  if (!val) return fallbackArr;
  if (Array.isArray(val)) return val;
  if (typeof val === 'object') {
    return [
      val.MAE ?? fallbackArr[0],
      val.RMSE ?? fallbackArr[1],
      val.R2 ?? fallbackArr[2],
      val.MAPE ?? fallbackArr[3]
    ];
  }
  return fallbackArr;
};
