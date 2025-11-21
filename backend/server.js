// backend/server.js
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const path = require('path');
const fs = require('fs');
const morgan = require('morgan');

const app = express();
app.use(cors());
app.use(express.json());
// HTTP request logging
app.use(morgan('tiny'));

// Simple in-memory cache for geocode results
const cache = {
  geocode: new Map(), // key -> { ts, data }
};

const CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes

async function axiosGetWithRetry(url, opts = {}, retries = 3, backoff = 300) {
  let lastErr = null;
  for (let i = 0; i < retries; i++) {
    try {
      return await axios.get(url, opts);
    } catch (err) {
      lastErr = err;
      await new Promise((r) => setTimeout(r, backoff * Math.pow(2, i)));
    }
  }
  throw lastErr;
}

const buildPath = path.join(__dirname, '../frontend/build');

app.get("/api/weather", async (req, res) => {
  const city = req.query.city || "Mumbai";
  const apiKey = process.env.OPENWEATHER_API_KEY || "f10556614d53089381782ddea7b9bb20"; // set OPENWEATHER_API_KEY in production
  const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&units=metric&appid=${apiKey}`;

  try {
    const response = await axios.get(url);
    res.json(response.data);
  } catch (err) {
    console.error("Weather fetch error:", err.message || err);
    res.status(500).json({ error: "Failed to fetch weather data" });
  }
});

// Provide predefined areas for cities (served to frontend). If a city isn't in the static file,
// geocode the city and synthesize 3 suggested areas near the city center using current weather
app.get('/api/areas', async (req, res) => {
  let cityQuery = (req.query.city || '').toString();
  let city = cityQuery.toLowerCase().trim();
  console.log('/api/areas request city=', cityQuery, 'normalized=', city);
  // normalize some common misspellings or aliases
  const aliases = {
    'bhuswal': 'bhusawal',
    'bhusaval': 'bhusawal',
  };
  if (aliases[city]) city = aliases[city];

  try {
    const areasFile = path.join(__dirname, 'data', 'areas.json');
    if (fs.existsSync(areasFile)) {
      const raw = fs.readFileSync(areasFile, 'utf8');
      const all = JSON.parse(raw);
      if (all[city] && all[city].length) return res.json(all[city]);
    }

    // Not found in static list: try to geocode and synthesize areas
    if (!cityQuery) return res.json([]);

    // Use Nominatim to geocode (limit=1)
    const nomKey = `nom:${cityQuery}`;
    let geoData = null;
    // check cache
    if (cache.geocode.has(nomKey)) {
      const entry = cache.geocode.get(nomKey);
      if (Date.now() - entry.ts < CACHE_TTL_MS) {
        geoData = entry.data;
        console.log('/api/areas: geocode cache hit for', cityQuery);
      } else {
        cache.geocode.delete(nomKey);
      }
    }

    const nomUrl = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(cityQuery + ', India')}&format=json&limit=1`;
    const geo = geoData ? { data: geoData } : await axiosGetWithRetry(nomUrl, { headers: { 'User-Agent': 'smart-farm-dashboard' } }, 3, 250);
    console.log('/api/areas geocode response length=', Array.isArray(geo.data) ? geo.data.length : 'not-array');
    if (!geo.data || !geo.data.length) {
      console.warn('/api/areas: no geocode results for', cityQuery, 'nomUrl=', nomUrl);
      return res.json([]);
    }
    const loc = geo.data[0];
    // store in cache
    try { cache.geocode.set(nomKey, { ts: Date.now(), data: geo.data }); } catch (e) {}
    console.log('/api/areas geocode loc=', { lat: loc.lat, lon: loc.lon, display_name: loc.display_name });
    const lat = parseFloat(loc.lat);
    const lon = parseFloat(loc.lon);

    // Fetch current weather to derive suitability
    const apiKey = process.env.OPENWEATHER_API_KEY || 'f10556614d53089381782ddea7b9bb20';
    const wurl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${apiKey}`;
    let weather = null;
    try {
      const wres = await axiosGetWithRetry(wurl, {}, 3, 250);
      weather = wres.data;
      console.log('/api/areas weather fetched: temp=', weather && weather.main && weather.main.temp, 'hum=', weather && weather.main && weather.main.humidity);
    } catch (e) {
      console.warn('/api/areas weather fetch failed for', lat, lon, 'error=', e.message || e);
      weather = null;
    }
    const hum = (weather && weather.main && typeof weather.main.humidity === 'number') ? Number(weather.main.humidity) : 50;
    const temp = (weather && weather.main && typeof weather.main.temp === 'number') ? Number(weather.main.temp) : 25;

    // Create suggested areas preferably on the outskirts using boundingbox if provided
    // boundingbox: [south, north, west, east]
    const bb = (loc && loc.boundingbox && loc.boundingbox.length === 4) ? loc.boundingbox.map(Number) : null;
    const outskirts = [];
    if (bb) {
      const south = bb[0], north = bb[1], west = bb[2], east = bb[3];
      // pick points near the bounding box edges (NW, SE, SW)
      outskirts.push({ name: `${cityQuery} NW outskirts`, latitude: +(north - 0.01).toFixed(6), longitude: +(west + 0.01).toFixed(6), radius: 2000 });
      outskirts.push({ name: `${cityQuery} SE outskirts`, latitude: +(south + 0.01).toFixed(6), longitude: +(east - 0.01).toFixed(6), radius: 2000 });
      outskirts.push({ name: `${cityQuery} SW outskirts`, latitude: +(south + 0.02).toFixed(6), longitude: +(west + 0.02).toFixed(6), radius: 1800 });
    } else {
      // Fallback to larger offsets if bounding box isn't available
      outskirts.push({ name: `${cityQuery} North outskirts`, latitude: +(lat + 0.03).toFixed(6), longitude: +lon.toFixed(6), radius: 2000 });
      outskirts.push({ name: `${cityQuery} East outskirts`, latitude: +lat.toFixed(6), longitude: +(lon + 0.03).toFixed(6), radius: 2000 });
      outskirts.push({ name: `${cityQuery} South outskirts`, latitude: +(lat - 0.03).toFixed(6), longitude: +lon.toFixed(6), radius: 1800 });
    }

    // Simple suitability heuristic using temp/humidity
    outskirts.forEach(s => {
      let score = 50;
      if (temp >= 18 && temp <= 30) score += 20;
      if (hum >= 40 && hum <= 70) score += 20;
      // small randomization to vary results between outskirts
      score += Math.round((Math.random() - 0.5) * 8);
      const label = score >= 80 ? 'High' : score >= 60 ? 'Medium' : 'Low';
      s.suitability = label;
    });
    return res.json(outskirts);
  } catch (err) {
    console.error('Areas read/error', err);
    return res.status(500).json([]);
  }
});

// POST /api/predict-insights
app.post('/api/predict-insights', (req, res) => {
  try {
  const { weather, sensorData, area } = req.body || {};
  console.log('Predict request received. weather present:', !!weather, 'sensorData length:', (sensorData||[]).length, 'area:', area || 'N/A');

    // helper parse nutrients
    const parseNutrients = (nutrients) => {
      if (!nutrients) return { n: 0, p: 0, k: 0 };
      const parts = String(nutrients).split('-').map(p => Number(p) || 0);
      return { n: parts[0]||0, p: parts[1]||0, k: parts[2]||0 };
    };

    const computeAverages = (data) => {
      if (!data || data.length === 0) return { avgMoisture:0, avgTemp:0, avgN:0, avgP:0, avgK:0 };
      const count = data.length;
      let sumMoisture=0, sumTemp=0, sumN=0, sumP=0, sumK=0;
      data.forEach(d => {
        sumMoisture += Number(d.soilMoisture||0);
        sumTemp += Number(d.temperature||0);
        const npk = parseNutrients(d.nutrients||'0-0-0');
        sumN += npk.n; sumP += npk.p; sumK += npk.k;
      });
      return {
        avgMoisture: Math.round(sumMoisture/count),
        avgTemp: Math.round(sumTemp/count),
        avgN: +(sumN/count).toFixed(1),
        avgP: +(sumP/count).toFixed(1),
        avgK: +(sumK/count).toFixed(1),
      };
    };

  let avg = computeAverages(sensorData || []);
    const humidity = weather && weather.main ? Number(weather.main.humidity||0) : 0;

    // estimate yield (basic heuristic)
    let score = 50;
    if (avg.avgTemp >= 18 && avg.avgTemp <= 30) score += 20;
    else if (avg.avgTemp >= 15 && avg.avgTemp < 18) score += 10;
    if (avg.avgMoisture >= 40 && avg.avgMoisture <= 60) score += 20;
    else if (avg.avgMoisture >= 30 && avg.avgMoisture < 40) score += 10;
    const nutrientScore = Math.min(20, (avg.avgN + avg.avgP + avg.avgK) / 3);
    score += Math.round(nutrientScore);
    const percent = Math.min(100, Math.round(score));
    let yieldLabel = 'Moderate'; if (percent >= 80) yieldLabel='High'; else if (percent < 50) yieldLabel='Low';

    // planting suggestion
    let plantingText = 'Soil moisture not ideal for planting — prepare seedbed and monitor moisture.';
    let plantingOk = false;
    if (avg.avgTemp >= 15 && avg.avgTemp <= 30 && avg.avgMoisture >= 35 && avg.avgMoisture <= 65) {
      plantingOk = true; plantingText = 'Good to plant now for many warm-season crops.';
    } else if (avg.avgTemp < 15) plantingText = 'Too cold now for most warm-season crops. Consider cool-season crops or wait.';
    else if (avg.avgTemp > 30) plantingText = 'High temperatures — delay planting heat-sensitive crops or irrigate to reduce stress.';

    // disease risk
    let dscore = 0;
    if (humidity >= 75) dscore += 2;
    if (avg.avgMoisture >= 60) dscore += 2;
    if (avg.avgTemp >= 18 && avg.avgTemp <= 28) dscore += 1;
    let diseaseRisk = 'Low'; if (dscore>=4) diseaseRisk='High'; else if (dscore>=2) diseaseRisk='Moderate';
    let diseaseAdvice = diseaseRisk === 'High' ? 'High risk of fungal diseases — improve drainage, reduce irrigation frequency, consider fungicide after diagnosis.' : diseaseRisk === 'Moderate' ? 'Moderate risk — monitor closely.' : 'Low disease risk currently.';

    // fertilizer recs
    const fert = [];
    if (avg.avgN < 12) fert.push('Apply nitrogen-rich fertilizer (e.g., urea).');
    if (avg.avgP < 8) fert.push('Add phosphorus (e.g., superphosphate).');
    if (avg.avgK < 8) fert.push('Add potassium (e.g., muriate of potash).');
    if (fert.length===0) fert.push('Nutrients appear adequate.');

    // irrigation plan: target 50% moisture
    const irrigation = (sensorData||[]).filter(z=>Number(z.soilMoisture)<40).map(z=>{
      const deficit = 50 - Number(z.soilMoisture);
      const water_mm = Math.max(5, Math.round(deficit*1));
      return { zone: z.zone, current: z.soilMoisture, water_mm };
    });

    const yieldObj = {
      percent,
      label: yieldLabel,
      note: yieldLabel === 'High' ? 'Conditions look favorable for a good yield.' : yieldLabel === 'Moderate' ? 'Yield may be average; consider small optimizations.' : 'Yield likely low; consider irrigation/fertilizer and check crop suitability.'
    };

    // Apply small area-based modifiers (so different areas produce slightly different predictions)
    const areaKey = (area || '').toString().toLowerCase();
    const areaModifiers = {
      hingna: { moisture: 2, n: 0.5 },
      kalmeshwar: { moisture: 3, n: 0.2 },
      kamptee: { moisture: -2, n: -0.5 },
      bhusawal: { moisture: 2, n: 0.4 },
      muktainagar: { moisture: 0, n: 0.0 },
      'sangli road': { moisture: 1, n: 0.1 },
      ichalkaranji: { moisture: -1, n: 0.0 },
      hatkanangale: { moisture: 0, n: 0.0 },
    };
    const mod = areaModifiers[areaKey] || null;
    if (mod) {
      avg = { ...avg };
      if (typeof mod.moisture === 'number') avg.avgMoisture = Math.round(avg.avgMoisture + mod.moisture);
      if (typeof mod.n === 'number') avg.avgN = +(avg.avgN + mod.n).toFixed(1);
    }

    // Apply area percent delta to make predictions visibly different per area
    const areaPercentDeltas = {
      hingna: 5,
      kalmeshwar: 7,
      kamptee: -6,
      bhusawal: 6,
      muktainagar: 1,
      'sangli road': 4,
      ichalkaranji: -2,
      hatkanangale: 0,
    };
    const percentDelta = areaPercentDeltas[areaKey] || 0;
    // city/area deterministic perturbation: derive from area or weather.name to vary by city
    const citySeedStr = (area || (weather && weather.name) || 'default').toString().toLowerCase();
    const hashString = (s) => {
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
      return h;
    };
    const seed = hashString(citySeedStr) % 1000;
    // noise between -8 and +8
    const cityNoise = ((seed % 17) - 8);
    const adjustedPercent = Math.max(0, Math.min(100, percent + percentDelta + cityNoise));

    // update yieldObj to reflect adjustedPercent
    const yieldObjFinal = { ...yieldObj, percent: adjustedPercent };

    const response = {
      yield: yieldObj,
      planting: { ok: plantingOk, text: plantingText },
      disease: { risk: diseaseRisk, advice: diseaseAdvice },
      fertilizer: fert,
      irrigation,
      averages: avg,
    };

    // Dynamic crop recommendation based on climate and soil averages
    const cropSuitability = (tempC, humPerc, soilMoist) => {
      // Define crops with ideal ranges (temp in C, humidity %, soil moisture %)
      const crops = {
        Rice: { t:[20,32], h:[70,100], m:[50,90] },
        Maize: { t:[18,30], h:[40,80], m:[35,65] },
        Wheat: { t:[10,24], h:[30,70], m:[30,60] },
        Soybean: { t:[20,30], h:[50,85], m:[35,65] },
        Cotton: { t:[20,35], h:[30,70], m:[25,60] },
        Sugarcane: { t:[20,35], h:[50,90], m:[50,90] },
        Sorghum: { t:[22,35], h:[30,70], m:[25,55] },
        Millets: { t:[25,38], h:[20,60], m:[20,50] }
      };
      const scores = [];
      Object.entries(crops).forEach(([name, ranges]) => {
        let score = 100;
        // temperature penalty
        if (tempC < ranges.t[0]) score -= (ranges.t[0] - tempC) * 3;
        if (tempC > ranges.t[1]) score -= (tempC - ranges.t[1]) * 3;
        // humidity penalty
        if (humPerc < ranges.h[0]) score -= (ranges.h[0] - humPerc) * 0.8;
        if (humPerc > ranges.h[1]) score -= (humPerc - ranges.h[1]) * 0.8;
        // soil moisture penalty
        if (soilMoist < ranges.m[0]) score -= (ranges.m[0] - soilMoist) * 0.7;
        if (soilMoist > ranges.m[1]) score -= (soilMoist - ranges.m[1]) * 0.7;
        scores.push({ name, score: Math.max(0, Math.round(score)) });
      });
      scores.sort((a,b)=>b.score-a.score);
      return scores;
    };

    // use available values: temp (from weather variable), hum (from weather variable), and avg.avgMoisture
    const tempC = Number((weather && weather.main && weather.main.temp) || avg.avgTemp || 25);
    const humPerc = Number((weather && weather.main && weather.main.humidity) || 50);
    const soilMoist = Number(avg.avgMoisture || 40);

    const scores = cropSuitability(tempC, humPerc, soilMoist);
    const top = scores[0];
    let crop = (top && top.name) || 'Maize';
    const alternatives = scores.slice(1,4).map(s=>({ crop: s.name, score: s.score }));

    let fertAdvice = 'NPK 10:26:26, 40kg/acre';
    if (avg.avgN >= 12 && avg.avgP >= 8 && avg.avgK >= 8) fertAdvice = 'Balanced fertilization; follow local recommendations.';
    // crop-specific sowing/planting windows (simple demo values)
    const CROP_SOWING_WINDOWS = {
      Rice: 'June 10–June 25',
      Maize: 'June 1–June 20',
      Wheat: 'October 15–November 5',
      Soybean: 'June 5–June 25',
      Sugarcane: 'February 1–March 15',
      Cotton: 'June 10–July 5',
      Sorghum: 'May 20–June 15',
      Millets: 'May 1–June 15',
    };
    const sowingWindow = CROP_SOWING_WINDOWS[crop] || 'Check local agronomic calendar';

    // Base expected yields (tons/acre) and base market price per crop (INR/quintal) — approximate demo values
    const baseYields = { Soybean: 2.2, Cotton: 1.2, Paddy: 2.8, Maize: 3.0, Wheat: 2.5, Millets: 1.5, Sugarcane: 60, Sorghum: 1.6 };
    const basePrices = { Soybean: 3800, Cotton: 40000, Paddy: 1600, Maize: 2300, Wheat: 2000, Millets: 1500, Sugarcane: 500, Sorghum: 1400 };
    const priceSensitivity = { Soybean: 10, Cotton: 50, Paddy: 6, Maize: 5, Wheat: 4, Millets: 3, Sugarcane: 1, Sorghum: 3 };

  // adjust expected yield by both percent and crop climate suitability
  const base = baseYields[crop] || 2.0;
  const climateMultiplier = Math.max(0.5, (top.score || 60) / 80); // higher score -> boost
  // incorporate cityNoise into yield slightly
  const yieldNoiseFactor = 1 + cityNoise / 200.0; // +/- ~4%
  const expectedYield = +(base * (adjustedPercent / 75) * climateMultiplier * yieldNoiseFactor).toFixed(2);
    const priceBase = basePrices[crop] || 2300;
    const sens = priceSensitivity[crop] || 5;
    const marketPrice = Math.max(0, priceBase + Math.round((adjustedPercent - 75) * sens));
    let nextIrrigationDays = 7;
    if (avg.avgMoisture < 30) nextIrrigationDays = 1;
    else if (avg.avgMoisture < 40) nextIrrigationDays = 3;
    else if (avg.avgMoisture < 50) nextIrrigationDays = 5;

    // add reasoning for recommendation
    const reasoning = `Top crop recommended based on climate/soil: ${crop} (score ${top.score}). Alternatives: ${alternatives.map(a=>a.crop).join(', ')}.`;

  response.cropRecommendation = { crop, score: top.score, alternatives, reasoning };
    response.fertilizerAdvice = fertAdvice;
    response.sowingWindow = sowingWindow;
    response.expectedYield = expectedYield; // tons/acre
    response.marketPriceForecast = marketPrice; // INR/quintal
    response.nextIrrigationDays = nextIrrigationDays;
  response.area = area || null;
  response.yield = yieldObjFinal; // replace yield with adjusted percent

        // sowing window already set above where crop is known
    console.log(`Area '${area || 'N/A'}' -> cropRecommendation='${crop}', marketPriceForecast=${marketPrice}`);

    return res.json(response);
  } catch (err) {
    console.error('Predict error', err);
    res.status(500).json({ error: 'Predict failed' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));

// List available model artifacts and metadata
app.get('/api/models', (req, res) => {
  try {
    const mlDir = path.join(__dirname, 'ml');
    if (!fs.existsSync(mlDir)) return res.json({ models: [] });
    const files = fs.readdirSync(mlDir).filter(f => f.endsWith('.joblib') || f.endsWith('.pkl'));
    const metaPath = path.join(mlDir, 'model_metadata.json');
    const metadata = fs.existsSync(metaPath) ? JSON.parse(fs.readFileSync(metaPath, 'utf8')) : null;
    return res.json({ models: files, metadata });
  } catch (e) {
    console.error('models list error', e);
    return res.status(500).json({ error: 'could not list models' });
  }
});

// ML predict endpoint (spawns python predict.py)
app.post('/api/predict-yield', (req, res) => {
  const { rainfall, temperature, humidity } = req.body || {};
  if (rainfall == null || temperature == null || humidity == null) return res.status(400).json({ error: 'missing fields' });
  const spawn = require('child_process').spawn;
  const py = spawn('python', [path.join(__dirname, 'ml', 'predict.py'), rainfall, temperature, humidity]);
  let out = '';
  py.stdout.on('data', (d) => { out += d.toString(); });
  py.stderr.on('data', (d) => console.error('py err', d.toString()));
  py.on('close', (code) => {
    try { const json = JSON.parse(out); return res.json(json); } catch (e) { return res.status(500).json({ error: 'predict failed', raw: out }); }
  });
});
