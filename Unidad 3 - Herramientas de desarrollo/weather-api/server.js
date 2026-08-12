const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

const GEOCODING_URL = 'https://geocoding-api.open-meteo.com/v1/search';
const FORECAST_URL = 'https://api.open-meteo.com/v1/forecast';

// Diccionario básico para traducir los códigos de clima (weathercode) de Open-Meteo
const WEATHER_CODES = {
  0: 'Cielo despejado',
  1: 'Mayormente despejado',
  2: 'Parcialmente nublado',
  3: 'Nublado',
  45: 'Niebla',
  48: 'Niebla con escarcha',
  51: 'Llovizna ligera',
  53: 'Llovizna moderada',
  55: 'Llovizna intensa',
  61: 'Lluvia ligera',
  63: 'Lluvia moderada',
  65: 'Lluvia intensa',
  71: 'Nevada ligera',
  73: 'Nevada moderada',
  75: 'Nevada intensa',
  80: 'Chubascos ligeros',
  81: 'Chubascos moderados',
  82: 'Chubascos violentos',
  95: 'Tormenta',
  96: 'Tormenta con granizo ligero',
  99: 'Tormenta con granizo intenso',
};

/**
 * Busca las coordenadas (lat/lon) de una ciudad usando la API de geocoding de Open-Meteo.
 */
async function getCoordinates(city) {
  const { data } = await axios.get(GEOCODING_URL, {
    params: { name: city, count: 1, language: 'es', format: 'json' },
  });

  if (!data.results || data.results.length === 0) {
    return null;
  }

  const { latitude, longitude, name, country, admin1 } = data.results[0];
  return { latitude, longitude, name, country, admin1 };
}

/**
 * Consulta el clima actual para unas coordenadas dadas.
 */
async function getWeather(latitude, longitude) {
  const { data } = await axios.get(FORECAST_URL, {
    params: {
      latitude,
      longitude,
      current_weather: true,
      timezone: 'auto',
    },
  });

  return data.current_weather;
}

app.get('/', (req, res) => {
  res.json({
    message: 'Weather API - consulta el clima por ciudad usando Open-Meteo',
    uso: '/weather?city=NombreDeLaCiudad',
    ejemplo: '/weather?city=Madrid',
  });
});

app.get('/weather', async (req, res) => {
  const { city } = req.query;

  if (!city || !city.trim()) {
    return res.status(400).json({ error: 'Debes indicar una ciudad. Ej: /weather?city=Madrid' });
  }

  try {
    const location = await getCoordinates(city.trim());

    if (!location) {
      return res.status(404).json({ error: `No se encontró la ciudad "${city}"` });
    }

    const currentWeather = await getWeather(location.latitude, location.longitude);

    if (!currentWeather) {
      return res.status(502).json({ error: 'No se pudo obtener el clima en este momento' });
    }

    res.json({
      ciudad: location.name,
      region: location.admin1 || null,
      pais: location.country,
      coordenadas: { latitude: location.latitude, longitude: location.longitude },
      clima: {
        temperatura: `${currentWeather.temperature} °C`,
        velocidad_viento: `${currentWeather.windspeed} km/h`,
        direccion_viento: `${currentWeather.winddirection}°`,
        descripcion: WEATHER_CODES[currentWeather.weathercode] || 'Desconocido',
        codigo: currentWeather.weathercode,
        hora: currentWeather.time,
      },
    });
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: 'Error al consultar el clima', detalle: error.message });
  }
});

app.use((req, res) => {
  res.status(404).json({ error: 'Ruta no encontrada' });
});

app.listen(PORT, () => {
  console.log(`Weather API escuchando en http://localhost:${PORT}`);
});
