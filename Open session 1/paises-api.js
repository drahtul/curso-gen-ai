import express from "express";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

const app = express();
const PORT = process.env.PORT || 3000;

// Deja registro de cada request que recibe la API
app.use((req, res, next) => {
  const inicio = Date.now();
  res.on("finish", () => {
    const hora = new Date().toLocaleTimeString();
    console.log(
      `[${hora}] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${Date.now() - inicio} ms)`,
    );
  });
  next();
});

const API_KEY = process.env.RESTCOUNTRIES_API_KEY;
const API_BASE = "https://api.restcountries.com/countries/v5";

let countriesCache = null;
let cacheTime = 0;
const CACHE_DURATION = 3600000; // 1 hora en ms

async function fetchCountries() {
  const now = Date.now();
  if (countriesCache && now - cacheTime < CACHE_DURATION) {
    console.log("   [cache] Lista de países servida desde caché");
    return countriesCache;
  }

  const headers = {
    Authorization: `Bearer ${API_KEY}`,
  };

  // limit alto (~todos los países) para que el catálogo del chatbot sea
  // grande y estable, condición necesaria para que el prompt caching
  // automático de OpenAI entre en juego (ver chatbot.py)
  console.log(`   [restcountries] GET ${API_BASE}?limit=100`);
  const response = await fetch(`${API_BASE}?limit=100`, { headers });
  if (!response.ok)
    throw new Error(`Error fetching countries: ${response.status}`);

  const data = await response.json();

  let countriesList = data.data.objects;
  countriesCache = countriesList;
  cacheTime = now;
  return countriesCache;
}

// Extrae solo los campos útiles de un país, descartando traducciones,
// nombres nativos y metadata que no aportan al chatbot.
function simplifyCountry(country) {
  const capital = country.capitals?.[0];

  return {
    name: country.names.common,
    officialName: country.names.official,
    capital: capital?.name || "Unknown",
    region: country.region || "Unknown",
    subregion: country.subregion || "Unknown",
    continents: country.continents || [],
    population: country.population,
    areaKm2: country.area?.kilometers,
    languages: (country.languages || []).map((lang) => lang.name),
    currencies: (country.currencies || []).map((currency) => ({
      code: currency.code,
      name: currency.name,
      symbol: currency.symbol,
    })),
    demonym: country.demonyms?.eng?.m || "Unknown",
    governmentType: country.government_type || "Unknown",
    timezones: country.timezones || [],
    callingCodes: country.calling_codes || [],
    borders: country.borders || [],
    landlocked: country.landlocked,
    drivingSide: country.cars?.driving_side || "Unknown",
    unMember: country.classification?.un_member,
    // Solo las membresías a las que efectivamente pertenece
    memberships: Object.entries(country.memberships || {})
      .filter(([, isMember]) => isMember)
      .map(([name]) => name),
    coordinates: country.coordinates,
    links: {
      wikipedia: country.links?.wikipedia || "",
      maps: country.links?.google_maps || "",
    },
  };
}

// GET /countries - Lista de países (nombre y región)
app.get("/countries", async (req, res) => {
  try {
    const countries = await fetchCountries();

    if (!Array.isArray(countries)) {
      throw new Error("Invalid response format from API");
    }

    const simplified = countries.map((country) => ({
      name: country.names.common,
      capital: country.capitals?.[0]?.name || "Unknown",
      region: country.region || "Unknown",
      population: country.population,
    }));
    res.json(simplified);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /countries/:name - Información detallada de un país
app.get("/countries/:name", async (req, res) => {
  try {
    const { name } = req.params;

    const headers = {
      Authorization: `Bearer ${API_KEY}`,
    };

    const urlExterna = `${API_BASE}/names.common/${encodeURIComponent(name)}`;
    console.log(`   [restcountries] GET ${urlExterna}`);

    const response = await fetch(urlExterna, { headers });
    if (!response.ok)
      throw new Error(`Error fetching country: ${response.status}`);

    const data = await response.json();
    const country = data.data?.objects?.[0];

    if (!country) {
      return res.status(404).json({ error: `Country "${name}" not found` });
    }

    res.json(simplifyCountry(country));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`API running at http://localhost:${PORT}`);
  console.log(`GET /countries - Lista de países`);
  console.log(`GET /countries/:name - Información de un país`);
});
