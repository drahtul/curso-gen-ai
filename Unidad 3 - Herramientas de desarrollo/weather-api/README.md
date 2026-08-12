# Weather API

API en Node.js + Express que consulta el clima actual de una ciudad usando
la API pública de [Open-Meteo](https://open-meteo.com/):

1. Geocoding API (`https://geocoding-api.open-meteo.com/v1/search`) para convertir el nombre de la ciudad en coordenadas (lat/lon).
2. Forecast API (`https://api.open-meteo.com/v1/forecast`) para obtener el clima actual de esas coordenadas.

## Instalación

```bash
cd weather-api
npm install
```

## Ejecución

```bash
npm start
```

El servidor levanta en `http://localhost:3000` (puede cambiarse con la variable de entorno `PORT`).

Modo desarrollo (recarga automática con `node --watch`):

```bash
npm run dev
```

## Endpoint

### `GET /weather?city={nombre}`

**Ejemplo:**

```
GET http://localhost:3000/weather?city=Madrid
```

**Respuesta:**

```json
{
  "ciudad": "Madrid",
  "region": "Comunidad Autónoma de Madrid",
  "pais": "España",
  "coordenadas": { "latitude": 40.4165, "longitude": -3.70256 },
  "clima": {
    "temperatura": "26.9 °C",
    "velocidad_viento": "4.3 km/h",
    "direccion_viento": "24°",
    "descripcion": "Cielo despejado",
    "codigo": 0,
    "hora": "2026-08-12T01:15"
  }
}
```

**Errores:**

- `400` si no se envía el parámetro `city`.
- `404` si no se encuentra la ciudad indicada.
- `500` si ocurre un error al consultar Open-Meteo.
