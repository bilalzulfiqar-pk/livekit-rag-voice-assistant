from __future__ import annotations

import logging
from time import perf_counter

import httpx
from livekit.agents.llm import Toolset, function_tool

from voice_agent.telemetry import VoiceAgentTelemetry

logger = logging.getLogger("livekit-rag-voice-agent.weather-tool")

WEATHER_FAILURE_MESSAGE = "I couldn't fetch the weather right now."
UNKNOWN_CITY_MESSAGE = "I couldn't find that city."
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_LABELS = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rainy",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snowy",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorms",
    96: "stormy with hail",
    99: "stormy with hail",
}


class WeatherToolset(Toolset):
    def __init__(
        self,
        *,
        telemetry: VoiceAgentTelemetry | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._telemetry = telemetry
        self._timeout = httpx.Timeout(
            timeout_seconds,
            connect=min(2.0, timeout_seconds),
            read=timeout_seconds,
            write=timeout_seconds,
            pool=timeout_seconds,
        )
        self._client: httpx.AsyncClient | None = None
        super().__init__(id="weather")

    async def setup(self) -> WeatherToolset:
        await self._ensure_client()
        await super().setup()
        return self

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await super().aclose()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    @function_tool(
        description=(
            "Use this tool only to get the current weather for a named city."
        )
    )
    async def get_current_weather(self, city: str) -> str:
        """Look up the current weather for a city.

        Args:
            city: The city name to look up, such as Lahore or Karachi.
        """

        cleaned_city = city.strip()
        if not cleaned_city:
            return UNKNOWN_CITY_MESSAGE

        if self._telemetry is not None:
            self._telemetry.publish_weather_querying()

        client = await self._ensure_client()
        start_time = perf_counter()
        try:
            geocode_response = await client.get(
                GEOCODING_URL,
                params={
                    "name": cleaned_city,
                    "count": 1,
                    "language": "en",
                    "format": "json",
                },
            )
            geocode_response.raise_for_status()
            geocode_payload = geocode_response.json()
            results = geocode_payload.get("results") or []
            if not results:
                if self._telemetry is not None:
                    self._telemetry.publish_weather_result(
                        success=False,
                        latency_ms=round((perf_counter() - start_time) * 1000),
                        fallback=True,
                    )
                return UNKNOWN_CITY_MESSAGE

            place = results[0]
            forecast_response = await client.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code",
                    "timezone": "auto",
                },
            )
            forecast_response.raise_for_status()
            current = (forecast_response.json().get("current") or {})
        except Exception as exc:
            logger.warning("Weather lookup failed", exc_info=exc)
            if self._telemetry is not None:
                self._telemetry.publish_weather_result(
                    success=False,
                    latency_ms=round((perf_counter() - start_time) * 1000),
                    fallback=True,
                )
            return WEATHER_FAILURE_MESSAGE

        temperature = current.get("temperature_2m")
        if temperature is None:
            if self._telemetry is not None:
                self._telemetry.publish_weather_result(
                    success=False,
                    latency_ms=round((perf_counter() - start_time) * 1000),
                    fallback=True,
                )
            return WEATHER_FAILURE_MESSAGE

        if self._telemetry is not None:
            self._telemetry.publish_weather_result(
                success=True,
                latency_ms=round((perf_counter() - start_time) * 1000),
                fallback=False,
            )

        label = WEATHER_CODE_LABELS.get(current.get("weather_code"), "steady")
        location_name = _format_place_name(place)
        rounded_temperature = round(float(temperature))
        return (
            f"{location_name} is {rounded_temperature} degrees Celsius "
            f"and {label} right now."
        )


def _format_place_name(place: dict[str, object]) -> str:
    primary = str(place.get("name", "")).strip()
    region = str(place.get("admin1", "")).strip()
    country = str(place.get("country", "")).strip()

    labels: list[str] = []
    for value in (primary, region, country):
        if value and value not in labels:
            labels.append(value)

    return ", ".join(labels) if labels else "That location"
