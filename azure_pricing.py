"""
Azure Retail Prices lookup for the GPU VM SKUs in `GPU_DATABASE`.

Uses the public, unauthenticated Azure Retail Prices API
(https://prices.azure.com/api/retail/prices) and caches results to a local
JSON file for 24h so we don't hit the API on every request.

Public surface
--------------
- AZURE_REGION             : str  (default "eastus", overridable via env)
- vm_name_to_arm_sku(name) : "NC40ads H100 v5" -> "Standard_NC40ads_H100_v5"
- get_pricing(sku_names)   : returns dict {sku_name: hourly_usd | None}
                             (None = SKU is not sold on-demand in this region)
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests

log = logging.getLogger(__name__)

AZURE_REGION = os.getenv("AZURE_PRICING_REGION", "eastus")
_CACHE_TTL_SECONDS = int(os.getenv("AZURE_PRICING_CACHE_TTL", str(24 * 3600)))
_CACHE_PATH = Path(
    os.getenv(
        "AZURE_PRICING_CACHE_PATH",
        str(Path(__file__).parent / ".cache" / "azure_pricing.json"),
    )
)
_API_URL = "https://prices.azure.com/api/retail/prices"
_HTTP_TIMEOUT = 10


def vm_name_to_arm_sku(name: str) -> str:
    """`"NC40ads H100 v5"` -> `"Standard_NC40ads_H100_v5"`."""
    return "Standard_" + name.strip().replace(" ", "_")


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.exists():
            with _CACHE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to read pricing cache: %s", e)
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to write pricing cache: %s", e)


def _fetch_one(arm_sku: str, region: str) -> float | None:
    """Return hourly USD for the cheapest Linux on-demand offer of `arm_sku`
    in `region`, or None if no on-demand listing exists.

    Filters:
      - serviceName    = 'Virtual Machines'
      - priceType      = 'Consumption'   (excludes Reservations)
      - armRegionName  = region
      - armSkuName     = arm_sku
    Then client-side excludes Spot / Low Priority and Windows.
    """
    flt = (
        f"serviceName eq 'Virtual Machines' "
        f"and priceType eq 'Consumption' "
        f"and armRegionName eq '{region}' "
        f"and armSkuName eq '{arm_sku}'"
    )
    try:
        resp = requests.get(
            _API_URL,
            params={"$filter": flt, "currencyCode": "USD"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        items = resp.json().get("Items", [])
    except Exception as e:  # noqa: BLE001
        log.warning("pricing fetch failed for %s: %s", arm_sku, e)
        return None

    candidates: list[float] = []
    for it in items:
        meter = (it.get("meterName") or "").lower()
        product = (it.get("productName") or "").lower()
        # Exclude Spot and Low Priority — we want pay-as-you-go pricing.
        if "spot" in meter or "low priority" in meter:
            continue
        # Exclude Windows-licensed offers; we deploy Linux pods.
        if "windows" in product:
            continue
        price = it.get("retailPrice")
        if isinstance(price, (int, float)) and price > 0:
            candidates.append(float(price))

    if not candidates:
        return None
    return min(candidates)


def get_pricing(
    sku_names: Iterable[str],
    region: str = AZURE_REGION,
    use_cache: bool = True,
) -> dict[str, float | None]:
    """Return `{vm_name: hourly_usd_or_None}` for each name in `sku_names`.

    A `None` value means the SKU has no on-demand Linux offer in `region`
    (typically because Azure doesn't sell that SKU there). Results are
    cached on disk for 24h per (region, sku) pair.
    """
    cache = _load_cache() if use_cache else {}
    region_cache = cache.setdefault(region, {})
    now = time.time()

    out: dict[str, float | None] = {}
    to_fetch: list[str] = []
    for name in sku_names:
        entry = region_cache.get(name)
        if (
            use_cache
            and isinstance(entry, dict)
            and (now - entry.get("ts", 0)) < _CACHE_TTL_SECONDS
            and "price" in entry
        ):
            out[name] = entry["price"]
        else:
            to_fetch.append(name)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(_fetch_one, vm_name_to_arm_sku(name), region): name
                for name in to_fetch
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    price = fut.result()
                except Exception:  # noqa: BLE001
                    price = None
                out[name] = price
                region_cache[name] = {"price": price, "ts": now}

        if use_cache:
            _save_cache(cache)

    return out
