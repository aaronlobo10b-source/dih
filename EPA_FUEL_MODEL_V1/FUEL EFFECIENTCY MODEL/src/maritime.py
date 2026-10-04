"""Data loading, INR assumption handling, and held-out fuel model evaluation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "maritime_fleet.db"
DEFAULT_TRAINING_DATA = PROJECT_ROOT / "data" / "maritime_synthetic.csv"
DEFAULT_ASSUMPTIONS = PROJECT_ROOT / "config" / "maritime_assumptions.json"
TARGET = "consumption_litres"
NUMERIC_FEATURES = (
    "capacity_tonnes",
    "load_factor",
    "distance_nm",
    "speed_knots",
    "engine_power_kw",
    "engine_efficiency",
    "age_years",
    "weather_factor",
    "sea_state",
    "draft_m",
    "fuel_efficiency_factor",
    "duration_hours",
)
CATEGORICAL_FEATURES = ("vessel_type", "fuel_id")
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class Fuel:
    fuel_id: str
    name: str
    price_usd_per_tonne: float
    price_inr_per_litre: float
    density_kg_per_litre: float
    lhv_mj_per_kg: float
    direct_kg_co2e_per_kg: float
    lifecycle_kg_co2e_per_kg: float
    efficiency_factor: float
    available: bool


@dataclass(frozen=True)
class Vessel:
    vessel_id: str
    vessel_name: str
    vessel_type: str
    capacity_tonnes: float
    engine_power_kw: float
    engine_efficiency: float
    max_speed_knots: float
    cruise_speed_knots: float
    draft_m: float
    age_years: int
    operating_cost_per_hour_inr: float
    maintenance_cost_per_hour_inr: float
    fuel_tank_capacity_litres: float
    compatible_fuels: tuple[str, ...]
    available: bool


@dataclass(frozen=True)
class Voyage:
    voyage_id: str
    origin_port: str
    destination_port: str
    distance_nm: float
    required_arrival_hours: float
    weather_factor: float
    sea_state: int
    current_knots: float


@dataclass(frozen=True)
class CargoDemand:
    cargo_id: str
    voyage_id: str
    cargo_type: str
    weight_tonnes: float
    delivery_deadline_hours: float
    priority: int
    compatible_vessel_types: tuple[str, ...]


@dataclass(frozen=True)
class FleetData:
    fuels: tuple[Fuel, ...]
    vessels: tuple[Vessel, ...]
    voyages: tuple[Voyage, ...]
    cargo_demands: tuple[CargoDemand, ...]
    assumptions_label: str
    usd_inr_rate: float
    fx_source: str
    fx_as_of_date: str | None
    training_data_rows: int


@dataclass(frozen=True)
class FuelPredictor:
    model: Pipeline
    metrics: dict[str, float | int | str]

    def predict(self, feature_rows: list[dict[str, Any]]) -> np.ndarray:
        frame = pd.DataFrame(feature_rows, columns=FEATURES)
        predictions = self.model.predict(frame)
        return np.maximum(np.asarray(predictions, dtype=float), 0.0)


def _rows(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute(query)]


def load_fleet_data(
    database_path: Path = DEFAULT_DATABASE,
    assumptions_path: Path = DEFAULT_ASSUMPTIONS,
    training_data_path: Path = DEFAULT_TRAINING_DATA,
) -> FleetData:
    """Load maritime-only data and derive every fuel price from the assumptions file."""
    with assumptions_path.open(encoding="utf-8") as file:
        assumptions = json.load(file)

    required_assumption_keys = {
        "currency",
        "assumptions_label",
        "usd_inr_rate",
        "fx_source",
        "fx_as_of_date",
        "fuel_prices_usd_per_tonne",
        "fuel_density_kg_per_litre",
    }
    missing = required_assumption_keys - assumptions.keys()
    if missing:
        raise ValueError(f"Missing required currency assumptions: {sorted(missing)}")
    if assumptions["currency"] != "INR":
        raise ValueError("The maritime cost calculations are configured for INR.")
    usd_inr_rate = float(assumptions["usd_inr_rate"])
    if not np.isfinite(usd_inr_rate) or usd_inr_rate <= 0:
        raise ValueError("usd_inr_rate must be a positive finite number.")
    if not assumptions["assumptions_label"]:
        raise ValueError("The illustrative-cost disclaimer must not be blank.")

    with closing(sqlite3.connect(database_path)) as connection:
        fuel_rows = _rows(connection, "SELECT * FROM fuels ORDER BY fuel_id")
        vessel_rows = _rows(connection, "SELECT * FROM vessels ORDER BY vessel_id")
        voyage_rows = _rows(connection, "SELECT * FROM voyages ORDER BY voyage_id")
        cargo_rows = _rows(connection, "SELECT * FROM cargo_demands ORDER BY cargo_id")
        vessel_fuels = _rows(
            connection, "SELECT vessel_id, fuel_id FROM vessel_fuels ORDER BY vessel_id, fuel_id"
        )
        cargo_compatibility = _rows(
            connection,
            "SELECT cargo_id, vessel_type FROM cargo_vessel_compatibility "
            "ORDER BY cargo_id, vessel_type",
        )

    db_fuel_ids = {row["fuel_id"] for row in fuel_rows}
    prices = assumptions["fuel_prices_usd_per_tonne"]
    densities = assumptions["fuel_density_kg_per_litre"]
    if db_fuel_ids != set(prices) or db_fuel_ids != set(densities):
        raise ValueError(
            "Fuel IDs in the assumptions file, fuel-price map, and maritime database must match."
        )
    if not isinstance(assumptions["fx_source"], str) or not assumptions["fx_source"].strip():
        raise ValueError("fx_source must identify the exchange-rate assumption.")
    for fuel_id in db_fuel_ids:
        price = float(prices[fuel_id])
        density = float(densities[fuel_id])
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"Fuel price for {fuel_id} must be positive and finite.")
        if not np.isfinite(density) or density <= 0:
            raise ValueError(f"Fuel density for {fuel_id} must be positive and finite.")

    fuels = tuple(
        Fuel(
            fuel_id=row["fuel_id"],
            name=row["name"],
            price_usd_per_tonne=float(prices[row["fuel_id"]]),
            price_inr_per_litre=(
                float(prices[row["fuel_id"]])
                * usd_inr_rate
                * float(densities[row["fuel_id"]])
                / 1000.0
            ),
            density_kg_per_litre=float(densities[row["fuel_id"]]),
            lhv_mj_per_kg=float(row["lhv_mj_per_kg"]),
            direct_kg_co2e_per_kg=float(row["direct_kg_co2e_per_kg"]),
            lifecycle_kg_co2e_per_kg=float(row["lifecycle_kg_co2e_per_kg"]),
            efficiency_factor=float(row["efficiency_factor"]),
            available=bool(row["available"]),
        )
        for row in fuel_rows
    )
    fuels_by_vessel: dict[str, list[str]] = {}
    for row in vessel_fuels:
        fuels_by_vessel.setdefault(row["vessel_id"], []).append(row["fuel_id"])
    vessels = tuple(
        Vessel(
            vessel_id=row["vessel_id"],
            vessel_name=row["vessel_name"],
            vessel_type=row["vessel_type"],
            capacity_tonnes=float(row["cargo_capacity_tonnes"]),
            engine_power_kw=float(row["engine_power_kw"]),
            engine_efficiency=float(row["engine_efficiency"]),
            max_speed_knots=float(row["max_speed_knots"]),
            cruise_speed_knots=float(row["cruise_speed_knots"]),
            draft_m=float(row["draft_m"]),
            age_years=int(row["age_years"]),
            operating_cost_per_hour_inr=float(row["operating_cost_per_hour"]),
            maintenance_cost_per_hour_inr=float(row["maintenance_cost_per_hour"]),
            fuel_tank_capacity_litres=float(row["fuel_tank_capacity_litres"]),
            compatible_fuels=tuple(fuels_by_vessel.get(row["vessel_id"], ())),
            available=bool(row["available"]),
        )
        for row in vessel_rows
    )
    compatible_types: dict[str, list[str]] = {}
    for row in cargo_compatibility:
        compatible_types.setdefault(row["cargo_id"], []).append(row["vessel_type"])
    voyages = tuple(
        Voyage(
            voyage_id=row["voyage_id"],
            origin_port=row["origin_port"],
            destination_port=row["destination_port"],
            distance_nm=float(row["distance_nm"]),
            required_arrival_hours=float(row["required_arrival_hours"]),
            weather_factor=float(row["weather_factor"]),
            sea_state=int(row["sea_state"]),
            current_knots=float(row["current_knots"]),
        )
        for row in voyage_rows
    )
    cargo_demands = tuple(
        CargoDemand(
            cargo_id=row["cargo_id"],
            voyage_id=row["voyage_id"],
            cargo_type=row["cargo_type"],
            weight_tonnes=float(row["weight_tonnes"]),
            delivery_deadline_hours=float(row["delivery_deadline_hours"]),
            priority=int(row["priority"]),
            compatible_vessel_types=tuple(compatible_types.get(row["cargo_id"], ())),
        )
        for row in cargo_rows
    )
    with training_data_path.open(encoding="utf-8-sig") as file:
        training_rows = sum(1 for _ in file) - 1
    if training_rows < 2:
        raise ValueError("The maritime training dataset must contain at least two records.")

    return FleetData(
        fuels=fuels,
        vessels=vessels,
        voyages=voyages,
        cargo_demands=cargo_demands,
        assumptions_label=str(assumptions["assumptions_label"]),
        usd_inr_rate=usd_inr_rate,
        fx_source=str(assumptions["fx_source"]),
        fx_as_of_date=assumptions["fx_as_of_date"],
        training_data_rows=training_rows,
    )


def train_fuel_predictor(
    training_data_path: Path = DEFAULT_TRAINING_DATA,
) -> FuelPredictor:
    """Fit on a fixed training split and report metrics only on held-out rows."""
    data = pd.read_csv(training_data_path)
    missing = set((*FEATURES, TARGET)) - set(data.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")
    if data.empty or data[TARGET].isna().any() or (data[TARGET] <= 0).any():
        raise ValueError("Training data must have complete, positive fuel-consumption targets.")

    train_rows, test_rows = train_test_split(
        data, test_size=0.2, random_state=42
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=160,
                    min_samples_leaf=2,
                    max_features=0.9,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_rows.loc[:, FEATURES], train_rows[TARGET])
    predictions = model.predict(test_rows.loc[:, FEATURES])
    metrics: dict[str, float | int | str] = {
        "mae_litres": float(mean_absolute_error(test_rows[TARGET], predictions)),
        "rmse_litres": float(np.sqrt(mean_squared_error(test_rows[TARGET], predictions))),
        "r2": float(r2_score(test_rows[TARGET], predictions)),
        "training_rows": int(len(train_rows)),
        "held_out_rows": int(len(test_rows)),
        "data_label": "SYNTHETIC / DEMO DATA",
    }
    return FuelPredictor(model=model, metrics=metrics)
