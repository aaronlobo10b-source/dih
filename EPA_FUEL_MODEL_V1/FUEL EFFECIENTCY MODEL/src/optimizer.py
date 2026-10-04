"""Discrete fleet deployment optimization and same-budget search benchmarks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .maritime import (
    DEFAULT_TRAINING_DATA,
    CargoDemand,
    FleetData,
    Fuel,
    FuelPredictor,
    Vessel,
    Voyage,
)

SPEED_FACTORS = (0.8, 0.95, 1.1)
OBJECTIVES = ("fuel", "cost", "emissions")


@dataclass(frozen=True)
class Weights:
    fuel: float = 0.34
    cost: float = 0.33
    emissions: float = 0.33

    def normalized(self) -> dict[str, float]:
        values = {"fuel": self.fuel, "cost": self.cost, "emissions": self.emissions}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Objective weights must be finite and non-negative.")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("At least one objective weight must be positive.")
        return {key: value / total for key, value in values.items()}


class ScenarioSearch:
    """Search a single voyage's cargo deployment with hard capacity/time/fuel constraints."""

    def __init__(
        self,
        fleet: FleetData,
        predictor: FuelPredictor,
        voyage: Voyage,
        cargo: CargoDemand,
        demand_tonnes: float,
        weights: Weights,
    ) -> None:
        if not math.isfinite(demand_tonnes) or demand_tonnes <= 0:
            raise ValueError("Cargo demand must be a positive finite value.")
        if cargo.voyage_id != voyage.voyage_id:
            raise ValueError("The selected cargo demand must belong to the selected voyage.")
        self.fleet = fleet
        self.predictor = predictor
        self.voyage = voyage
        self.cargo = cargo
        self.demand_tonnes = float(demand_tonnes)
        self.weights = weights.normalized()
        self.fuels = {fuel.fuel_id: fuel for fuel in fleet.fuels}
        self.vessels = tuple(
            vessel
            for vessel in fleet.vessels
            if vessel.available
            and vessel.vessel_type in cargo.compatible_vessel_types
            and vessel.capacity_tonnes > 0
        )
        if not self.vessels:
            raise ValueError("No available cargo-compatible vessels were found.")

        self.options: list[list[tuple[str, float] | None]] = [
            [
                None,
                *(
                    (fuel_id, min(vessel.max_speed_knots, vessel.cruise_speed_knots * factor))
                    for fuel_id in vessel.compatible_fuels
                    if fuel_id in self.fuels and self.fuels[fuel_id].available
                    for factor in SPEED_FACTORS
                ),
            ]
            for vessel in self.vessels
        ]
        self._prediction_cache: dict[tuple[str, int, str, int], float] = {}
        self.scales = self._objective_scales()

    def _features(
        self,
        vessel: Vessel,
        fuel: Fuel,
        allocated_tonnes: float,
        speed_knots: float,
    ) -> dict[str, Any]:
        load_factor = min(max(allocated_tonnes / vessel.capacity_tonnes, 0.0), 1.0)
        effective_speed = speed_knots + self.voyage.current_knots
        if effective_speed <= 0:
            raise ValueError("The selected vessel speed and current do not produce positive progress.")
        duration = self.voyage.distance_nm / effective_speed * self.voyage.weather_factor
        return {
            "capacity_tonnes": vessel.capacity_tonnes,
            "load_factor": load_factor,
            "distance_nm": self.voyage.distance_nm,
            "speed_knots": speed_knots,
            "engine_power_kw": vessel.engine_power_kw,
            "engine_efficiency": vessel.engine_efficiency,
            "age_years": vessel.age_years,
            "weather_factor": self.voyage.weather_factor,
            "sea_state": self.voyage.sea_state,
            "draft_m": vessel.draft_m,
            "fuel_efficiency_factor": fuel.efficiency_factor,
            "duration_hours": duration,
            "vessel_type": vessel.vessel_type,
            "fuel_id": fuel.fuel_id,
        }

    def _predict(
        self,
        vessel: Vessel,
        fuel: Fuel,
        allocated_tonnes: float,
        speed_knots: float,
    ) -> tuple[float, float]:
        key = (
            vessel.vessel_id,
            int(round(allocated_tonnes)),
            fuel.fuel_id,
            int(round(speed_knots * 100)),
        )
        duration = (
            self.voyage.distance_nm
            / (speed_knots + self.voyage.current_knots)
            * self.voyage.weather_factor
        )
        if key not in self._prediction_cache:
            row = self._features(vessel, fuel, allocated_tonnes, speed_knots)
            self._prediction_cache[key] = float(self.predictor.predict([row])[0])
        return self._prediction_cache[key], duration

    def _objective_scales(self) -> dict[str, float]:
        """Use fixed dataset-derived unit scales so algorithm scores are comparable."""
        import pandas as pd

        training = pd.read_csv(DEFAULT_TRAINING_DATA)
        intensity = float(
            (
                training["consumption_litres"]
                / (training["capacity_tonnes"] * training["distance_nm"])
            ).median()
        )
        reference_fuel = max(
            1.0, intensity * self.demand_tonnes * self.voyage.distance_nm
        )
        mean_price = float(np.mean([fuel.price_inr_per_litre for fuel in self.fuels.values()]))
        mean_density_lifecycle = float(
            np.mean(
                [
                    fuel.density_kg_per_litre * fuel.lifecycle_kg_co2e_per_kg
                    for fuel in self.fuels.values()
                ]
            )
        )
        mean_hourly_cost = float(
            np.mean(
                [
                    vessel.operating_cost_per_hour_inr
                    + vessel.maintenance_cost_per_hour_inr
                    for vessel in self.vessels
                ]
            )
        )
        cruise = max(
            1.0, float(np.mean([vessel.cruise_speed_knots for vessel in self.vessels]))
        )
        expected_duration = self.voyage.distance_nm / cruise * self.voyage.weather_factor
        return {
            "fuel": reference_fuel,
            "cost": max(1.0, reference_fuel * mean_price + mean_hourly_cost * expected_duration),
            "emissions": max(1.0, reference_fuel * mean_density_lifecycle),
        }

    def evaluate(self, genome: tuple[int, ...] | list[int]) -> dict[str, Any]:
        if len(genome) != len(self.vessels):
            raise ValueError("Genome length must match the eligible vessel count.")
        selected: list[tuple[Vessel, Fuel, float]] = []
        for vessel, options, choice in zip(self.vessels, self.options, genome, strict=True):
            if not isinstance(choice, (int, np.integer)) or choice < 0 or choice >= len(options):
                raise ValueError("Genome contains a vessel option outside its allowed range.")
            option = options[choice]
            if option is not None:
                fuel_id, speed = option
                selected.append((vessel, self.fuels[fuel_id], speed))
        selected.sort(key=lambda item: (-item[0].capacity_tonnes, item[0].vessel_id))

        remaining = self.demand_tonnes
        legs: list[dict[str, Any]] = []
        violation = 0.0
        for vessel, fuel, speed in selected:
            if remaining <= 0:
                break
            allocated = min(remaining, vessel.capacity_tonnes)
            fuel_litres, duration = self._predict(vessel, fuel, allocated, speed)
            fuel_cost = fuel_litres * fuel.price_inr_per_litre
            operational_cost = (
                vessel.operating_cost_per_hour_inr
                + vessel.maintenance_cost_per_hour_inr
            ) * duration
            fuel_mass_kg = fuel_litres * fuel.density_kg_per_litre
            direct_emissions = fuel_mass_kg * fuel.direct_kg_co2e_per_kg
            lifecycle_emissions = fuel_mass_kg * fuel.lifecycle_kg_co2e_per_kg
            legs.append(
                {
                    "vessel_id": vessel.vessel_id,
                    "vessel_name": vessel.vessel_name,
                    "vessel_type": vessel.vessel_type,
                    "allocated_tonnes": allocated,
                    "speed_knots": speed,
                    "fuel_id": fuel.fuel_id,
                    "fuel_name": fuel.name,
                    "duration_hours": duration,
                    "predicted_fuel_litres": fuel_litres,
                    "fuel_price_inr_per_litre": fuel.price_inr_per_litre,
                    "fuel_cost_inr": fuel_cost,
                    "operational_cost_inr": operational_cost,
                    "direct_emissions_kg": direct_emissions,
                    "lifecycle_emissions_kg": lifecycle_emissions,
                    "fuel_tank_capacity_litres": vessel.fuel_tank_capacity_litres,
                    "fuel_tank_feasible": fuel_litres <= vessel.fuel_tank_capacity_litres,
                }
            )
            remaining -= allocated

        if remaining > 1e-7:
            violation += remaining / self.demand_tonnes
        if not legs:
            violation += 1.0

        total_fuel = sum(leg["predicted_fuel_litres"] for leg in legs)
        fuel_cost = sum(leg["fuel_cost_inr"] for leg in legs)
        operational_cost = sum(leg["operational_cost_inr"] for leg in legs)
        total_cost = fuel_cost + operational_cost
        direct_emissions = sum(leg["direct_emissions_kg"] for leg in legs)
        lifecycle_emissions = sum(leg["lifecycle_emissions_kg"] for leg in legs)
        arrival_hours = max((leg["duration_hours"] for leg in legs), default=0.0)
        for leg in legs:
            if not leg["fuel_tank_feasible"]:
                violation += (
                    leg["predicted_fuel_litres"] - leg["fuel_tank_capacity_litres"]
                ) / leg["fuel_tank_capacity_litres"]
        deadline = min(
            self.voyage.required_arrival_hours, self.cargo.delivery_deadline_hours
        )
        if arrival_hours > deadline:
            violation += (arrival_hours - deadline) / max(deadline, 1.0)

        normalized_components = {
            "fuel": total_fuel / self.scales["fuel"],
            "cost": total_cost / self.scales["cost"],
            "emissions": lifecycle_emissions / self.scales["emissions"],
        }
        objective_score = sum(
            self.weights[name] * normalized_components[name] for name in OBJECTIVES
        )
        feasible = violation <= 1e-9
        return {
            "feasible": feasible,
            "violation": violation,
            "objective_score": objective_score if feasible else 1_000_000.0 + violation,
            "normalized_components": normalized_components,
            "demand_tonnes": self.demand_tonnes,
            "allocated_tonnes": self.demand_tonnes - remaining,
            "fuel_litres": total_fuel,
            "fuel_cost_inr": fuel_cost,
            "operational_cost_inr": operational_cost,
            "total_cost_inr": total_cost,
            "direct_emissions_kg": direct_emissions,
            "lifecycle_emissions_kg": lifecycle_emissions,
            "arrival_hours": arrival_hours,
            "deadline_hours": deadline,
            "legs": legs,
        }

    def greedy_plan(self) -> tuple[tuple[int, ...], int]:
        """Greedily add compatible vessel configurations with the lowest unit objective."""
        selected: set[int] = set()
        genome = [0] * len(self.vessels)
        examined = 0
        while sum(
            self.vessels[index].capacity_tonnes for index in selected
        ) < self.demand_tonnes:
            best: tuple[float, int, int] | None = None
            for vessel_index, options in enumerate(self.options):
                if vessel_index in selected:
                    continue
                for option_index in range(1, len(options)):
                    trial = genome.copy()
                    trial[vessel_index] = option_index
                    result = self.evaluate(trial)
                    examined += 1
                    leg = next(
                        (item for item in result["legs"] if item["vessel_id"] == self.vessels[vessel_index].vessel_id),
                        None,
                    )
                    if leg is None or leg["duration_hours"] > result["deadline_hours"]:
                        continue
                    unit_score = (
                        self.weights["fuel"] * leg["predicted_fuel_litres"] / self.scales["fuel"]
                        + self.weights["cost"]
                        * (leg["fuel_cost_inr"] + leg["operational_cost_inr"])
                        / self.scales["cost"]
                        + self.weights["emissions"]
                        * leg["lifecycle_emissions_kg"]
                        / self.scales["emissions"]
                    ) / leg["allocated_tonnes"]
                    candidate = (unit_score, vessel_index, option_index)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                break
            _, vessel_index, option_index = best
            selected.add(vessel_index)
            genome[vessel_index] = option_index
        return tuple(genome), examined


def run_benchmark(
    search: ScenarioSearch,
    evaluation_budget: int = 1024,
    seed: int = 42,
    population_size: int = 32,
) -> list[dict[str, Any]]:
    """Compare categorical QEA, uniform random search, and a one-pass greedy baseline."""
    if evaluation_budget < population_size * 2 or evaluation_budget % population_size:
        raise ValueError("Evaluation budget must be a multiple of population_size and at least twice it.")
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")

    greedy_started = time.perf_counter()
    greedy_genome, greedy_evaluations = search.greedy_plan()
    greedy_result = search.evaluate(greedy_genome)
    greedy_runtime = time.perf_counter() - greedy_started

    random_generator = np.random.default_rng(seed)
    random_best = greedy_result
    random_evaluations = 0
    random_started = time.perf_counter()
    for _ in range(evaluation_budget):
        genome = tuple(
            int(random_generator.integers(0, len(options))) for options in search.options
        )
        result = search.evaluate(genome)
        random_evaluations += 1
        if result["objective_score"] < random_best["objective_score"]:
            random_best = result
    random_runtime = time.perf_counter() - random_started

    q_generator = np.random.default_rng(seed)
    amplitudes = [
        np.full(len(options), 1.0 / math.sqrt(len(options)), dtype=float)
        for options in search.options
    ]
    quantum_best = greedy_result
    quantum_evaluations = 0
    quantum_started = time.perf_counter()
    generations = evaluation_budget // population_size
    for _ in range(generations):
        population = [
            tuple(
                int(q_generator.choice(len(probabilities), p=probabilities**2))
                for probabilities in amplitudes
            )
            for _ in range(population_size)
        ]
        evaluated_population: list[tuple[tuple[int, ...], dict[str, Any]]] = []
        for genome in population:
            result = search.evaluate(genome)
            evaluated_population.append((genome, result))
            quantum_evaluations += 1
            if result["objective_score"] < quantum_best["objective_score"]:
                quantum_best = result
        target_genome, _ = min(
            evaluated_population, key=lambda item: item[1]["objective_score"]
        )
        learning_rate = 0.18
        mutation_rate = 0.025
        for gene_index, probabilities in enumerate(amplitudes):
            target = np.zeros_like(probabilities)
            target[target_genome[gene_index]] = 1.0
            current = probabilities**2
            updated = (1.0 - learning_rate) * current + learning_rate * target
            updated = (1.0 - mutation_rate) * updated + mutation_rate / len(updated)
            amplitudes[gene_index] = np.sqrt(updated / updated.sum())
    quantum_runtime = time.perf_counter() - quantum_started

    def result_row(
        name: str,
        result: dict[str, Any],
        runtime: float,
        evaluations: int,
    ) -> dict[str, Any]:
        return {
            "algorithm": name,
            "objective_score": result["objective_score"],
            "feasible": result["feasible"],
            "evaluations": evaluations,
            "runtime_seconds": runtime,
            "fuel_litres": result["fuel_litres"],
            "total_cost_inr": result["total_cost_inr"],
            "lifecycle_emissions_kg": result["lifecycle_emissions_kg"],
            "arrival_hours": result["arrival_hours"],
            "result": result,
        }

    return [
        result_row("Quantum-inspired categorical search", quantum_best, quantum_runtime, quantum_evaluations),
        result_row("Uniform random search", random_best, random_runtime, random_evaluations),
        result_row("Greedy fleet heuristic", greedy_result, greedy_runtime, greedy_evaluations + 1),
    ]
