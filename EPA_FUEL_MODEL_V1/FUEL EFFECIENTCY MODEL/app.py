from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.maritime import load_fleet_data, train_fuel_predictor
from src.optimizer import ScenarioSearch, Weights, run_benchmark


@st.cache_resource
def get_predictor():
    return train_fuel_predictor()


st.set_page_config(page_title="Green fleet decision support", page_icon=":material/sailing:", layout="wide")
st.title("Green fleet decision support")
st.caption(
    "Maritime-only fuel prediction and deployment planning. "
    "Classical quantum-inspired search; no quantum hardware is used."
)

try:
    fleet = load_fleet_data()
    predictor = get_predictor()
except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error(f"Could not load the maritime data or model: {error}")
    st.stop()

st.warning(
    f"{fleet.assumptions_label}. "
    "Voyages, fleet characteristics, fuel prices, and hourly operating costs are illustrative demo inputs."
)
st.caption(
    "Fuel price conversion: USD/tonne × USD/INR × density (kg/litre) ÷ 1,000 = INR/litre. "
    "Fuel cost = litres × INR/litre; total cost = fuel cost + (operating + maintenance INR/hour) × voyage hours."
)

with st.sidebar:
    st.header("Scenario")
    voyage_by_id = {voyage.voyage_id: voyage for voyage in fleet.voyages}
    voyage_id = st.selectbox("Voyage", options=list(voyage_by_id))
    voyage = voyage_by_id[voyage_id]
    cargos = [cargo for cargo in fleet.cargo_demands if cargo.voyage_id == voyage_id]
    if not cargos:
        st.error("No cargo demand is linked to this voyage.")
        st.stop()
    cargo = st.selectbox(
        "Cargo demand",
        options=cargos,
        format_func=lambda value: f"{value.cargo_id} · {value.cargo_type}",
    )
    eligible_vessels = [
        vessel
        for vessel in fleet.vessels
        if vessel.available and vessel.vessel_type in cargo.compatible_vessel_types
    ]
    max_demand = sum(vessel.capacity_tonnes for vessel in eligible_vessels)
    if max_demand <= 0:
        st.error("No available compatible vessels can serve this cargo.")
        st.stop()
    demand_tonnes = st.slider(
        "Cargo demand (tonnes)",
        min_value=100,
        max_value=max(100, int(max_demand)),
        value=min(max(100, int(cargo.weight_tonnes)), int(max_demand)),
        step=100,
    )
    st.subheader("Objective weights")
    fuel_weight = st.slider("Fuel", min_value=0, max_value=100, value=34)
    cost_weight = st.slider("Cost", min_value=0, max_value=100, value=33)
    emissions_weight = st.slider("Lifecycle emissions", min_value=0, max_value=100, value=33)
    evaluation_budget = st.select_slider(
        "Search evaluations per stochastic method",
        options=[256, 512, 1024, 2048],
        value=512,
    )
    seed = st.number_input("Reproducibility seed", min_value=0, max_value=2**31 - 1, value=42)
    optimize = st.button("Run scenario and benchmark", type="primary", icon=":material/play_arrow:")

weights = Weights(fuel=fuel_weight, cost=cost_weight, emissions=emissions_weight)
try:
    normalized_weights = weights.normalized()
except ValueError as error:
    st.error(str(error))
    st.stop()

metric_columns = st.columns(4)
metric_columns[0].metric("Available vessels", len(eligible_vessels))
metric_columns[1].metric("Alternative fuels", len(fleet.fuels) - 2)
metric_columns[2].metric("Voyage distance", f"{voyage.distance_nm:,.0f} nm")
metric_columns[3].metric("Delivery deadline", f"{min(voyage.required_arrival_hours, cargo.delivery_deadline_hours):,.0f} h")
st.caption(
    "Weights are normalized automatically: "
    f"fuel {normalized_weights['fuel']:.0%} · cost {normalized_weights['cost']:.0%} · "
    f"lifecycle emissions {normalized_weights['emissions']:.0%}."
)

with st.expander("Held-out fuel-prediction check", expanded=True):
    score_columns = st.columns(4)
    score_columns[0].metric("Model", "Extra Trees")
    score_columns[1].metric("Test MAE", f"{predictor.metrics['mae_litres']:,.0f} L")
    score_columns[2].metric("Test RMSE", f"{predictor.metrics['rmse_litres']:,.0f} L")
    score_columns[3].metric("Test R²", f"{predictor.metrics['r2']:.3f}")
    st.caption(
        f"Fixed 80/20 hold-out: {predictor.metrics['training_rows']:,} training rows and "
        f"{predictor.metrics['held_out_rows']:,} test rows, from {fleet.training_data_rows:,} "
        f"{predictor.metrics['data_label'].lower()}. These metrics do not establish performance on real voyages."
    )

benchmark_signature = (
    voyage.voyage_id,
    cargo.cargo_id,
    int(demand_tonnes),
    fuel_weight,
    cost_weight,
    emissions_weight,
    int(evaluation_budget),
    int(seed),
)
if optimize:
    with st.spinner("Evaluating the fleet search and conventional baselines..."):
        search = ScenarioSearch(
            fleet=fleet,
            predictor=predictor,
            voyage=voyage,
            cargo=cargo,
            demand_tonnes=demand_tonnes,
            weights=weights,
        )
        benchmark = run_benchmark(
            search,
            evaluation_budget=int(evaluation_budget),
            seed=int(seed),
        )
    st.session_state["benchmark_signature"] = benchmark_signature
    st.session_state["benchmark_results"] = benchmark
elif st.session_state.get("benchmark_signature") == benchmark_signature:
    benchmark = st.session_state["benchmark_results"]

if optimize or st.session_state.get("benchmark_signature") == benchmark_signature:
    st.subheader("Benchmark")
    comparison = pd.DataFrame(
        [
            {
                "Method": item["algorithm"],
                "Feasible": "Yes" if item["feasible"] else "No",
                "Objective (lower is better)": item["objective_score"],
                "Evaluations": item["evaluations"],
                "Runtime (s)": item["runtime_seconds"],
                "Fuel (L)": item["fuel_litres"],
                "Total cost (INR)": item["total_cost_inr"],
                "Lifecycle emissions (kg CO₂e)": item["lifecycle_emissions_kg"],
                "Arrival (h)": item["arrival_hours"],
            }
            for item in benchmark
        ]
    )
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(
        "Quantum-inspired categorical search and uniform random search use the same evaluation budget "
        "and seed. Greedy is a deterministic, one-pass baseline with its candidate count shown separately. "
        "Both stochastic methods start from the same greedy candidate. "
        "Results depend on the synthetic data, assumptions, seed, and selected weights; the best method "
        "may change from scenario to scenario."
    )
    st.download_button(
        "Download benchmark CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name=f"maritime-benchmark-{voyage.voyage_id}.csv",
        mime="text/csv",
    )

    best = min((item for item in benchmark if item["feasible"]), key=lambda item: item["objective_score"], default=None)
    if best is None:
        st.error(
            "No feasible deployment was found. Increase the delivery deadline or reduce the cargo demand "
            "before treating this scenario as deployable."
        )
    else:
        result = best["result"]
        st.subheader(f"Recommended deployment · {best['algorithm']}")
        result_columns = st.columns(4)
        result_columns[0].metric("Fuel", f"{result['fuel_litres']:,.0f} L")
        result_columns[1].metric("Total cost", f"₹{result['total_cost_inr']:,.0f}")
        result_columns[2].metric("Lifecycle emissions", f"{result['lifecycle_emissions_kg']:,.0f} kg CO₂e")
        result_columns[3].metric("Arrival", f"{result['arrival_hours']:,.1f} h")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Vessel": leg["vessel_name"],
                        "Type": leg["vessel_type"],
                        "Cargo (t)": leg["allocated_tonnes"],
                        "Speed (kn)": leg["speed_knots"],
                        "Fuel": leg["fuel_name"],
                        "Predicted fuel (L)": leg["predicted_fuel_litres"],
                        "Fuel price (INR/L)": leg["fuel_price_inr_per_litre"],
                        "Fuel cost (INR)": leg["fuel_cost_inr"],
                        "Operating + maintenance (INR)": leg["operational_cost_inr"],
                        "Tank feasible": leg["fuel_tank_feasible"],
                    }
                    for leg in result["legs"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.success(
            f"Cargo allocated: {result['allocated_tonnes']:,.0f} / {demand_tonnes:,.0f} tonnes. "
            f"Arrival deadline: {result['deadline_hours']:,.1f} hours."
        )
        st.download_button(
            "Download benchmark results JSON",
            data=json.dumps(
                {
                    "scenario": voyage.voyage_id,
                    "cargo_id": cargo.cargo_id,
                    "demand_tonnes": demand_tonnes,
                    "objective_weights": normalized_weights,
                    "random_seed": int(seed),
                    "data_label": "SYNTHETIC / DEMO SCENARIO",
                    "assumptions_label": fleet.assumptions_label,
                    "benchmark": [
                        {key: value for key, value in item.items() if key != "result"}
                        for item in benchmark
                    ],
                    "recommendation": result,
                },
                indent=2,
                allow_nan=False,
            ),
            file_name=f"maritime-scenario-{voyage.voyage_id}.json",
            mime="application/json",
        )
else:
    st.info("Set the scenario and objective weights, then run the benchmark to compare deployment plans.")

with st.expander("Fleet and fuel reference"):
    st.subheader("Compatible vessel inventory")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Vessel": vessel.vessel_name,
                    "Type": vessel.vessel_type,
                    "Cargo capacity (t)": vessel.capacity_tonnes,
                    "Cruise speed (kn)": vessel.cruise_speed_knots,
                    "Max speed (kn)": vessel.max_speed_knots,
                    "Operating cost (INR/h)": vessel.operating_cost_per_hour_inr,
                    "Maintenance (INR/h)": vessel.maintenance_cost_per_hour_inr,
                    "Allowed fuels": ", ".join(vessel.compatible_fuels),
                }
                for vessel in eligible_vessels
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Editable synthetic fuel-price assumptions")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Fuel": fuel.name,
                    "Illustrative USD/tonne": fuel.price_usd_per_tonne,
                    "Density (kg/L)": fuel.density_kg_per_litre,
                    "Derived INR/L": fuel.price_inr_per_litre,
                    "Lifecycle factor (kg CO₂e/kg)": fuel.lifecycle_kg_co2e_per_kg,
                }
                for fuel in fleet.fuels
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Exchange-rate assumption: USD 1 = INR {fleet.usd_inr_rate:g}. "
        f"{fleet.fx_source} Rate date: {fleet.fx_as_of_date or 'not set'}."
    )
