import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.maritime import load_fleet_data, train_fuel_predictor
from src.optimizer import ScenarioSearch, Weights, run_benchmark


class MaritimePlatformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fleet = load_fleet_data()
        cls.predictor = train_fuel_predictor()

    def test_marine_diesel_price_is_derived_from_editable_assumptions(self):
        mdo = next(fuel for fuel in self.fleet.fuels if fuel.fuel_id == "mdo")
        self.assertAlmostEqual(mdo.price_inr_per_litre, 53.55, places=8)

    def test_held_out_model_metrics_and_positive_prediction(self):
        metrics = self.predictor.metrics
        self.assertEqual(metrics["data_label"], "SYNTHETIC / DEMO DATA")
        self.assertEqual(metrics["training_rows"] + metrics["held_out_rows"], 1800)
        self.assertGreaterEqual(metrics["mae_litres"], 0)
        self.assertGreaterEqual(metrics["rmse_litres"], 0)
        voyage = self.fleet.voyages[0]
        cargo = next(item for item in self.fleet.cargo_demands if item.voyage_id == voyage.voyage_id)
        search = ScenarioSearch(
            self.fleet, self.predictor, voyage, cargo, cargo.weight_tonnes, Weights()
        )
        result = search.evaluate(tuple(1 for _ in search.vessels))
        self.assertGreater(result["fuel_litres"], 0)

    def test_benchmark_uses_equal_search_budgets_and_checks_constraints(self):
        voyage = self.fleet.voyages[0]
        cargo = next(item for item in self.fleet.cargo_demands if item.voyage_id == voyage.voyage_id)
        search = ScenarioSearch(
            self.fleet, self.predictor, voyage, cargo, cargo.weight_tonnes, Weights()
        )
        benchmark = run_benchmark(search, evaluation_budget=64, population_size=8, seed=10)
        by_name = {item["algorithm"]: item for item in benchmark}
        self.assertEqual(
            by_name["Quantum-inspired categorical search"]["evaluations"], 64
        )
        self.assertEqual(by_name["Uniform random search"]["evaluations"], 64)
        self.assertIn("Greedy fleet heuristic", by_name)
        best = min(
            (item for item in benchmark if item["feasible"]),
            key=lambda item: item["objective_score"],
        )
        result = best["result"]
        self.assertAlmostEqual(result["allocated_tonnes"], cargo.weight_tonnes)
        self.assertLessEqual(result["arrival_hours"], result["deadline_hours"])
        self.assertTrue(all(leg["fuel_tank_feasible"] for leg in result["legs"]))

    def test_second_voyage_is_feasible_and_impossible_cargo_is_rejected(self):
        voyage = self.fleet.voyages[1]
        cargo = next(item for item in self.fleet.cargo_demands if item.voyage_id == voyage.voyage_id)
        eligible_capacity = sum(
            vessel.capacity_tonnes
            for vessel in self.fleet.vessels
            if vessel.available and vessel.vessel_type in cargo.compatible_vessel_types
        )
        search = ScenarioSearch(
            self.fleet, self.predictor, voyage, cargo, cargo.weight_tonnes, Weights()
        )
        greedy_genome, _ = search.greedy_plan()
        self.assertTrue(search.evaluate(greedy_genome)["feasible"])

        impossible = ScenarioSearch(
            self.fleet,
            self.predictor,
            voyage,
            cargo,
            eligible_capacity + 1,
            Weights(),
        )
        infeasible_result = impossible.evaluate(
            tuple(len(options) - 1 for options in impossible.options)
        )
        self.assertFalse(infeasible_result["feasible"])
        self.assertGreater(infeasible_result["violation"], 0)

    def test_zero_objective_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            Weights(0, 0, 0).normalized()

    def test_app_renders_and_keeps_benchmark_after_download_reruns(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        self.assertFalse(app.exception)

        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("Quantum-inspired categorical search" in str(frame.value) for frame in app.dataframe))
        self.assertIn("benchmark_results", app.session_state)

        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("Quantum-inspired categorical search" in str(frame.value) for frame in app.dataframe))


if __name__ == "__main__":
    unittest.main()
