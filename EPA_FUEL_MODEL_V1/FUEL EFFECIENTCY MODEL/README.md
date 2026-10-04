# Green fleet decision support

This maritime-only prototype predicts ship fuel consumption and compares fleet deployments for a selected voyage and cargo demand. It includes an interactive Streamlit interface, a held-out prediction check, a discrete quantum-inspired search, conventional search baselines, and downloadable scenario results.

## Run locally

From this folder, create/activate a Python environment, install the dependencies, and start the app:

```powershell
python -m venv .venv-app
.\.venv-app\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```

## What the demo does

- Trains an Extra Trees regressor on `data/maritime_synthetic.csv`. It uses a fixed 80/20 train/test split and reports MAE, RMSE, and R² on held-out rows. Every model feature is a vessel, fuel, route, speed, load, or weather field; fuel consumption is the target, not an input.
- Uses `data/maritime_fleet.db` to load vessels, fuel compatibility, voyages, cargo demands, and cargo-vessel compatibility.
- Searches vessel, speed, and compatible fuel choices for a single selected voyage/cargo scenario. It enforces cargo capacity, vessel fuel compatibility, fuel-tank capacity, and the earlier of the voyage and cargo deadlines.
- Minimizes a weighted, dataset-scaled objective over fuel use, total INR cost, and lifecycle emissions. The weights are normalized automatically.
- Compares quantum-inspired categorical search with uniform random search at the same evaluation budget and seed, and shows a one-pass greedy fleet heuristic as an additional conventional baseline. Runtime and evaluation counts are reported; no method is presumed to win.
- Warm-starts both stochastic methods from the same greedy candidate; that seed candidate is not counted in their reported stochastic search evaluations.
- Allows reproducible settings and scenario outputs to be downloaded as CSV and JSON.

The quantum-inspired search uses classical categorical probability amplitudes and sampling. It does **not** use a quantum computer, quantum annealer, or quantum speedup.

## Currency and illustrative inputs

`config/maritime_assumptions.json` is the single editable source for USD-per-tonne fuel prices, fuel densities, the USD/INR exchange-rate assumption, and their provenance. For each fuel:

```text
price_INR_per_litre =
    price_USD_per_tonne × USD_INR_rate × density_kg_per_litre ÷ 1,000
```

Fuel cost is predicted litres × derived INR/litre. Total operational cost is fuel cost plus (vessel operating cost/hour + maintenance cost/hour) × voyage hours. The current fuel prices and USD/INR rate are illustrative placeholders. Vessel hourly costs, route data, vessel specifications, fuel factors, and emission factors are also demo assumptions. The interface labels them as synthetic; none should be presented as current supplier quotes or operational data.

Update the price, rate, density, exchange-rate source, and `fx_as_of_date` together before presenting any cost estimate as current. The built-in currency formula is tested against the example `USD 750/tonne × INR 85/USD × 0.84 kg/L = INR 53.55/L`.

## Data and limitations

The fleet database and 1,800-row maritime CSV are synthetic/demo data. The held-out prediction score measures fit to a held-out portion of that same synthetic dataset; it is not validation on observed voyages. Legacy optimization totals and fuel-price columns already stored in the database are not used by the app; current prices are recalculated from the editable assumptions file. The generated recommendation is a prototype decision aid, not a seaworthiness, engineering, safety, regulatory, or procurement approval.

Optimization is scoped to **one selected voyage and its cargo at a time**. It does not coordinate vessel availability across simultaneous voyages, model port turnaround or route networks, or establish real-world lifecycle-emissions factors. Benchmark outcomes can change by seed, scenario, and objective weights. Use real, consistently sourced fuel, vessel, voyage, emissions, and price data plus domain validation before operational use.
