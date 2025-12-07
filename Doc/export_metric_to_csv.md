# Exporting metric curves from PlatEMO logs to CSV

Use `export_metric_to_csv.m` to extract any recorded performance metric (for example IGD or HV) from a PlatEMO experiment log and save it as a CSV file. The CSV will contain two columns: the cumulative number of function evaluations (`FE`) and the selected metric.

## Usage
```matlab
export_metric_to_csv('Data/NSGAII/NSGAII_DTLZ2_M2_D12_1.mat', ...
                     'IGD', 'igd_curve.csv');
```

* **MATFILE**: Path to the `.mat` file produced by PlatEMO in the `Data` folder.
* **METRICNAME**: Name of the metric stored in the file (for example `IGD`, `HV`, `GD`).
* **OUTCSV**: Output CSV path, e.g., `igd_curve.csv`.

After running, `OUTCSV` will have a header row (`FE,<metric name>`) followed by one row per generation. You can import this CSV into MATLAB, Python, Excel, or plotting tools to draw convergence curves.
