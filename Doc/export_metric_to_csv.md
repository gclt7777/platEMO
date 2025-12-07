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

## Multiple runs (e.g., 30 trials)
PlatEMO 默认每次运行都会生成一个 `.mat` 文件，其中的 `metric.IGD` 就是一条完整的收敛曲线。若进行了 30 次独立运行，常见做法是：

* **单条曲线**：直接选取一条运行的曲线（通常选中位数或平均表现的那一条），而不是 30 次中“最优”的曲线，这样更能反映算法的典型表现。
* **汇总曲线**：也可以在每个迭代/FE 点对 30 条曲线做逐点平均或中位数，得到一条平滑的平均/中位 IGD 曲线；需要保证各曲线的 FE 序列对齐或先做插值。
* **最终指标**：如果只关注最终 IGD 值，才会按 30 次运行的最终 IGD 取最佳/最差/平均/中位数等统计量。

因此，画 IGD 收敛曲线时通常不会直接使用“30 次运行中最好的那条”，而是选取代表性的单次运行，或对多条曲线做均值/中位数汇总。

## PlatEMO GUI 如何绘制 IGD 收敛曲线
在 PlatEMO 的“Experimental module”里，选中表格中的 IGD 指标并点击图标显示曲线时，后台调用 `module_exp.GetMetricValue(..., showAll=true)`：

1. 对所选算法×问题组合的每次运行，读取 `metric.IGD`（必要时即时计算）；
2. 将每次运行的 FE 序列和 IGD 序列逐点堆叠后分别做简单平均；
3. 用平均 FE 作为横轴、平均 IGD 作为纵轴画出一条“Mean convergence profile”。

也就是说，GUI 自带的 IGD 曲线不是取 30 次运行中的最佳/单次曲线，而是对可用的每次运行进行逐点平均后绘制的均值收敛曲线。

> 结论：如果运行了 30 次（或其他次数），GUI 默认就是把这 30 条曲线在相同迭代/FE 位置上做算术平均后绘图，并不会挑选其中“最好”的那条。
