function export_metric_to_csv(matFile, metricName, outCsv)
%EXPORT_METRIC_TO_CSV Extract a convergence metric from PlatEMO .mat log and save as CSV.
%   EXPORT_METRIC_TO_CSV(MATFILE, METRICNAME, OUTCSV) loads the given
%   MATFILE produced by PlatEMO (e.g., Data/NSGAII/...mat), reads the
%   cumulative function evaluations and the specified metric sequence, and
%   writes them to OUTCSV with two columns: FE and the chosen metric.
%
%   Example:
%       export_metric_to_csv('Data/NSGAII/NSGAII_DTLZ2_M2_D12_1.mat', ...
%                            'IGD', 'igd_curve.csv');
%
%   The resulting CSV will contain a header row with columns "FE" and the
%   metric name.

arguments
    matFile (1,1) string
    metricName (1,1) string
    outCsv (1,1) string
end

% Load the result and metric structures from the MAT file
raw = load(matFile, 'result', 'metric');

if ~isfield(raw, 'result') || ~isfield(raw, 'metric')
    error('The MAT file must contain ''result'' and ''metric'' variables.');
end

if ~iscell(raw.result) || size(raw.result, 2) < 1
    error('The ''result'' variable must be a cell array with FE values in column 1.');
end

if ~isfield(raw.metric, metricName)
    error('Metric "%s" not found in the MAT file.', metricName);
end

% Extract FE counts and metric values
feValues = cell2mat(raw.result(:,1));
metricValues = raw.metric.(metricName)(:);

if numel(feValues) ~= numel(metricValues)
    error('The lengths of FE values and the metric "%s" do not match.', metricName);
end

% Create a table with clear column names
T = table(feValues, metricValues, 'VariableNames', {'FE', char(metricName)});

% Write to CSV
writetable(T, outCsv);

end
