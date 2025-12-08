function summary = export_igd_tree(dataRoot,algorithms,problems,objectives,decisions,optimums,runsToAverage,outputRoot)
%EXPORT_IGD_TREE Aggregate IGD means into a structured folder hierarchy.
%   SUMMARY = EXPORT_IGD_TREE(DATAROOT, ALGORITHMS, PROBLEMS, OBJECTIVES,
%   DECISIONS, OPTIMUMS) iterates over each combination of PROBLEMS, the
%   objective counts in OBJECTIVES, and the decision counts in DECISIONS. For
%   every algorithm in ALGORITHMS it searches DATAROOT/<algorithm> for result
%   files following the pattern '<algorithm>_<problem>_M<M>_D<D>_*.mat',
%   computes the IGD over the first 20 runs, and saves the per-run IGD values
%   and their mean to
%   <outputRoot>/<problem>_M<M>_D<D>/<algorithm>/average_igd.csv.
%
%   OPTIMUMS supplies the reference set for IGD calculation. It can be:
%     - a scalar (matrix or file path) applied to all problems,
%     - a struct/containers.Map with keys matching either
%       '<problem>_M<M>_D<D>' or '<problem>'.
%
%   SUMMARY is a struct array with fields configName, algorithm, csvPath, and
%   meanIGD for quick inspection of the outputs.
%
%   EXPORT_IGD_TREE(..., RUNSTOAVERAGE) overrides the number of runs averaged
%   (default 20). EXPORT_IGD_TREE(..., RUNSTOAVERAGE, OUTPUTROOT) overrides the
%   directory where the hierarchy is written (default
%   fullfile(DATAROOT,'IGD_Summary')).
%
%   Example:
%       algorithms = {'DVCOEA','EAGO','EAGOA','LERD','RKSTAEA','MDCS','RCCO'};
%       problems   = arrayfun(@(i) sprintf('LSMOP%d', i), 1:9, 'UniformOutput',false);
%       objectives = [10 15 20];
%       decisions  = [1000 2000 3000 5000];
%       summary = export_igd_tree('PlatEMO/Data', algorithms, problems, ...
%                                 objectives, decisions, 'PF.mat');
%       fprintf('Saved %d IGD CSV files.\n', numel(summary));
%
%   See also COMPUTE_AVERAGE_IGD.

    arguments
        dataRoot (1,:) char
        algorithms (1,:) cell
        problems (1,:) cell
        objectives (1,:) double {mustBePositive,mustBeInteger}
        decisions (1,:) double {mustBePositive,mustBeInteger}
        optimums
        runsToAverage (1,1) double {mustBePositive,mustBeInteger} = 20
        outputRoot (1,:) char = ''
    end

    if isempty(outputRoot)
        outputRoot = fullfile(dataRoot,'IGD_Summary');
    end

    % Preallocate summary with the maximum possible size to avoid growing in the
    % nested loops, then trim at the end.
    maxEntries = numel(problems) * numel(objectives) * numel(decisions) * numel(algorithms);
    summary(maxEntries) = struct('configName','', 'algorithm','', 'csvPath','', 'meanIGD',nan);
    idx = 0;

    for p = 1:numel(problems)
        problem = problems{p};
        for m = 1:numel(objectives)
            M = objectives(m);
            for d = 1:numel(decisions)
                D = decisions(d);
                configName = sprintf('%s_M%d_D%d', problem, M, D);

                for a = 1:numel(algorithms)
                    algorithm = algorithms{a};
                    resultDir = fullfile(dataRoot, algorithm);
                    pattern   = sprintf('%s_%s_M%d_D%d_*.mat', algorithm, problem, M, D);

                    outputCsv = fullfile(outputRoot, configName, algorithm, 'average_igd.csv');
                    optimum   = resolve_optimum(optimums, configName, problem);

                    stats = compute_average_igd(resultDir, optimum, runsToAverage, outputCsv, pattern);

                    idx = idx + 1;
                    summary(idx).configName = configName;
                    summary(idx).algorithm  = algorithm;
                    summary(idx).csvPath    = outputCsv;
                    summary(idx).meanIGD    = stats.meanIGD;
                end
            end
        end
    end

    summary = summary(1:idx);
end

function optimum = resolve_optimum(optimums, configKey, problem)
%RESOLVE_OPTIMUM Select the reference set for a configuration.
%   The precedence is: configuration-specific key, problem key, then fallback
%   to a scalar optimums value.

    if isa(optimums,'containers.Map')
        if isKey(optimums, configKey)
            optimum = optimums(configKey);
            return;
        end
        if isKey(optimums, problem)
            optimum = optimums(problem);
            return;
        end
    elseif isstruct(optimums)
        if isfield(optimums, configKey)
            optimum = optimums.(configKey);
            return;
        end
        if isfield(optimums, problem)
            optimum = optimums.(problem);
            return;
        end
    end

    % Fallback: treat OPTIMUMS as a scalar (path or matrix) applicable to all
    % configurations.
    optimum = optimums;
end
