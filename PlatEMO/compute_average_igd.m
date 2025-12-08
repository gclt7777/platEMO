function stats = compute_average_igd(resultDir,optimum,numRunsToAverage,outputCsvPath)
%COMPUTE_AVERAGE_IGD Compute the IGD statistics from a set of result files.
%   STATS = COMPUTE_AVERAGE_IGD(RESULTDIR, OPTIMUM) loads all .mat files in
%   RESULTDIR, extracts the objective values with LOAD_OBJS, and evaluates
%   the inverted generational distance (IGD) against OPTIMUM. OPTIMUM can be
%   either a matrix of reference points or the path to a .mat file that
%   contains them. The function returns the IGD values for each processed
%   file and their mean in the struct STATS.
%
%   STATS = COMPUTE_AVERAGE_IGD(RESULTDIR, OPTIMUM, NUMRUNSTOAVERAGE)
%   limits the calculation to the first NUMRUNSTOAVERAGE result files (after
%   sorting them alphabetically). This is useful when you want the average
%   over a subset such as 20 runs out of 30.
%
%   STATS = COMPUTE_AVERAGE_IGD(..., OUTPUTCSVPATH) additionally writes the
%   IGD values (per run and the mean) to the CSV file specified by
%   OUTPUTCSVPATH. Directories in OUTPUTCSVPATH are created automatically when
%   needed.
%
%   Example:
%       % Compute the mean IGD of the first 20 runs stored under Data/RCCO
%       % using a reference set saved in "PF.mat".
%       stats = compute_average_igd('PlatEMO/Data/RCCO','PF.mat',20);
%       fprintf('Mean IGD over %d runs: %g\n',numel(stats.igdPerRun),stats.meanIGD);
%
%   The returned struct has two fields:
%       - igdPerRun : A column vector with one IGD value per processed file.
%       - meanIGD   : The arithmetic mean of igdPerRun.
%
%   This helper relies on PlatEMO's LOAD_OBJS and IGD utilities so it can
%   parse common result formats produced by PlatEMO experiments.

    arguments
        resultDir (1,:) char
        optimum
        numRunsToAverage (1,1) double {mustBePositive,mustBeInteger} = inf
        outputCsvPath (1,:) char = ''
    end

    files = dir(fullfile(resultDir,'*.mat'));
    if isempty(files)
        error('No .mat files found in %s.',resultDir);
    end

    % Sort files deterministically so that the first N runs are consistent.
    [~,order] = sort({files.name});
    files = files(order);

    % Allow selecting a subset (e.g., 20 of 30) while respecting available files.
    numRuns = min(numRunsToAverage,numel(files));
    igdValues = nan(numRuns,1);

    if ischar(optimum) || isstring(optimum)
        optimum = load_objs(char(optimum));
    end

    for i = 1:numRuns
        objs = load_objs(fullfile(resultDir,files(i).name));
        igdValues(i) = IGD(objs,optimum);
    end

    stats.igdPerRun = igdValues;
    stats.meanIGD   = mean(igdValues);

    if ~isempty(outputCsvPath)
        [outputDir,~,~] = fileparts(outputCsvPath);
        if ~isempty(outputDir) && ~isfolder(outputDir)
            mkdir(outputDir);
        end

        runLabels = [compose('Run%d',1:numRuns)';"Mean"];
        igdForCsv = [igdValues; stats.meanIGD];
        igdTable = table(runLabels, igdForCsv, 'VariableNames',{'Run','IGD'});
        writetable(igdTable, outputCsvPath);

        stats.csvPath = outputCsvPath;
    end
end
