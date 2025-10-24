function totalTime = SAGIF(objs, ref, outDir, pro, tc, samplingFactor, selectionSize, seed)
%SAHVES Surrogate-assisted hypervolume environmental selection.
%   Mirrors the Python SAGIF implementation with sequential surrogate-based
%   survivor selection and Monte Carlo hypervolume updates.
%
%   totalTime = SAHVES(objs, ref, outDir, pro, tc, samplingFactor,
%   selectionSize, seed)
%
%   Inputs
%   ------
%   objs           : Population objective matrix (populationSize x M).
%   ref            : Reference point used for hypervolume computation.
%   outDir         : Directory for CSV summaries (optional).
%   pro            : Problem identifier string.
%   tc             : Time budget in seconds (Inf for no limit).
%   samplingFactor : Fraction of filtered candidates evaluated exactly.
%   selectionSize  : Number of survivors to retain.
%   seed           : Random seed (optional).
%
%   Output
%   ------
%   totalTime      : Total time consumed by the selection loop.

    if nargin < 5 || isempty(tc)
        tc = Inf;
    end
    if nargin < 6 || isempty(samplingFactor)
        samplingFactor = 0.035;
    end
    if nargin < 7 || isempty(selectionSize)
        selectionSize = 100;
    end
    if nargin < 8
        seed = [];
    end

    validateattributes(objs, {'numeric'}, {'2d', 'real', 'finite'}, mfilename, 'objs');
    validateattributes(ref, {'numeric'}, {'vector', 'real', 'finite'}, mfilename, 'ref');

    objs = double(objs);
    ref = double(ref(:)');
    [populationSize, numObj] = size(objs);
    if numel(ref) ~= numObj
        error('Reference point must match the number of objectives.');
    end

    if ~isempty(seed)
        rng(seed, 'twister');
    else
        rng('shuffle');
    end

    selectionSize = min(selectionSize, populationSize);

    MONTE_CARLO_SAMPLES = 8000;
    FILTER_FRACTION = 0.7;

    selectedIndices = zeros(1, 0);
    remaining = 1:populationSize;
    selectedObjs = zeros(0, numObj);
    currentHV = 0.0;

    trainingX = zeros(0, numObj);
    trainingY = zeros(0, 1);
    surrogate = initRBFN(numObj);
    maxTrainingSize = max(5 * selectionSize, 200);

    timer = tic;
    while numel(selectedIndices) < selectionSize && ~isempty(remaining)
        elapsed = toc(timer);
        if isfinite(tc) && elapsed >= tc
            break;
        end

        candidateObjs = objs(remaining, :);
        if surrogate.trained
            predictions = rbfnPredict(surrogate, candidateObjs);
        else
            predictions = prod(max(ref - candidateObjs, 1e-12), 2);
        end

        [~, bestPos] = max(predictions);
        chosenIdx = remaining(bestPos);
        chosenObj = objs(chosenIdx, :);

        if isempty(selectedObjs)
            newSelectedObjs = chosenObj;
        else
            newSelectedObjs = [selectedObjs; chosenObj];
        end

        hvWithCandidate = computeHV(newSelectedObjs, ref, MONTE_CARLO_SAMPLES);
        contribution = hvWithCandidate - currentHV;
        if contribution <= 0
            contribution = max(contribution, 1e-12);
        end

        selectedIndices(end + 1) = chosenIdx; %#ok<AGROW>
        selectedObjs = newSelectedObjs;
        currentHV = hvWithCandidate;

        remaining(bestPos) = [];

        [trainingX, trainingY] = updateTrainingSet(trainingX, trainingY, chosenObj, contribution, maxTrainingSize);

        elapsed = toc(timer);
        if numel(selectedIndices) >= selectionSize || (isfinite(tc) && elapsed >= tc) || isempty(remaining)
            break;
        end

        filtered = l1Filter(objs, remaining, selectedObjs, FILTER_FRACTION);
        sampleSize = max(1, round(numel(filtered) * samplingFactor));
        sampleSize = min(sampleSize, numel(filtered));
        sampleIndices = sampleWithoutReplacement(filtered, sampleSize);

        if ~isempty(sampleIndices)
            sampleObjs = objs(sampleIndices, :);
            sampleHV = zeros(numel(sampleIndices), 1);
            for i = 1:numel(sampleIndices)
                augmented = [selectedObjs; sampleObjs(i, :)];
                hvValue = computeHV(augmented, ref, MONTE_CARLO_SAMPLES);
                delta = hvValue - currentHV;
                if delta <= 0
                    delta = max(delta, 1e-12);
                end
                sampleHV(i) = delta;
            end
            [trainingX, trainingY] = updateTrainingSet(trainingX, trainingY, sampleObjs, sampleHV, maxTrainingSize);
        end

        if size(trainingX, 1) >= 1
            surrogate = rbfnFit(surrogate, trainingX, trainingY);
        end
    end

    totalTime = toc(timer);

    if nargin >= 3 && ~isempty(outDir)
        if exist(outDir, 'dir') ~= 7
            mkdir(outDir);
        end
        fileName = fullfile(outDir, sprintf('sagifr_%s_%d.csv', pro, numObj));
        row = [currentHV, numel(selectedIndices), totalTime, double(selectedIndices - 1)];
        fid = fopen(fileName, 'a');
        if fid ~= -1
            cleaner = onCleanup(@() fclose(fid));
            fmt = [repmat('%g,', 1, numel(row) - 1), '%g\n'];
            fprintf(fid, fmt, row);
            clear cleaner;
        end
    end
end

function hv = computeHV(points, ref, sampleCount)
    if isempty(points)
        hv = 0.0;
        return;
    end
    points = min(points, ref - 1e-9);
    lower = min(points, [], 1);
    lower = min(lower, ref - 1.0);
    lower = min(lower, ref - 1e-4);
    span = max(ref - lower, 1e-9);

    samples = lower + span .* rand(sampleCount, size(points, 2));
    pointTensor = reshape(points, [size(points, 1), size(points, 2), 1]);
    sampleTensor = permute(samples, [3, 2, 1]);
    mask = all(bsxfun(@le, pointTensor, sampleTensor), 2);
    dominated = squeeze(mask);
    if isempty(dominated)
        dominated = false(0, sampleCount);
    end
    if size(dominated, 1) == 1
        dominated = dominated.';
    end
    dominatedAny = any(dominated, 1);
    hv = prod(span) * mean(dominatedAny);
end

function filtered = l1Filter(objs, candidates, selectedObjs, fraction)
    if isempty(candidates) || isempty(selectedObjs)
        filtered = candidates;
        return;
    end
    subsetSize = max(1, ceil(numel(candidates) * fraction));
    distances = zeros(numel(candidates), 1);
    for idx = 1:numel(candidates)
        diff = abs(selectedObjs - objs(candidates(idx), :));
        distances(idx) = sum(diff(:));
    end
    [~, order] = sort(distances, 'descend');
    order = order(1:subsetSize);
    filtered = candidates(order);
end

function choice = sampleWithoutReplacement(candidates, sz)
    if isempty(candidates)
        choice = candidates;
        return;
    end
    sz = max(1, min(sz, numel(candidates)));
    perm = randperm(numel(candidates), sz);
    choice = candidates(perm);
end

function [trainingX, trainingY] = updateTrainingSet(trainingX, trainingY, newX, newY, maxSize)
    if isempty(newX)
        return;
    end
    if size(newX, 1) ~= numel(newY)
        newY = newY(:);
    end
    if size(newX, 1) == 1 && numel(newY) == 1
        newX = reshape(newX, 1, []);
    end
    if isempty(trainingX)
        trainingX = newX;
        trainingY = newY(:);
    else
        trainingX = [trainingX; newX];
        trainingY = [trainingY; newY(:)];
    end
    excess = size(trainingX, 1) - maxSize;
    if excess > 0
        trainingX(1:excess, :) = [];
        trainingY(1:excess) = [];
    end
end

function model = initRBFN(inputShape)
    model.inputShape = inputShape;
    model.kernel = 'gaussian';
    model.regularization = 1e-8;
    model.centers = [];
    model.weights = [];
    model.bias = 0.0;
    model.sigma = [];
    model.trained = false;
end

function model = rbfnFit(model, X, y)
    X = double(X);
    y = double(y(:));
    if isempty(X)
        model.centers = [];
        model.weights = [];
        model.bias = 0.0;
        model.sigma = [];
        model.trained = false;
        return;
    end
    model.centers = X;
    model.sigma = computeSigma(X);
    Phi = kernelFunction(model.centers, model.centers, model.sigma).';
    Phi = Phi + model.regularization * eye(size(Phi));
    onesVec = ones(size(Phi, 1), 1);
    designMatrix = [Phi, onesVec];
    solution = designMatrix \ y;
    model.weights = solution(1:end-1);
    model.bias = solution(end);
    model.trained = true;
end

function preds = rbfnPredict(model, X)
    X = double(X);
    if isempty(X)
        preds = zeros(0, 1);
        return;
    end
    if ~model.trained || isempty(model.centers) || isempty(model.weights)
        preds = zeros(size(X, 1), 1);
        return;
    end
    Phi = kernelFunction(model.centers, X, model.sigma).';
    preds = Phi * model.weights + model.bias;
end

function sigma = computeSigma(centers)
    numCenters = size(centers, 1);
    if numCenters <= 1
        sigma = 1.0;
        return;
    end
    distances = zeros(numCenters);
    for i = 1:numCenters
        for j = i+1:numCenters
            d = norm(centers(i, :) - centers(j, :));
            distances(i, j) = d;
            distances(j, i) = d;
        end
    end
    nonZero = distances(distances > 0);
    if isempty(nonZero)
        sigma = 1.0;
    else
        sigma = mean(nonZero) / sqrt(2 * numCenters);
    end
    sigma = max(sigma, 1e-12);
end

function Phi = kernelFunction(centers, points, sigma)
    if isempty(centers) || isempty(points)
        Phi = zeros(size(centers, 1), size(points, 1));
        return;
    end
    diff = permute(centers, [1, 3, 2]) - permute(points, [3, 1, 2]);
    sqNorm = sum(diff .^ 2, 3);
    Phi = exp(-0.5 * sqNorm / (sigma ^ 2));
end
