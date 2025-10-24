classdef SAGIF < ALGORITHM
%SAGIF Surrogate-assisted hypervolume-based environmental selection.
%   This implementation adapts the original standalone SAGIF survivor
%   selection procedure to the class-based interface adopted by PlatEMO
%   3.x.  The algorithm follows a standard generational loop where
%   offspring are created via the default GA operator and an environmental
%   selection based on the SAGIF procedure is applied to truncate the
%   combined population.

%------------------------------- Reference --------------------------------
% R. Cheng, J. Tian, Y. Jin, and X. Zhang, "A surrogate-assisted
% evolutionary algorithm with online transfer learning for expensive
% multiobjective optimization," IEEE Transactions on Evolutionary
% Computation, 2020, 24(4): 577-591.  (The environmental selection module
% is closely aligned with the SAGIF strategy.)
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [samplingFactor, filterFraction, monteCarloSamples, rngSeed] = ...
                Algorithm.ParameterSet(0.035, 0.7, 8000, 0);

            if rngSeed > 0
                rng(rngSeed, 'twister');
            else
                rng('shuffle');
            end

            Population = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Fitness    = rand(numel(Population), 1);
                MatingPool = TournamentSelection(2, Problem.N, Fitness);
                Offspring  = OperatorGA(Problem, Population(MatingPool));
                Union      = [Population, Offspring];

                PopObj = Union.objs;
                PopCon = Union.cons;

                feasible = all(PopCon <= 0, 2);
                if any(feasible)
                    refPoint = max(PopObj(feasible, :), [], 1) + 1;
                    Next = SAGIF.EnvironmentalSelection(PopObj, PopCon, Problem.N, ...
                        refPoint, samplingFactor, filterFraction, monteCarloSamples);
                else
                    % If no feasible solutions exist, resort to the solutions
                    % with the smallest aggregated constraint violation.
                    cv = sum(max(0, PopCon), 2);
                    [~, order] = sort(cv);
                    Next = order(1:Problem.N);
                end

                Population = Union(Next);
            end
        end
    end

    methods(Static, Access = private)
        function Next = EnvironmentalSelection(PopObj, PopCon, N, refPoint, samplingFactor, filterFraction, mcSamples)
            candidates = 1:size(PopObj,1);
            feasible   = candidates(all(PopCon<=0,2));
            if isempty(feasible)
                Next = candidates(1:min(N,numel(candidates)));
                return;
            end

            if numel(feasible) <= N
                Next = feasible;
                if numel(Next) < N
                    infeasible = setdiff(candidates, feasible);
                    cv = sum(max(0, PopCon(infeasible,:)),2);
                    [~, order] = sort(cv);
                    append = infeasible(order(1:min(N-numel(Next), numel(order))));
                    Next   = [Next(:); append(:)];
                end
                return;
            end

            objs = PopObj(feasible,:);
            refPoint = min(refPoint, max(objs,[],1) + 1);

            surrogate = SAGIF.initRBFN(size(objs,2));
            trainingX = zeros(0, size(objs,2));
            trainingY = zeros(0, 1);
            maxTrainingSize = max(5 * N, 200);
            selectedObjs = zeros(0, size(objs,2));
            selected    = zeros(N,1);
            currentHV   = 0;
            remaining   = 1:size(objs,1);
            selectedCnt = 0;

            while selectedCnt < N && ~isempty(remaining)
                candidateObjs = objs(remaining,:);
                if surrogate.trained
                    predictions = SAGIF.rbfnPredict(surrogate, candidateObjs);
                else
                    predictions = prod(max(refPoint - candidateObjs, 1e-12), 2);
                end

                [~, bestPos] = max(predictions);
                chosenLocalIdx = remaining(bestPos);
                chosenObj = objs(chosenLocalIdx, :);

                newSelectedObjs = [selectedObjs; chosenObj];
                hvWithCandidate = SAGIF.computeHV(newSelectedObjs, refPoint, mcSamples);
                contribution = hvWithCandidate - currentHV;
                if contribution <= 0
                    contribution = max(contribution, 1e-12);
                end

                selectedCnt = selectedCnt + 1;
                selected(selectedCnt) = feasible(chosenLocalIdx);
                selectedObjs = newSelectedObjs;
                currentHV = hvWithCandidate;

                remaining(bestPos) = [];
                [trainingX, trainingY] = SAGIF.updateTrainingSet(trainingX, trainingY, chosenObj, contribution, maxTrainingSize);

                if selectedCnt >= N || isempty(remaining)
                    continue;
                end

                filtered = SAGIF.l1Filter(objs, remaining, selectedObjs, filterFraction);
                sampleSize = max(1, round(numel(filtered) * samplingFactor));
                sampleSize = min(sampleSize, numel(filtered));
                sampleIndices = SAGIF.sampleWithoutReplacement(filtered, sampleSize);

                if ~isempty(sampleIndices)
                    sampleObjs = objs(sampleIndices, :);
                    sampleHV = zeros(numel(sampleIndices), 1);
                    for i = 1:numel(sampleIndices)
                        augmented = [selectedObjs; sampleObjs(i, :)];
                        hvValue = SAGIF.computeHV(augmented, refPoint, mcSamples);
                        delta = hvValue - currentHV;
                        if delta <= 0
                            delta = max(delta, 1e-12);
                        end
                        sampleHV(i) = delta;
                    end
                    [trainingX, trainingY] = SAGIF.updateTrainingSet(trainingX, trainingY, sampleObjs, sampleHV, maxTrainingSize);
                end

                if size(trainingX,1) >= 1
                    surrogate = SAGIF.rbfnFit(surrogate, trainingX, trainingY);
                end
            end

            selected = selected(1:selectedCnt);
            if numel(selected) < N
                remainingGlobal = setdiff(feasible, selected);
                selected = [selected(:); remainingGlobal(1:min(N-numel(selected), numel(remainingGlobal)))'];
            end

            if numel(selected) < N
                infeasible = setdiff(candidates, selected);
                cv = sum(max(0, PopCon(infeasible,:)),2);
                [~, order] = sort(cv);
                selected = [selected(:); infeasible(order(1:min(N-numel(selected), numel(order))))'];
            end

            Next = selected(:);
        end

        function hv = computeHV(points, refPoint, sampleCount)
            if isempty(points)
                hv = 0.0;
                return;
            end
            refPoint = refPoint(:)';
            points = min(points, refPoint - 1e-9);
            lower = min(points, [], 1);
            lower = min(lower, refPoint - 1.0);
            lower = min(lower, refPoint - 1e-4);
            span = max(refPoint - lower, 1e-9);

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

        function filtered = l1Filter(objs, remaining, selectedObjs, fraction)
            if isempty(remaining) || isempty(selectedObjs)
                filtered = remaining;
                return;
            end
            subsetSize = max(1, ceil(numel(remaining) * fraction));
            distances = zeros(numel(remaining), 1);
            for idx = 1:numel(remaining)
                diff = abs(selectedObjs - objs(remaining(idx), :));
                distances(idx) = sum(diff(:));
            end
            [~, order] = sort(distances, 'descend');
            order = order(1:subsetSize);
            filtered = remaining(order);
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
            model.sigma = 1.0;
            model.trained = false;
        end

        function model = rbfnFit(model, X, y)
            X = double(X);
            y = double(y(:));
            if isempty(X)
                model.centers = [];
                model.weights = [];
                model.bias = 0.0;
                model.sigma = 1.0;
                model.trained = false;
                return;
            end
            model.centers = X;
            model.sigma = SAGIF.computeSigma(X);
            Phi = SAGIF.kernelFunction(model.centers, model.centers, model.sigma).';
            Phi = Phi + model.regularization * eye(size(Phi, 1));
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
            Phi = SAGIF.kernelFunction(model.centers, X, model.sigma).';
            preds = Phi * model.weights + model.bias;
        end

        function sigma = computeSigma(centers)
            numCenters = size(centers, 1);
            if numCenters <= 1
                sigma = 1.0;
                return;
            end
            sqDist = max(0, bsxfun(@plus, sum(centers.^2,2), sum(centers.^2,2)') - 2*(centers*centers.'));
            distances = sqrt(sqDist);
            distances(1:numCenters+1:end) = Inf;
            finiteVals = distances(~isinf(distances) & distances > 0);
            if isempty(finiteVals)
                sigma = 1.0;
            else
                sigma = mean(finiteVals) / sqrt(2 * numCenters);
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
            Phi = exp(-0.5 * sqNorm / max(sigma ^ 2, 1e-24));
        end
    end
end
