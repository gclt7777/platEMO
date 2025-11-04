function [PV,DV] = VariableClustering(Problem, Population, nSel, nPer)
% Detect the kind of each decision variable (LMEA-style clustering)
%
% Inputs:
%   Problem    : PlatEMO problem object
%   Population : array of INDIVIDUAL objects
%   nSel       : number of base solutions sampled
%   nPer       : number of perturbations per base solution
%
% Outputs:
%   PV : indices of variables considered diversity-related
%   DV : indices of variables considered convergence-related

    [N, D] = size(Population.decs);

    % Use the first front for normalization
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end
    span = fmax - fmin;
    span(span==0) = 1;

    %% Containers for statistics
    Angle  = zeros(D, nSel);
    RMSE   = zeros(D, nSel);

    %% Choose nSel base solutions
    Sample = randi(N, 1, nSel);

    for i = 1:D
        drawnow();

        % --- Construct perturbations for the i-th decision variable ---
        Decs      = repmat(Population(Sample).decs, nPer, 1);
        Decs(:,i) = unifrnd(Problem.lower(i), Problem.upper(i), size(Decs,1), 1);

        % --- Evaluate via Problem.Evaluation (tracks FE automatically) ---
        newPopu   = Problem.Evaluation(Decs);

        % --- For each base solution, perform PCA-1 fitting to gather stats ---
        for j = 1:nSel
            idx    = j:nSel:size(Decs,1);
            Points = newPopu(idx).objs;

            % Normalize and remove mean
            Pn = (Points - fmin) ./ span;
            Pn = Pn - mean(Pn,1);

            % First principal direction
            [~,~,V] = svd(Pn,'econ');
            v1 = V(:,1)';
            v1 = v1 ./ max(norm(v1),eps);

            % RMSE to the line
            proj  = sum(Pn .* v1, 2);
            resid = Pn - proj .* v1;
            RMSE(i,j) = sqrt(mean(sum(resid.^2,2)));

            % Angle with hyperplane defined by ones vector
            normal = ones(1,size(v1,2));
            sine   = abs(sum(v1.*normal,2)) / (norm(v1)*norm(normal));
            Angle(i,j) = real(asin(min(max(sine,0),1)) / pi * 180);
        end
    end

    %% Determine variable types: filter by RMSE then cluster angles
    VariableKind = (mean(RMSE,2) < 1e-2)';
    result       = kmeans(Angle, 2)';
    if any(result(VariableKind)==1) && any(result(VariableKind)==2)
        m1 = mean(mean(Angle(result==1 & VariableKind, :)));
        m2 = mean(mean(Angle(result==2 & VariableKind, :)));
        if m1 > m2
            VariableKind = VariableKind & (result==1);
        else
            VariableKind = VariableKind & (result==2);
        end
    end

    PV = find(~VariableKind);
    DV = find(VariableKind);
end
