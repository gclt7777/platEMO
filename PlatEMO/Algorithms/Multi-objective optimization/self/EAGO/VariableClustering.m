function [PV,DV] = VariableClustering(varargin)
% Detect the kind of each decision variable
%
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    narginchk(4,4);

    Population = varargin{2};
    nSel       = varargin{3};
    nPer       = varargin{4};

    Problem = locateProblemHandle(varargin{1});

    [N,D] = size(Population.decs);
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end

    %% Calculate the proper values of each decision variable
    Angle  = zeros(D,nSel);
    RMSE   = zeros(D,nSel);
    Sample = randi(N,1,nSel);
    for i = 1 : D
        drawnow();
        % Generate several random solutions by perturbing the i-th dimension
        if Problem.maxFE - Problem.FE < nSel*nPer
            break;
        end
        Decs      = repmat(Population(Sample).decs,nPer,1);
        Decs(:,i) = unifrnd(Problem.lower(i),Problem.upper(i),size(Decs,1),1);
        newPopu   = Problem.Evaluation(Decs);
        if length(newPopu) < nSel*nPer
            break;
        end
        for j = 1 : nSel
            % Normalize the objective values of the current perturbed solutions
            Points = newPopu(j:nSel:end).objs;
            Points = (Points-repmat(fmin,size(Points,1),1))./repmat(fmax-fmin,size(Points,1),1);
            Points = Points - repmat(mean(Points,1),nPer,1);
            % Calculate the direction vector of the determining line
            [~,~,V] = svd(Points);
            Vector  = V(:,1)'./norm(V(:,1)');
            % Calculate the root mean square error
            error = zeros(1,nPer);
            for k = 1 : nPer
                error(k) = norm(Points(k,:)-sum(Points(k,:).*Vector)*Vector);
            end
            RMSE(i,j) = sqrt(sum(error.^2));
            % Calculate the angle between the line and the hyperplane
            normal     = ones(1,size(Vector,2));
            sine       = abs(sum(Vector.*normal,2))./norm(Vector)./norm(normal);
            Angle(i,j) = real(asin(sine)/pi*180);
        end
    end

    %% Detect the kind of each decision variable
    VariableKind = (mean(RMSE,2)<1e-2)';
    result       = kmeans(Angle,2)';
    if any(result(VariableKind)==1) && any(result(VariableKind)==2)
        if mean(mean(Angle(result==1&VariableKind,:))) > mean(mean(Angle(result==2&VariableKind,:)))
            VariableKind = VariableKind & result==1;
        else
            VariableKind = VariableKind & result==2;
        end
    end
    PV = find(~VariableKind);
    DV = find(VariableKind);
end

function Problem = locateProblemHandle(arg1)
% Locate the PROBLEM handle irrespective of whether a Global wrapper is used

    if isa(arg1,'PROBLEM')
        Problem = arg1;
        return;
    end

    if isobject(arg1) && isprop(arg1,'problem')
        Problem = arg1.problem;
        if isa(Problem,'PROBLEM')
            return;
        end
    end

    if isstruct(arg1) && isfield(arg1,'problem')
        Problem = arg1.problem;
        if isa(Problem,'PROBLEM')
            return;
        end
    end

    error('VariableClustering:InvalidInput', ...
          ['First argument must be a PROBLEM instance or a wrapper that ', ...
           'stores the problem handle in the field/property ''problem''.']);
end
