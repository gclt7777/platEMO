function [PV,DV] = VariableClustering_A(varargin)
% Detect the kind of each decision variable (variant used by EAGOA)
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

    [PV,DV] = EAGOA.VariableClustering(Problem,Population,nSel,nPer);
end

function Problem = locateProblemHandle(arg1)
% Locate the PROBLEM handle irrespective of whether a Global wrapper is used

    if isa(arg1,'PROBLEM')
        Problem = arg1;
        return;
    end

    if isobject(arg1)
        if isprop(arg1,'problem')
            Problem = arg1.problem;
            if isa(Problem,'PROBLEM')
                return;
            end
        end
        if isprop(arg1,'Problem')
            Problem = arg1.Problem;
            if isa(Problem,'PROBLEM')
                return;
            end
        end
        if isprop(arg1,'pro')
            Problem = arg1.pro;
            if isa(Problem,'PROBLEM')
                return;
            end
        end
    end

    if isstruct(arg1)
        if isfield(arg1,'problem')
            Problem = arg1.problem;
            if isa(Problem,'PROBLEM')
                return;
            end
        end
        if isfield(arg1,'Problem')
            Problem = arg1.Problem;
            if isa(Problem,'PROBLEM')
                return;
            end
        end
        if isfield(arg1,'pro')
            Problem = arg1.pro;
            if isa(Problem,'PROBLEM')
                return;
            end
        end
    end

    error('VariableClustering_A:InvalidInput', ...
          ['First argument must be a PROBLEM instance or a wrapper that ', ...
           'stores the problem handle in the field/property ''problem''.']);
end
