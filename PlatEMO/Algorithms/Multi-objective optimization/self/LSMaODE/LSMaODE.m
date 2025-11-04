classdef LSMaODE < ALGORITHM
% <algorithm> <L>
% LSMaODE implements the multipopulation-based differential evolution
% for large-scale many-objective optimization proposed by Zhang et al.

%------------------------------- Reference --------------------------------
% K. Zhang, C. Shen and G. G. Yen,
% "Multipopulation-Based Differential Evolution for Large-Scale
%  Many-Objective Optimization," IEEE Transactions on Cybernetics, 2022,
%  doi: 10.1109/TCYB.2022.3178929.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [mutationStrength,proportion] = Algorithm.ParameterSet(10,0.1);

            %% Generate random population
            Population = Problem.Initialization();
            if isempty(Population)
                return;
            end

            lower   = Problem.lower;
            upper   = Problem.upper;
            popSize = numel(Population);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                rawGroup = floor(proportion*popSize);
                if rawGroup <= 0
                    if proportion > 0 && popSize > 0
                        localgroup = 1;
                    else
                        localgroup = 0;
                    end
                else
                    localgroup = min(popSize,rawGroup);
                end
                if localgroup > 0
                    localPopulation = Population(1:localgroup);
                else
                    localPopulation = Population([]);
                end

                for ~ = 1 : 20
                    L1list = [];
                    for i = 1 : popSize
                        Parent = Population(i);
                        if i <= localgroup
                            for k = 1 : Problem.D
                                OffspringDec = Mutation(localgroup,i,k,Parent,lower,upper,mutationStrength,localPopulation,[]);
                                Offspring    = Problem.Evaluation(OffspringDec);
                                [Population(i),localPopulation] = UpdateIndividual(Parent,Offspring,Population,localPopulation,i,true);
                            end
                        else
                            if isempty(L1list)
                                FrontNo = NDSort(Population.objs,Population.cons,popSize);
                                L1list  = find(FrontNo == 1);
                            end
                            OffspringDec = Mutation(localgroup,i,[],Parent,lower,upper,mutationStrength,Population,L1list);
                            Offspring    = Problem.Evaluation(OffspringDec);
                            [Population(i),localPopulation] = UpdateIndividual(Parent,Offspring,Population,localPopulation,i,false);
                        end
                    end
                    if localgroup > 0
                        localPopulation = Population(1:localgroup);
                    else
                        localPopulation = Population([]);
                    end
                end
            end
        end
    end
end

function [Selected,localPopulation] = UpdateIndividual(Parent,Offspring,Population,localPopulation,index,isLocal)
% Apply survival selection between the parent and the offspring.

    pair = [Parent,Offspring];
    [FrontNo,MaxFNo] = NDSort(pair.objs,pair.cons,2);

    if MaxFNo ~= 1
        if FrontNo(2) == 1
            Selected = Offspring;
        else
            Selected = Parent;
        end
        if isLocal && index <= length(localPopulation)
            localPopulation(index) = Selected;
        end
        return;
    end

    if isLocal
        referencePop = localPopulation;
    else
        referencePop = Population;
    end
    if ~isempty(referencePop) && index <= length(referencePop)
        referencePop(index) = [];
    else
        referencePop = referencePop([]);
    end

    if isempty(referencePop)
        refObjs = [];
    else
        refObjs = referencePop.objs;
    end

    off_dom_count = CalDomcount(refObjs,Offspring.objs);
    par_dom_count = CalDomcount(refObjs,Parent.objs);

    if off_dom_count < par_dom_count
        Selected = Offspring;
    elseif off_dom_count > par_dom_count
        Selected = Parent;
    else
        if isempty(referencePop)
            Par_MED = 0;
            off_MED = 0;
        else
            [Par_MED,off_MED] = MED(referencePop,Parent,Offspring);
        end
        if off_MED > Par_MED
            Selected = Offspring;
        else
            Selected = Parent;
        end
    end

    if isLocal && index <= length(localPopulation)
        localPopulation(index) = Selected;
    end
end
