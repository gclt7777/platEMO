classdef LSMaODE < ALGORITHM
% <2022> <many> <real/integer> <large>
% Large-scale many-objective differential evolution
% mutationStrength --- 10  --- Step size controller in Gaussian mutation
% proportion       --- 0.1 --- Proportion of solutions assigned to the local subpopulation

%------------------------------- Reference --------------------------------
% K. Zhang, C. Shen and G. G. Yen,
% "Multipopulation-Based Differential Evolution for Large-Scale Many-Objective Optimization,"
%  IEEE Transactions on Cybernetics, 2022, doi: 10.1109/TCYB.2022.3178929.
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [mutationStrength,proportion] = Algorithm.ParameterSet(10,0.1);
            if isempty(mutationStrength)
                mutationStrength = 10;
            end
            if isempty(proportion)
                proportion = 0.1;
            end

            %% Generate random population
            Population = Problem.Initialization();
            lower      = Problem.lower;
            upper      = Problem.upper;
            %% Use the current evaluation budget from the platform
            %  Do not cache it too early in case the GUI updates maxFE just
            %  before execution. The inner loops will respect this budget.

            %% Optimization
            while Algorithm.NotTerminated(Population)
                maxBudget  = Problem.maxFE;
                stopSearch = false;
                localgroup      = floor(proportion*Problem.N);
                localPopulation = Population(1:localgroup);
                L1list          = [];
                for iter = 1 : 20 %#ok<NASGU>
                    if stopSearch
                        break;
                    end
                    for i = 1 : Problem.N
                        stopSearch = Problem.FE >= maxBudget;
                        if stopSearch
                            break;
                        end
                        if i <= localgroup
                            for k = 1 : Problem.D
                                stopSearch = Problem.FE >= maxBudget;
                                if stopSearch
                                    break;
                                end
                                Parent         = Population(i);
                                Offspring_dec  = Mutation(localgroup,i,k,Parent,lower,upper,mutationStrength,localPopulation,[]);
                                Offspring      = Problem.Evaluation(Offspring_dec);
                                mat_Population = [Parent,Offspring];
                                [FrontNo,MaxFNo] = NDSort(mat_Population.objs,mat_Population.cons,2);
                                if MaxFNo ~= 1
                                    Population(i) = mat_Population(FrontNo==1);
                                else
                                    temp_Population    = localPopulation;
                                    temp_Population(i) = [];
                                    off_dom_count = CalDomcount(temp_Population.objs,Offspring.objs);
                                    Par_dom_count = CalDomcount(temp_Population.objs,Parent.objs);
                                    if off_dom_count < Par_dom_count
                                        Population(i) = Offspring;
                                    elseif off_dom_count == Par_dom_count
                                        temp_localPopulation    = localPopulation;
                                        temp_localPopulation(i) = [];
                                        [Par_MED,off_MED] = MED(temp_localPopulation,Parent,Offspring);
                                        if off_MED > Par_MED
                                            Population(i) = Offspring;
                                        end
                                    end
                                end
                            end
                        end

                        if i == localgroup+1
                            [GFrontNo,~] = NDSort(Population.objs,Population.cons,Problem.N);
                            List   = 1 : Problem.N;
                            L1list = List(GFrontNo==1);
                        end

                        if i > localgroup
                            stopSearch = Problem.FE >= maxBudget;
                            if stopSearch
                                break;
                            end
                            Parent         = Population(i);
                            Offspring_dec  = Mutation(localgroup,i,[],Parent,lower,upper,mutationStrength,Population,L1list);
                            Offspring      = Problem.Evaluation(Offspring_dec);
                            mat_Population = [Parent,Offspring];
                            [FrontNo,MaxFNo] = NDSort(mat_Population.objs,mat_Population.cons,2);
                            if MaxFNo ~= 1
                                Population(i) = mat_Population(FrontNo==1);
                            else
                                temp_Population    = Population;
                                temp_Population(i) = [];
                                off_dom_count = CalDomcount(temp_Population.objs,Offspring.objs);
                                Par_dom_count = CalDomcount(temp_Population.objs,Parent.objs);
                                if off_dom_count < Par_dom_count
                                    Population(i) = Offspring;
                                elseif off_dom_count == Par_dom_count
                                    [Par_MED,off_MED] = MED(temp_Population,Parent,Offspring);
                                    if off_MED > Par_MED
                                        Population(i) = Offspring;
                                    end
                                end
                            end
                        end
                    end
                    if terminate
                        break;
                    end
                end
                if terminate
                    break;
                end
                if stopSearch
                    break;
                end
            end
        end
    end
end
