classdef SPTEA < ALGORITHM
% <multi> <real/integer> <large/none>
% evolutionary algorithm via self-guided problem transformation
% tec --- 1 --- type of environmental selection. 1 = NSGA-II (Default), 2 = NSGA-III, 3 = TDEA, 4 = MOEAC
% nvg --- 2 --- number of variable groups. Default = 2
% tvg --- 1 --- type of variable grouping. 1 = random (Default), 2 = linear, 3 = ordered, 4 = contribution-based grouping

%------------------------------- Reference --------------------------------
% .
%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [tec,nvg,tvg] = Algorithm.ParameterSet(1,2,1);
			
			%% variable grouping
			group = VariableGrouping(Problem.D,nvg,tvg);

            %% Optimization-Using the environmental selection proposed in NSGA-II
            if tec==1
                % Generate random population
                Population = Problem.Initialization();
                % Optimization
                while Algorithm.NotTerminated(Population)
                    %% SPT-based reproduction
                    if mod(Problem.FE,100*Problem.N) == 0
                        %if nvg < Problem.D/2
                        if nvg > 2
                            nvg = nvg-1;
                        end
                        group = VariableGrouping(Problem.D,nvg,tvg);
                    end
					Offspring  = ReproductionSPTDE(Problem,Population,nvg,group);
                    Population = EnvironmentalSelection_NSGAII([Population,Offspring],Problem.N);
                end
            end
			
			%% Optimization-Using the environmental selection proposed in NSGA-III
            if tec==2
                % Generate the reference points and random population
                [Z,Problem.N] = UniformPoint(Problem.N,Problem.M);
                Population    = Problem.Initialization();
                Zmin          = min(Population(all(Population.cons<=0,2)).objs,[],1);
                % Optimization
                while Algorithm.NotTerminated(Population)
                    if mod(Problem.FE,100*Problem.N) == 0 
                        if nvg > 2
                            nvg = nvg-1;
                        %if nvg < Problem.D/2
                            %nvg = nvg+1;
                        end
                        group = VariableGrouping(Problem.D,nvg,tvg);
                    end
                    %Offspring  = ReproductionDE(Problem,Population);
                    Offspring  = ReproductionSPTDE(Problem,Population,nvg,group);
                    Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
                    Population = EnvironmentalSelection_NSGAIII([Population,Offspring],Problem.N,Z,Zmin);
                end
            end
			
			%% Optimization-Using the environmental selection proposed in TDEA
            if tec==3
                %% Generate the reference points and random population
                [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
                Population    = Problem.Initialization();
                [z,znad]      = deal(min(Population.objs),max(Population.objs));
				%% Optimization
				while Algorithm.NotTerminated(Population)
					%MatingPool = randi(length(Population),1,Problem.N);
                    if mod(Problem.FE,100*Problem.N) == 0
                        %if nvg > 2
                            %nvg = nvg-1;
                        if nvg < Problem.D/2
                            nvg = nvg+1;
                        end
                        group = VariableGrouping(Problem.D,nvg,tvg);
                    end
					Offspring  = ReproductionHybrid(Problem,Population,nvg,group);
					[Population,z,znad] = EnvironmentalSelection_TDEA([Population,Offspring],W,Problem.N,z,znad);
				end
            end
			
            %% Optimization-Using the environmental selection proposed in MOEAC
            if tec==4
                %% Generate the reference points and random population
                [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
                Population    = Problem.Initialization();
                [z,znad]      = deal(min(Population.objs),max(Population.objs));
				%% Optimization
				while Algorithm.NotTerminated(Population)
					%MatingPool = randi(length(Population),1,Problem.N);
                    if mod(Problem.FE,100*Problem.N) == 0
                        %if nvg > 2
                            %nvg = nvg-1;
                        if nvg < Problem.D/2
                            nvg = nvg+1;
                        end
                        group = VariableGrouping(Problem.D,nvg,tvg);
                    end
					Offspring  = ReproductionHybrid(Problem,Population,nvg,group);
					[Population,z,znad] = EnvironmentalSelection_MOEAC([Population,Offspring],Problem.N,z,znad,Problem.N);
				end
            end

        end
    end
end