function EAGO(Global)
% <algorithm> <L>


%------------------------------- Reference --------------------------------
% Q. Deng, Q. Kang, L. Zhang, M. C. Zhou and J. An, Objective Space-based 
% Population Generation to Accelerate Evolutionary Algorithms for Large-scale
% Many-objective Optimization, IEEE Transactions on Evolutionary Computation, 
% vol. 27, no. 2, pp. 326-340, Apr. 2023.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Parameter setting
    [nSel,nPer, c_fmin, c_fmax] = Global.ParameterSet(5,30,0.01,10);
    %[nSel,nPer, c_fmin, c_fmax] = Global.ParameterSet(10,50,1,1);
    %%[nSel,nPer,nCor,type] = Global.ParameterSet(5,30,5,1);
    %% Generate random population
	Population = Global.Initialization();
	Population_distribute = Population;
	objs = Population.objs;
	x1 = Population.decs;
	R = objs\x1;
	R1 = R;
	
	[Div_V,con_V] = VariableClustering(Global,Population,nSel,nPer);


	[Div_V1,con_V1,con_V_plus] = VariableAnalysis1(Global,Population,nSel,nPer, R1);
 %   pasue();
    first = true;
	N = Global.N;
	change = true;
    while Global.NotTermination(Population)
    
    
        % Convergence optimization
        if Global.evaluated / Global.evaluation < 0.9
			temp = 10;
		else
			temp = 1;
		end
        
        
 
        if change 

            
            for j = 1 : temp
				for ij = 1 : length(con_V)
					drawnow();
					Population = ConvergenceOptimization_R(Population,con_V(ij), R,Global);
				end
			end
            
			
            for j = 1 : temp			
				for i = 1 : 1
					drawnow();
					Population = DistributionOptimization(Population,Div_V, R,Global);			   
				end

			end
           
			change = false;
		
		else
        
				if rank(Population.objs) == Global.M
					objs = Population.objs;
					x1 = Population.decs;
					R2 = objs\x1;
					[Div_V2,con_V1,con_V_plus] = VariableAnalysis2(Global,Population,nSel,nPer, R1);
					if length(Div_V2) <= length(Div_V1)
						Div_V1 = Div_V2;
						R1 = R2;
					end
				else
					[Div_V2,con_V1,con_V_plus] = VariableAnalysis2(Global,Population,nSel,nPer, R1);
					if length(Div_V2) <= length(Div_V1)
						Div_V1 = Div_V2;
					end
				end

			%con_V1 = intersect(con_V1,con_V); if Global.problem = SDTLZ;
			for j = 1 : 1
				for i = 1 : 35
					drawnow();                    
					Population = ConvergenceOptimization_1(Population,con_V1, R1,Global);
				end
				for i = 1 : 35
					drawnow();
					Population = ConvergenceOptimization_2(Population,con_V_plus, R1,Global);
				end
				for i = 1 : 10
					drawnow();
					Population = DistributionOptimization2(Population,Div_V1, R1,Global);
				end
			end
		
			change = true;
		end

    end

end

function Population = ConvergenceOptimization_R(Population,con_V, R,Global)
    [N,D] = size(Population.decs);

	OffDec = Population.decs;
   
	NewObjs = GAhalf3(Population(randperm(N)).objs, N);

	Offspring_Convergence = NewObjs * R(:, con_V);
	
	NewObjs2 = GAhalf2(Population(randperm(N)).objs, N);

	Offspring_Convergence2 = NewObjs2 * R(:, con_V);
   
	OffDec(:,con_V) = OffDec(randperm(N),con_V) + (Offspring_Convergence - Offspring_Convergence2)*0.5;
	OffDec(:,con_V) = min(max(OffDec(:,con_V),repmat(Global.lower(con_V),N,1)),repmat(Global.upper(con_V),N,1));
		
	Offspring          = INDIVIDUAL(OffDec);
	

	
	

	allCon  = calCon([Population.objs;Offspring.objs]);
	Con     = allCon(1:N);
	newCon  = allCon(N+1:end);
	updated = Con > newCon;
	Population(updated) = Offspring(updated);
	Con(updated)        = newCon(updated);

end


function Population2 = DistributionOptimization(Population,Div_V, R,Global)
    
	N            = length(Population);
	OffDec       = Population(TournamentSelection(2,N,calCon(Population.objs))).decs;
	%OffDec       =  Population.decs;
	

    NewObjs = GAhalf3(Population.objs, N);

	Offspring_Convergence = NewObjs * R(:, Div_V);
	
	NewObjs2 = GAhalf2(Population.objs, N);

	Offspring_Convergence2 = NewObjs2 * R(:, Div_V);
    
    
   %NewDec = OffDec(:,Div_V);
   %Offspring_Convergence(Offspring_Convergence < repmat(Global.lower(Div_V), N, 1)) = NewDec(Offspring_Convergence < repmat(Global.lower(Div_V), N, 1));
   %Offspring_Convergence(Offspring_Convergence > repmat(Global.upper(Div_V), N, 1)) = NewDec(Offspring_Convergence > repmat(Global.upper(Div_V), N, 1));
   

   %OffDec(:,Div_V) = sqrt(OffDec(:,Div_V).*unifrnd(repmat(Global.lower(Div_V),size(OffDec,1),1),repmat(Global.upper(Div_V),size(OffDec,1),1)));
	temp = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
	temp2 = OffDec(:,Div_V);
	a = randperm(N);
	temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
	OffDec(:,Div_V) = (temp) + (Offspring_Convergence-Offspring_Convergence2)*(1/2);
    OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Global.lower(Div_V),N,1)),repmat(Global.upper(Div_V),N,1));


	
	%OffDec(:,Div_V) = OffDec(randperm(N),Div_V) + Offspring_Convergence;
    %OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Global.lower(Div_V),N,1)),repmat(Global.upper(Div_V),N,1));


    Offspring    = INDIVIDUAL(OffDec);
    Population2   = EnvironmentalSelection([Population,Offspring],N);

end



function [Div_V,con_V, con_V_plus] = VariableAnalysis2(Global,Population,nSel,nPer,R)

	
	
	VariableKind = false(1,Global.D);
    VariableKind_plus = false(1,Global.D);
    for i = 1 : Global.D
        drawnow();
        
		Sample = randi(Global.N,1,nSel);
		result = zeros(1,nSel);
		for j = 1 : nSel
            
			Decs      = repmat(Population(Sample(j)).decs,nPer,1);
			Decs(:,i) = unifrnd(Global.lower(i),Global.upper(i),size(Decs,1),1);
			%newPopu_random   = INDIVIDUAL(Decs);
			newPopu_random = Global.problem.CalObj(Decs);
			Global.evaluated = Global.evaluated + size(newPopu_random, 1);
			newPopu_random_average = sum(newPopu_random, 1)/nPer;

			
			NewObjs = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Global.M))*0.25;
			Offspring = NewObjs * R(:, i);
			Decs(:,i) = min(max(Offspring,repmat(Global.lower(i),nPer,1)),repmat(Global.upper(i),nPer,1));
			%newPopu_reflex   = INDIVIDUAL(Decs);
			newPopu_reflex = Global.problem.CalObj(Decs);
			Global.evaluated = Global.evaluated + size(newPopu_reflex, 1);
			newPopu_reflex_average = sum(newPopu_reflex, 1)/nPer;
			
			
			NewObjs_plus = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Global.M))*0.25;
			Offspring_plus = NewObjs_plus * R(:, i);
			Decs(:,i) = min(max(Offspring_plus,repmat(Global.lower(i),nPer,1)),repmat(Global.upper(i),nPer,1));
			%newPopu_reflex   = INDIVIDUAL(Decs);
			newPopu_reflex_plus = Global.problem.CalObj(Decs);
			Global.evaluated = Global.evaluated + size(newPopu_reflex_plus, 1);
			newPopu_reflex_plus_average = sum(newPopu_reflex_plus, 1)/nPer;
			
			
			
			
			
			
			if(sum(newPopu_reflex_average.*newPopu_reflex_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average))
				result(j) = 1;
			elseif(sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average)*0.9 <= sum(newPopu_random_average.*newPopu_random_average) && sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) < sum(newPopu_reflex_average.*newPopu_reflex_average))
				result(j) = 2;			
			end

        end
		
		if sum(result == 1) >= sum(result == 2) && sum(result == 1) >= sum(result == 0)
			VariableKind(i) = true;
		elseif (sum(result == 2) > sum(result == 1) && sum(result == 2) >= sum(result == 0))
			VariableKind_plus(i) = true;
		end
    end
    

    Div_V = find(~(VariableKind | VariableKind_plus));
    con_V = find(VariableKind);
	con_V_plus =find(VariableKind_plus);
end

function [Div_V,con_V, con_V_plus] = VariableAnalysis1(Global,Population,nSel,nPer,R)


	
	VariableKind = false(1,Global.D);
    VariableKind_plus = false(1,Global.D);
    for i = 1 : Global.D
        drawnow();

		Sample = randi(Global.N,1,nSel);
		result = zeros(1,nSel);
		for j = 1 : nSel
            
			Decs      = repmat(Population(Sample(j)).decs,nPer,1);
			Decs(:,i) = unifrnd(Global.lower(i),Global.upper(i),size(Decs,1),1);
			%newPopu_random   = INDIVIDUAL(Decs);
			newPopu_random = Global.problem.CalObj(Decs);
			Global.evaluated = Global.evaluated + size(newPopu_random, 1);
			newPopu_random_average = sum(newPopu_random, 1)/nPer;

			
			NewObjs = repmat(Population(Sample(j)).objs,nPer,1) - repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Global.M))*0.25;
			Offspring = NewObjs * R(:, i);
			Decs(:,i) = min(max(Offspring,repmat(Global.lower(i),nPer,1)),repmat(Global.upper(i),nPer,1));
			%newPopu_reflex   = INDIVIDUAL(Decs);
			newPopu_reflex = Global.problem.CalObj(Decs);
			Global.evaluated = Global.evaluated + size(newPopu_reflex, 1);
			newPopu_reflex_average = sum(newPopu_reflex, 1)/nPer;
			
			
			NewObjs_plus = repmat(Population(Sample(j)).objs,nPer,1) + repmat(Population(Sample(j)).objs,nPer,1).*(rand(nPer, Global.M))*0.25;
			Offspring_plus = NewObjs_plus * R(:, i);
			Decs(:,i) = min(max(Offspring_plus,repmat(Global.lower(i),nPer,1)),repmat(Global.upper(i),nPer,1));
			%newPopu_reflex   = INDIVIDUAL(Decs);
			newPopu_reflex_plus = Global.problem.CalObj(Decs);
			Global.evaluated = Global.evaluated + size(newPopu_reflex_plus, 1);
			newPopu_reflex_plus_average = sum(newPopu_reflex_plus, 1)/nPer;
			
			

			if(sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_random_average.*newPopu_random_average) && sum(newPopu_reflex_average.*newPopu_reflex_average) <= sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average))
				result(j) = 1;
			elseif(sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) <= sum(newPopu_random_average.*newPopu_random_average) && sum(newPopu_reflex_plus_average.*newPopu_reflex_plus_average) < sum(newPopu_reflex_average.*newPopu_reflex_average))
				result(j) = 2;			
			end

        end
		
		if sum(result == 1) >= sum(result == 2) && sum(result == 1) >= sum(result == 0)
			VariableKind(i) = true;
		elseif (sum(result == 2) > sum(result == 1) && sum(result == 2) >= sum(result == 0))
			VariableKind_plus(i) = true;
		end
    end
    

    Div_V = find(~(VariableKind | VariableKind_plus));
    con_V = find(VariableKind);
	con_V_plus =find(VariableKind_plus);
end



function Population = ConvergenceOptimization_1(Population,con_V, R,Global)
    [N,D] = size(Population.decs);
    Con   = calCon(Population.objs);


	OffDec = Population.decs;

	
	NewObjs = GAhalf2(Population.objs, N);
	Offspring_Convergence = NewObjs * R(:, con_V);					
	NewDec = min(max(Offspring_Convergence,repmat(Global.lower(con_V),N,1)),repmat(Global.upper(con_V),N,1));
						
	OffDec(:,con_V) = (NewDec + OffDec(:,con_V))/2;

	Offspring          = INDIVIDUAL(OffDec);
	
	
	% Update each solution
	allCon  = calCon([Population.objs;Offspring.objs]);
	Con     = allCon(1:N);
	newCon  = allCon(N+1:end);
	updated = Con > newCon;
	Population(updated) = Offspring(updated);
	Con(updated)        = newCon(updated);

end

function Population = ConvergenceOptimization_2(Population,con_V_plus, R,Global)
    [N,D] = size(Population.decs);
    Con   = calCon(Population.objs);


	OffDec = Population.decs;
				
	NewObjs = GAhalf1(Population.objs, N);
	Offspring_Convergence = NewObjs * R(:, con_V_plus);					
	NewDec = min(max(Offspring_Convergence,repmat(Global.lower(con_V_plus),N,1)),repmat(Global.upper(con_V_plus),N,1));
					
	
	OffDec(:,con_V_plus) = (NewDec + OffDec(:,con_V_plus))/2;
	
		
	
	Offspring          = INDIVIDUAL(OffDec);
	

	
	% Update each solution
	allCon  = calCon([Population.objs;Offspring.objs]);
	Con     = allCon(1:N);
	newCon  = allCon(N+1:end);
	updated = Con > newCon;
	Population(updated) = Offspring(updated);
	Con(updated)        = newCon(updated);

end

function Population = DistributionOptimization2(Population,Div_V, R,Global)
% Distribution optimization
	N            = length(Population);
	OffDec       = Population(TournamentSelection(2,N,calCon(Population.objs))).decs;
	NewObjs = GAhalf1(Population.objs, N);

	Offspring_Convergence = NewObjs * R(:, Div_V);
	
	NewObjs2 = GAhalf2(Population.objs, N);

	Offspring_Convergence2 = NewObjs2 * R(:, Div_V);
	
	%OffDec(:,Div_V) = unifrnd(repmat(Global.lower(Div_V),size(OffDec,1),1),repmat(Global.upper(Div_V),size(OffDec,1),1));
	temp = sqrt(OffDec(randperm(N),Div_V).*OffDec(randperm(N),Div_V));
	temp2 = OffDec(:,Div_V);
	a = randperm(N);
	temp(temp>OffDec(a,Div_V)) = temp2(temp>OffDec(a,Div_V));
	OffDec(:,Div_V) = (temp) + (Offspring_Convergence-Offspring_Convergence2)*(1/2);
    OffDec(:,Div_V) = min(max(OffDec(:,Div_V),repmat(Global.lower(Div_V),N,1)),repmat(Global.upper(Div_V),N,1));


	Offspring    = INDIVIDUAL(OffDec);
	Population   = EnvironmentalSelection([Population,Offspring],N);
	
end






function Offspring = GAhalf1(Parent, N)
	
	Offspring = Parent + Parent.*(rand(size(Parent)))*0.25;
end

function Offspring = GAhalf2(Parent, N)
	
	Offspring = Parent - Parent.*(rand(size(Parent)))*0.25;
end

function Offspring = GAhalf3(Parent,N1)
    % 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 2 1.5 0.75 0.25
    c_fmin = 0.01;
    % 1 5 10 15 20 25 30 35 40 45 50 55 60
    c_fmax = 10;
    [proC,disC,proM,disM] = deal(1,20,1,20);
	
	
	FrontNo    = NDSort(Parent,1) == 1;
	%Populationtemp = Parent(FrontNo,:);
	lower = min(Parent, [], 1)*c_fmin;
	upper = max(Parent, [], 1)*c_fmax;
	

    Parent1 = Parent(1:floor(end/2),:);
    Parent2 = Parent(floor(end/2)+1:floor(end/2)*2,:);
    [N,D]   = size(Parent1);
    Global  = GLOBAL.GetObj();
 

	%% Genetic operators for real encoding
	% Simulated binary crossover
	beta = zeros(N,D);
	mu   = rand(N,D);
	beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
	beta(mu>0.5)  = (2-2*mu(mu>0.5)).^(-1/(disC+1));
	beta = beta.*(-1).^randi([0,1],N,D);
	beta(rand(N,D)<0.5) = 1;
	beta(repmat(rand(N,1)>proC,1,D)) = 1;
	Offspring = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
				 (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];
	% Polynomial mutation
	
	Lower = repmat(lower,2*N,1);
    Upper = repmat(upper,2*N,1);
	Site  = rand(2*N,D) < proM/D;
	mu    = rand(2*N,D);
	temp  = Site & mu<=0.5;
	Offspring       = min(max(Offspring,Lower),Upper);
	Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
					  (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
	temp = Site & mu>0.5; 
	Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
					  (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));


end


function Offspring = GAhalf4(Parent, N)
	
	Offspring = Parent - Parent.*(rand(size(Parent))-0.5);
end


function Con = calCon(PopuObj)
% Calculate the convergence of each solution

    FrontNo = NDSort(PopuObj,inf);
    Con     = sum(PopuObj,2);
    Con     = FrontNo'*(max(Con)-min(Con)) + Con;
end



