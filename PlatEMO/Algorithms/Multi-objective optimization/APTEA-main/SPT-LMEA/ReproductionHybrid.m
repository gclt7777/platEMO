function Offspring = ReproductionHybrid(Problem,Population,nvg,group)
%  Copyright (C) 2023 Songbai Liu
%  Songbai Liu <songbai@szu.edu.cn>
	
	%% Parameter setting
    [CR,F,proM,disM] = deal(1.0,0.5,2,20);
    
	Lower = Problem.lower;
	Upper = Problem.upper;
    % For each solution in the current population
    for i = 1 : Problem.N
		
		% Choose two different random parents
		p = randperm(Problem.N, 2); % p = randperm(n,k) 返回行向量，其中包含在 1 到 n（包括二者）之间随机选择的 k 个唯一整数
		while p(1)==i || p(2)==i
			p = randperm(Problem.N, 2); 
		end		
       
     	% Generate an child
		Parents = cell(1, 3);
		Parents{1} = Population(i).decs;
		Parents{2} = Population(p(1)).decs;
		Parents{3} = Population(p(2)).decs;
		[N, D] = size(Parents{1});
		child = Parents{1};
		r = rand(1);
		if r(1) < 0.5
			Site = rand(N,D) < CR; %CR = 1.0
			child(Site) = child(Site) + F*(Parents{2}(Site)-Parents{3}(Site));
		else
			normParents = normalizedVariable(Parents,D,Lower,Upper);
			code = encode(normParents,group,nvg);
			newCode = searchInTransferedSpace(code,nvg);
			child = decode(newCode,normParents{1},group,nvg,Lower,Upper);
		end
		%% Polynomial mutation
		Site  = rand(N,D) < proM/D;
		mu    = rand(N,D);
		temp  = Site & mu<=0.5;
		child       = min(max(child,Lower),Upper);
		child(temp) = child(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
					  (1-(child(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
		temp = Site & mu>0.5; 
		child(temp) = child(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
					  (1-(Upper(temp)-child(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
		
		%Evaluation of the new child
		child = Problem.Evaluation(child);
		
		%add the new child to the offspring population
		Offspring(i) = child;
	end                            
end

function normParents = normalizedVariable(Parents,D,Lower,Upper)
	normParents = Parents;
	len = size(Parents,2);
	for i = 1:len
		for j = 1:D
			normParents{i}(j) =	 (Parents{i}(j)-Lower(j))/(Upper(j)-Lower(j));
		end
	end
end

function code = encode(Parents,group,nvg)
	len = size(Parents,2);
	code = cell(1,len);
	for i = 1:len
		for j = 1:nvg
			code{i}(j) = 0;
			gLen = size(group{j},2);
			for k = 1:gLen
				code{i}(j) = code{i}(j)+Parents{i}(group{j}(k));
			end
			code{i}(j) = code{i}(j)/gLen;
		end
	end
end

function newCode = searchInTransferedSpace(code,nvg)
	lr = 0.5;
	for i = 1:nvg
		newCode(i) = code{1}(i) + lr*(code{2}(i)-code{3}(i));
		if newCode(i) < 0
			newCode(i) = 0.000001;
		end
		if newCode(i) > 1
			newCode(i) = 0.999999;
		end
	end
end

function child = decode(newCode,normParent,group,nvg,Lower,Upper)
	for i = 1:nvg
		gLen = size(group{i},2);
		sum = 0;
		for j = 1:gLen
			cVar = group{i}(j);
			sum = sum + normParent(cVar);
		end
		if sum == 0
			sum = 0.000001;
		end
		for j = 1:gLen
			cVar = group{i}(j);
			child(cVar) = (normParent(cVar)/sum)*newCode(i)*gLen*(Upper(cVar)-Lower(cVar)) + Lower(cVar);
			if child(cVar) < Lower(cVar)
				child(cVar) = Lower(cVar) + 0.000001;
			end
			if child(cVar) > Upper(cVar)
				child(cVar) = Upper(cVar) - 0.000001;
			end
		end
	end
end