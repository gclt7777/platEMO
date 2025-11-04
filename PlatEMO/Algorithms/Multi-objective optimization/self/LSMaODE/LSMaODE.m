function LSMaODE(Global)
% <algorithm> <L>


%------------------------------- Reference --------------------------------
% K. Zhang, C. Shen and G. G. Yen, 
% "Multipopulation-Based Differential Evolution for Large-Scale Many-Objective Optimization," 
%  IEEE Transactions on Cybernetics, 2022, doi: 10.1109/TCYB.2022.3178929.
%--------------------------------------------------------------------------


		mutationStrength=10;proportion=0.1;
        %% Generate random population
        Population = Global.Initialization();
        lower=Global.lower;
        upper=Global.upper;
        %% Optimization
        while Global.NotTermination(Population)
            localgroup=floor(proportion*Global.N);
            localPopulation=Population(1:localgroup);
            for jjjii = 1:20
            for i = 1 : Global.N
                if i<= localgroup
                    for k=1:Global.D
                        %%%依次按照维度变异
                        Parent=Population(i);
                        Offspring_dec=Mutation(localgroup,i,k,Parent,lower,upper,mutationStrength,localPopulation,[]);
                        Offspring=INDIVIDUAL(Offspring_dec);
                        mat_Population=[Parent,Offspring];
                        [FrontNo,MaxFNo] = NDSort(mat_Population.objs,mat_Population.cons,2);
                        if MaxFNo~=1
                            Population(i)=mat_Population(FrontNo==1);
                        else
                            temp_Population=localPopulation;%%Population
                            temp_Population(i)=[];
                            off_dom_count=CalDomcount(temp_Population.objs,Offspring.objs);
                            Par_dom_count=CalDomcount(temp_Population.objs,Parent.objs);
                            if off_dom_count<Par_dom_count
                                Population(i)=Offspring;
                            elseif off_dom_count==Par_dom_count
                                temp_localPopulation=localPopulation;
                                temp_localPopulation(i)=[];
                                [Par_MED,off_MED]=MED(temp_localPopulation,Parent,Offspring);
                                
                                if off_MED>Par_MED
                                    Population(i)=Offspring;
                                end
                            end
                        end
                    end
                end
                %%%%%%%非支配排序
                if i ==localgroup+1
                    [GFrontNo,~] = NDSort(Population.objs,Population.cons,Global.N);
                    List=1:Global.N;
                    L1list=List(GFrontNo==1);
                end
                %%%%%后90%的个体
                if i >localgroup
                    Parent=Population(i);
                    Offspring_dec=Mutation(localgroup,i,[],Parent,lower,upper,mutationStrength,Population,L1list);
                    Offspring=INDIVIDUAL(Offspring_dec);
                    mat_Population=[Parent,Offspring];
                    [FrontNo,MaxFNo] = NDSort(mat_Population.objs,mat_Population.cons,2);
                    if MaxFNo~=1
                        Population(i)=mat_Population(FrontNo==1);
                    else
                        temp_Population=Population;
                        temp_Population(i)=[];
                        off_dom_count=CalDomcount(temp_Population.objs,Offspring.objs);
                        Par_dom_count=CalDomcount(temp_Population.objs,Parent.objs);
                        if off_dom_count<Par_dom_count
                            Population(i)=Offspring;
                        elseif off_dom_count==Par_dom_count
                            [Par_MED,off_MED]=MED(temp_Population,Parent,Offspring);
                            if off_MED>Par_MED
                                Population(i)=Offspring;
                            end
                        end
                    end
                end
            end

            end


        end
    end

