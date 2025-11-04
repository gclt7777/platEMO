function [ParentC,ParentM] = MatingSelection(CA,DA,Fit,FE,maxFE,N)
% 匹配算子选择

    if isempty(CA)
        ParentC = INDIVIDUAL.empty();
        ParentM = INDIVIDUAL.empty();
        return;
    end
    if isempty(DA)
        DA = CA;
    end

    preference = FE/maxFE;
    if preference < 0.5 || (preference > 0.5 && rand < 0.5)
        MatingPool1 = TournamentSelection(2,ceil(N/2),Fit(:));
        ParentC     = [CA(MatingPool1),DA(randi(length(DA),1,ceil(N/2)))];
        MatingPool2 = TournamentSelection(2,N,Fit(:));
        ParentM     = CA(MatingPool2);
    else
        DAParent1 = randi(length(DA),1,ceil(N/2));
        DAParent2 = randi(length(DA),1,ceil(N/2));
        Dominate  = any(DA(DAParent1).objs<DA(DAParent2).objs,2) - any(DA(DAParent1).objs>DA(DAParent2).objs,2);
        MatingPool1 = TournamentSelection(2,ceil(N/2),Fit(:));
        ParentC     = [DA([DAParent1(Dominate==1),DAParent2(Dominate~=1)]),CA(MatingPool1)];
        ParentM     = DA(randi(length(DA),1,N));
    end
end
