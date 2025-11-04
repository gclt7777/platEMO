function [CA,Fit] = UpdateCA(CA,New,MaxSize)
% Update the convergence archive.

    CA = [CA,New];
    N  = length(CA);
    if N <= MaxSize
        Fit = Calculate_DDC(CA.objs,N);
        return;
    end

    CAObj  = CA.objs;
    Choose = false(1,N);
    ddC    = Calculate_DDC(CAObj,N);
    [~,index] = sort(ddC,'ascend');
    Choose(index(1:MaxSize)) = true;
    CA   = CA(Choose);
    Fit  = ddC(Choose);
end
