function Domcount = CalDomcount(PopObj,input_obj)
% Count the number of solutions in PopObj that dominate the input object.

    if isempty(PopObj)
        Domcount = 0;
        return;
    end

    N = size(PopObj,1);
    Dominate = false(1,N);
    for j = 1 : N
        better = any(input_obj < PopObj(j,:));
        worse  = any(input_obj > PopObj(j,:));
        if ~better && worse
            Dominate(j) = true;
        end
    end
    Domcount = sum(Dominate);
end
