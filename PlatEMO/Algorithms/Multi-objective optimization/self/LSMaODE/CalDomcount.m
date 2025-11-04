function Domcount = CalDomcount(PopObj,input_pop)


    N = size(PopObj,1);

    Dominate = false(1,N);
        for j = 1 : N
            k = any(input_pop<PopObj(j,:)) - any(input_pop>PopObj(j,:));
            if k == -1
                Dominate(j) = true;
            end
        end
 
    
    Domcount = sum(Dominate);
end