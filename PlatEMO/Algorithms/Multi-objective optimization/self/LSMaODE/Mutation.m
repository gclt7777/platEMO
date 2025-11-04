function [Offspring_dec] = Mutation(localgroup,i,k,Parent,lower,upper,mutationStrength,Population,L1list)
if i<=localgroup
    %%%% subPopulation
    if rand>0.1
        Offspring_dec=Parent.dec;
        sigma=(upper(k)-lower(k))./mutationStrength;
        Offspring_dec(k)=Offspring_dec(k)+normrnd(0,sigma);
        while any((Offspring_dec<lower)|(Offspring_dec>upper))
            Offspring_dec=Parent.dec;
            Offspring_dec(k)=Offspring_dec(k)+normrnd(0,sigma);
        end
    else
        temp_localPop=Population;
        temp_localPop(i)=[];
        len=length(temp_localPop);
        idx=randperm(len);
        Offspring_dec=Parent.dec;
        Parent1=temp_localPop(idx(1));
        Parent2=temp_localPop(idx(2));
        Parent3=temp_localPop(idx(3));
        if rand>0.5
            L1POP=[Parent1,Parent2];
            tempDEC=Parent1.dec+rand.*(Parent2.dec-Parent1.dec);
        else
            L1POP=[Parent1,Parent2,Parent3];
            tempDEC=Parent1.dec+rand.*(Parent2.dec-Parent3.dec);
        end
        Offspring_dec(k)=tempDEC(k);
        L1dec=L1POP.decs;
        L1bound_up=max(L1dec,[],1);
        L1bound_low=min(L1dec,[],1);
        Offspring_dec(k)=min(max(L1bound_low(k),Offspring_dec(k)),L1bound_up(k));
    end
else
    %%%% subPopulation
    L1len=length(L1list);
    if rand>0.5&&L1len>=3
        idx=randperm(L1len);
        Parent1=Population(L1list(idx(1)));
        Parent2=Population(L1list(idx(2)));
        Parent3=Population(L1list(idx(3)));
        if rand>0.5
            L1POP=[Parent1,Parent2];
            Offspring_dec=Parent1.dec+rand.*(Parent2.dec-Parent1.dec);
        else
            L1POP=[Parent1,Parent2,Parent3];
            Offspring_dec=Parent1.dec+rand.*(Parent2.dec-Parent3.dec);
        end
        L1dec=L1POP.decs;
        L1bound_up=max(L1dec,[],1);
        L1bound_low=min(L1dec,[],1);
        %%%%%限制Offspring_dec的范围
        Offspring_dec=min(max(L1bound_low,Offspring_dec),L1bound_up);
    else %%%%交叉
        Offspring_dec=Parent.dec;
        d=length(Offspring_dec);
        k=randperm(d,1);
        sigma=(upper(k)-lower(k))./mutationStrength;
        Offspring_dec(k)=Offspring_dec(k)+normrnd(0,sigma);
        while any((Offspring_dec<lower)|(Offspring_dec>upper))
            k=randperm(d,1);
            Offspring_dec=Parent.dec;
            Offspring_dec(k)=Offspring_dec(k)+normrnd(0,sigma);
        end
    end
    
end


end


