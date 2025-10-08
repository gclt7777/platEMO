classdef OSL_LK < ALGORITHM
% <2025> <multi/many> <real/ineger>
% OSL_LK —— Many-objective OSL (Linear + PBI/Angle preimage) with diversification
% 用法（命令行示例）：
%   platemo('algorithm',@OSL_LK,'problem',@DTLZ2,'M',10,'N',120,'maxFE',1e5);

    methods
        function main(Algorithm,Problem)
            % ----------------- 参数 -----------------
            [nInject, perBucket, step_eta, pbi_theta, ...
             trigger, period_tau, stagn_K, stagn_tol, ...
             preimage_loss, preimage_multi, preimage_it, ...
             pc, etaC, pm, etaM] = Algorithm.ParameterSet( ...
                48, 6, 0.20, 5.0, ...            % 注入数、桶内保留、PBI步长/θ
                1, 25, 10, 1e-4, ...              % 触发：1=stagn, 2=periodic, 3=always
                1, 2, 200, ...                    % 预像：1=PBI, 2=AngMag；多起点/迭代
                1.0, 20, 1.0, 20);                % SBX/PM 参数

            % 参考向量（优先用 PlatEMO 自带）
            try
                W = UniformPoint(Problem.N,Problem.M);
            catch
                W = genRefVecSimple(Problem.M,12);
            end
            W = normRows(W);

            % 初始化
            Population = Problem.Initialization();
            bestHist   = []; lastBest = inf; %#ok<NASGU>
            gen        = 0;

            while Algorithm.NotTerminated(Population)
                gen = gen + 1;

                % ========= 正常进化（SBX+PM） =========
                OffDec    = variationSBXPM(Population.decs, Problem.lower, Problem.upper, pc, etaC, pm, etaM);
                Offspring = Problem.Evaluation(OffDec);

                % ========= 是否触发 OSL 注入 =========
                trig = shouldTrigger(bestHist, Population.objs, W, trigger, period_tau, stagn_K, stagn_tol);
                Inj = [];
                if trig
                    InjDec = osl_inject_linear(Population.decs, Population.objs, ...
                                               @(x)Problem.CalObj(x), ...
                                               Problem.lower, Problem.upper, ...
                                               W, nInject, perBucket, step_eta, pbi_theta, ...
                                               preimage_loss, preimage_multi, preimage_it);
                    if ~isempty(InjDec)
                        Inj = Problem.Evaluation(InjDec);
                    end
                end

                % ========= 环境选择（NSGA-II风格） =========
                if isempty(Inj)
                    Population = envSelect_NSGA2([Population,Offspring],Problem.N);
                else
                    Population = envSelect_NSGA2([Population,Offspring,Inj],Problem.N);
                end

                % ========= 更新触发参考（PBI最优） =========
                curBest = bestPBI(Population.objs,W,pbi_theta);
                if isempty(bestHist), bestHist = curBest; else, bestHist(end+1) = curBest; end
            end
        end
    end
end

% ===================== 线性版注入 =====================
function Xinj = osl_inject_linear(PopDec,PopObj,CalObj,lb,ub,W,nInject,perBucket,step_eta,pbi_theta,preimage_loss,preimage_multi,preimage_it)
    [Yn,zmin,zmax] = normY(PopObj);
    K  = size(W,1);
    idx= routeBuckets(Yn,W);
    Xinj = [];

    Ycand = []; srcK = [];
    for k=1:K
        Yk = Yn(idx==k,:);
        if size(Yk,1) < 8, continue; end
        % PBI 定向一步
        Ystep = pbiStep(Yk,W(k,:),pbi_theta,step_eta);
        % 局部线性回归 Y->Ystep（带偏置的岭回归）
        mdl = trainLinear(Yk,Ystep,1e-3);
        Yhat= predictLinear(mdl,Yk);

        % 桶内按 PBI 改善排序，保留前 perBucket
        g0 = pbi(Yk,     W(k,:),pbi_theta);
        g1 = pbi(Yhat,   W(k,:),pbi_theta);
        [~,ord] = sort(g1-g0,'ascend');
        L = min(perBucket,numel(ord));
        pick = ord(1:L);
        Ycand = [Ycand; Yhat(pick,:)]; %#ok<AGROW>
        srcK  = [srcK;  k*ones(L,1)];  %#ok<AGROW>
    end
    if isempty(Ycand), return; end

    sel = maxminCos(Ycand,nInject);
    Ysel_n = Ycand(sel,:);
    Ysel   = denormY(Ysel_n,zmin,zmax);

    switch preimage_loss
        case 1 % PBI 差
            Xinj = preimagePBI(Ysel,CalObj,lb,ub,srcK(sel),W,zmin,pbi_theta,preimage_multi,preimage_it);
        otherwise % 角度+幅值
            Xinj = preimageAngMag(Ysel,CalObj,lb,ub,0.7,preimage_multi,preimage_it);
    end
    Xinj = clipBounds(unique(round(Xinj,12),'rows'),lb,ub);
end

% ===================== 线性岭回归（带偏置） =====================
function mdl = trainLinear(X, Y, lam)
    % X: (n x M)  Y: (n x M)  lam: 正则化系数
    % 拟合 [X,1] * Wb ≈ Y
    Z  = [X, ones(size(X,1),1)];              % 拼偏置
    A  = Z' * Z + lam * eye(size(Z,2));
    B  = Z' * Y;
    Wb = A \ B;                                % 闭式解
    mdl.W = Wb(1:end-1,:);                     % 权重
    mdl.b = Wb(end,:);                         % 偏置
end
function Yhat = predictLinear(mdl, X)
    % 预测：X * W + b
    Yhat = X * mdl.W + repmat(mdl.b, size(X,1), 1);
end

% ===================== 触发策略/度量 =====================
function yes = shouldTrigger(bestHist, Y, W, trigger, period_tau, stagn_K, stagn_tol)
    switch trigger
        case 3 % always
            yes = true;
        case 2 % periodic
            yes = (mod(numel(bestHist)+1, period_tau) == 1);
        otherwise % stagnation
            b = bestPBI(Y,W,5.0);
            hh = [bestHist(:); b];
            K = min(stagn_K, numel(hh)-1);
            if K <= 0, yes = false; return; end
            yes = (hh(end-K) - hh(end) < stagn_tol);
    end
end

function best = bestPBI(Y,W,theta)
    Yn = normY(Y); K = size(W,1);
    gmin = inf(size(Yn,1),1);
    for k=1:K
        gmin = min(gmin, pbi(Yn, W(k,:), theta));
    end
    best = min(gmin);
end

% ===================== NSGA-II 环境选择 =====================
function Pop = envSelect_NSGA2(PopAll,N)
    Obj = PopAll.objs; % (n x M)
    FrontNo = NDSort(Obj,inf);
    Pop = [];
    sel = false(1,length(PopAll));
    F = 1;
    remain = N;
    while remain > 0
        idx = find(FrontNo==F);
        if isempty(idx), break; end
        if length(idx) <= remain
            sel(idx) = true;
            remain = remain - length(idx);
        else
            % Crowding distance on this front
            CD = crowdingDistance(Obj(idx,:));
            [~,ord] = sort(CD,'descend');
            sel(idx(ord(1:remain))) = true;
            remain = 0;
        end
        F = F+1;
    end
    Pop = PopAll(sel);
end

function CD = crowdingDistance(Y)
    [N,M] = size(Y);
    CD = zeros(N,1);
    for m=1:M
        [~,I] = sort(Y(:,m));
        CD(I(1))   = inf; CD(I(end)) = inf;
        ym = Y(I,m);
        den = max(ym)-min(ym); if den==0, den=1; end
        for i=2:N-1
            CD(I(i)) = CD(I(i)) + (ym(i+1)-ym(i-1))/den;
        end
    end
end

% ===================== 预像（两种损失） =====================
function X = preimagePBI(Yt,CalObj,lb,ub,srcK,W,zmin,theta,nStart,maxIt)
    n = size(Yt,1); D = numel(lb); X = zeros(n,D);
    for i=1:n
        wi = W(max(1,srcK(i)),:);
        fun = @(x) objPBI(x,Yt(i,:),wi,CalObj,zmin,theta);
        X(i,:) = multistartLocal(fun,lb,ub,D,nStart,maxIt);
    end
end
function f = objPBI(x,yt,w,CalObj,zmin,theta)
    y = CalObj(x);
    fx = pbiSingle(y - zmin, w, theta);
    ft = pbiSingle(yt - zmin, w, theta);
    f  = (fx - ft)^2;
end
function g = pbiSingle(F,w,theta)
    w = w(:)'/max(norm(w),1e-12);
    proj = sum(F.*w,2); r = F - proj.*w;
    g = proj + theta*sqrt(sum(r.^2,2));
end

function X = preimageAngMag(Yt,CalObj,lb,ub,alpha,nStart,maxIt)
    n = size(Yt,1); D = numel(lb); X = zeros(n,D);
    for i=1:n
        yt = Yt(i,:);
        fun = @(x) objAngMag(x,yt,CalObj,alpha);
        X(i,:) = multistartLocal(fun,lb,ub,D,nStart,maxIt);
    end
end
function f = objAngMag(x,yt,CalObj,alpha)
    y  = CalObj(x);
    cy = dot(y,yt)/(max(norm(y),1e-12)*max(norm(yt),1e-12));
    f  = alpha*(1-cy) + (1-alpha)*norm(y-yt)/max(norm(yt),1e-12);
end

function xbest = multistartLocal(fun,lb,ub,D,nStart,maxIt)
    useFmincon = license('test','Optimization_Toolbox');
    xbest=(lb+ub)/2; fbest=inf;
    if useFmincon
        opt = optimoptions('fmincon','display','off','MaxIterations',maxIt,'Algorithm','interior-point');
        for s=1:nStart
            x0 = lb + rand(1,D).*(ub-lb);
            try, [x,fv] = fmincon(fun,x0,[],[],[],[],lb,ub,[],opt);
            catch, x=x0; fv=fun(x0); end
            if fv<fbest, fbest=fv; xbest=x; end
        end
    else
        for s=1:nStart*3
            x0 = lb + rand(1,D).*(ub-lb);
            [x,fv] = fminsearch(@(z) fun(min(max(z,lb),ub)), x0, optimset('display','off'));
            x = min(max(x,lb),ub);
            if fv<fbest, fbest=fv; xbest=x; end
        end
    end
end

% ===================== 目标空间工具 =====================
function g = pbi(Yn,w,theta)
    w = w(:)'/max(norm(w),1e-12);
    proj = sum(Yn.*w,2); r = Yn - proj.*w; d2 = sqrt(sum(r.^2,2));
    g = proj + theta*d2;
end
function Yn2 = pbiStep(Yn,w,theta,eta)
    w = w(:)'/max(norm(w),1e-12);
    proj = sum(Yn.*w,2); r = Yn - proj.*w; d2 = sqrt(sum(r.^2,2));
    u_r  = r ./ max(d2,1e-12);
    grad = repmat(w,size(Yn,1),1) + theta*u_r;
    Yn2  = max(Yn - eta*grad, 0);
end
function [Yn,zmin,zmax] = normY(Y)
    zmin = min(Y,[],1); zmax = max(Y,[],1);
    Yn = (Y - zmin)./max(zmax-zmin,1e-12);
end
function Y = denormY(Yn,zmin,zmax)
    Y = Yn.*max(zmax-zmin,1e-12) + zmin;
end
function Wn = normRows(W), Wn = W./max(sqrt(sum(W.^2,2)),1e-12); end
function idx = routeBuckets(Yn,Wn)
    C = Yn*Wn'; d = sqrt(sum(Yn.^2,2)); d(d<1e-12)=1;
    idx = zeros(size(Yn,1),1);
    for i=1:size(Yn,1)
        [~,idx(i)] = max(C(i,:)./d(i));
    end
end
function W = genRefVecSimple(M,H)
    P = compo(H,M); W = P/H; W = normRows(W);
    function P = compo(H,M)
        if M==1, P=H; return; end
        P=[]; for i=0:H
            Q = compo(H-i,M-1); P = [P; [i*ones(size(Q,1),1),Q]]; %#ok<AGROW>
        end
    end
end
function X = clipBounds(X,lb,ub), X = min(max(X,lb),ub); end

% ===================== 变异算子（SBX+PM） =====================
function Off = variationSBXPM(Dec,lb,ub,pc,etaC,pm,etaM)
    [N,D] = size(Dec); Off = zeros(N,D);
    for i=1:2:N
        p1 = randi(N); p2 = randi(N);
        c1 = Dec(p1,:); c2 = Dec(p2,:);
        if rand < pc
            u = rand(1,D); beta = zeros(1,D);
            beta(u<=0.5) = (2*u(u<=0.5)).^(1/(etaC+1));
            beta(u>0.5 ) = (2-2*u(u>0.5)).^(-1/(etaC+1));
            child1 = 0.5*((1+beta).*c1 + (1-beta).*c2);
            child2 = 0.5*((1-beta).*c1 + (1+beta).*c2);
        else
            child1 = c1; child2 = c2;
        end
        % Polynomial mutation
        child1 = polyMut(child1,lb,ub,pm,etaM);
        child2 = polyMut(child2,lb,ub,pm,etaM);
        Off(i,:) = child1;
        if i+1<=N, Off(i+1,:) = child2; end
    end
end
function c = polyMut(c,lb,ub,pm,etaM)
    D = numel(c);
    for j=1:D
        if rand < pm
            u = rand;
            if u<0.5, delta=(2*u)^(1/(etaM+1))-1;
            else,     delta=1-(2-2*u)^(1/(etaM+1));
            end
            c(j) = c(j) + (ub(j)-lb(j))*delta;
        end
    end
    c = min(max(c,lb),ub);
end
