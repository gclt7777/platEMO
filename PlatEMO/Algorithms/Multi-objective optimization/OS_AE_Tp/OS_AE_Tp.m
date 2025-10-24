classdef OS_AE_Tp < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% OS_AE_T : 目标空间分组 + 线性AE(PCA) + DE三策略生成 + 线性T子块写回
% - 第一道网 S：按 T 能量唯一指派（决定“更新谁”）
% - 第二道网 VA（全局一次）：把变量分为 收敛/扩张/多样（决定“怎么更”）
% - 保守更新T：rank(Y)==M 且 本代多样性列不增 时，才接受新T（可选EMA平滑）
%
% 关键参数（可命令行覆盖）：
%   K, topFrac         : 目标聚类数；指派后每组取前 topFrac 形成 S_k
%   lambda             : 岭回归拟合 T 的正则
%   eta                : 决策半步写回系数
%   kcap, kfrac        : 组内 PCA 潜维上限与比例
%   nSel, nPer         : VA 采样参数
%   mode3              : 0=二类（收敛/多样），1=三类（收敛/扩张/多样）
%   periodT            : 每隔多少代仅更新 T（fixS=0 时也会重分组）
%   emaA               : T 的 EMA 平滑系数（0 关闭）
%   pbestFrac          : 收敛DE中的 pbest 比例
%   Fcon,CRcon / Fexp,CRexp / Fdiv,CRdiv : 三条DE流的参数
%   fixS               : 1 固定 S；0 周期性重分组
%   gateT              : 1 启用“满秩+更干净”门控更新T
%   periodVA           : 全局VA重判周期（=1表示每代重判）
%
% 示例：
%   main('-algorithm',@OS_AE_T,'-problem',@DTLZ2,'-M',3,'-D',12,'-N',92,'-evaluation',5e4);

    methods
        function main(obj, Problem)
            [K,topFrac,lambda,eta,kcap,kfrac,nSel,nPer,mode3,periodT,emaA, ...
             pbestFrac,Fcon,CRcon,Fexp,CRexp,Fdiv,CRdiv,fixS,gateT,periodVA] = ...
             obj.ParameterSet(3,0.2,1e-5,0.5,4,1/3,3,10,1,5,0.0, 0.2,0.6,0.9, 0.7,0.9, 0.8,0.7, 1, 1, 1);

            Population = Problem.Initialization();
            gen        = 1;
            map_prev   = [];
            divCountPrev = inf;        % 上一次“多样性列”计数（越小越干净）
            % 初始化：分 O / 指派 S / 拟合 T
            [O_groups, S_groups, map] = group_and_assign_T(Problem, Population, K, topFrac, lambda);
            if emaA>0, map_prev = map; end
            % 全局VA标签的缓存（降频用）
            S_con_g_prev  = []; S_plus_g_prev = []; S_div_g_prev = [];

            while obj.NotTerminated(Population)
                X = Population.decs;   % N×D
                Y = Population.objs;   % N×M
                OffDec = X;

                % —— 候选 T 与（可选）重分 O/S —— 
                if gen==1 || mod(gen-1,periodT)==0
                    if fixS==0
                        [O_groups, S_groups, map_new0] = group_and_assign_T(Problem, Population, K, topFrac, lambda);
                    else
                        map_new0 = fitT_ridgeSVD(Y, X, lambda);
                    end
                else
                    map_new0 = fitT_ridgeSVD(Y, X, lambda);
                end

                % —— 全局VA：只对会写回的列做（第二张网） —— 
                S_total = unique([S_groups{:}]);   % 统一需要判别的列
                recomputeVA = (gen==1) || (mod(gen-1,periodVA)==0) || isempty(S_con_g_prev);
                if recomputeVA
                    [S_con_g,S_plus_g,S_div_g] = va_split_global_on_set(Problem,Population,S_total,map,nSel,nPer,mode3);
                else
                    S_con_g  = S_con_g_prev; 
                    S_plus_g = S_plus_g_prev; 
                    S_div_g  = S_div_g_prev;
                end
                divCountNow = numel(S_div_g);

                % —— 满秩 + 更干净 才接受新T（仿EAGO），否则用旧T —— 
                fullRankNow = (rank(Y) == Problem.M);
                if gateT && fullRankNow && (divCountNow <= divCountPrev)
                    map_accept   = map_new0; 
                    divCountPrev = divCountNow;
                else
                    map_accept   = map;
                end

                % —— 可选：EMA 平滑T（只平滑T，统计量取最新） —— 
                if emaA>0 && ~isempty(map_prev)
                    map_s.T    = (1-emaA)*map_prev.T + emaA*map_accept.T;
                    map_s.muY  = map_accept.muY;  map_s.sigY = map_accept.sigY;
                    map_s.muX  = map_accept.muX;  map_s.sigX = map_accept.sigX;
                    map        = map_s; 
                    map_prev   = map_s;
                else
                    map      = map_accept;
                    map_prev = map_accept;
                end

                % 缓存VA（降频时沿用）
                S_con_g_prev  = S_con_g; 
                S_plus_g_prev = S_plus_g; 
                S_div_g_prev  = S_div_g;

                % —— 按组生成与写回（只改各自 S_k ∩ 全局标签） —— 
                for k = 1:numel(O_groups)
                    Ok = O_groups{k}; 
                    Sk = S_groups{k};
                    if isempty(Ok) || isempty(Sk), continue; end

                    % 第二张网：取交集
                    S_con  = intersect(Sk, S_con_g);
                    S_plus = intersect(Sk, S_plus_g);
                    S_div  = intersect(Sk, S_div_g);

                    % 目标侧：线性AE(PCA) + 三条DE流
                    Yk   = Y(:,Ok);
                    Y_con = ae_pca_generate_DE(Yk, Y, 'con', kcap, kfrac, struct('F',Fcon,'CR',CRcon,'pbestFrac',pbestFrac));
                    if mode3==1
                        Y_exp = ae_pca_generate_DE(Yk, Y, 'exp', kcap, kfrac, struct('F',Fexp,'CR',CRexp));
                    else
                        Y_exp = Y_con; % 二类模式：扩张沿用收敛流
                    end
                    Y_div = ae_pca_generate_DE(Yk, Y, 'div', kcap, kfrac, struct('F',Fdiv,'CR',CRdiv));

                    % 写回（只改交集列；多样性通道含底噪+探索增量）
                    OffDec = writeback_DE(Problem, OffDec, X, Y, Ok, Sk, S_con, S_plus, S_div, map, Y_con, Y_exp, Y_div, eta);
                end

                % —— 裁剪 + 评估 + 环境选择 —— 
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(OffDec);
                Population = env_select_local([Population, Offspring], Problem.N);

                gen = gen + 1;
            end
        end
    end
end

%% ========================= 工具：分组/映射 =========================
function [O_groups, S_groups, map] = group_and_assign_T(Problem, Population, K, topFrac, lambda)
X = Population.decs; Y = Population.objs;
[~,D] = size(X);  M = size(Y,2);
if isempty(K) || ~isscalar(K) || K<1, K = min(3,M); end
if M==1, K_eff=1; else, K_eff = min(K, max(1, M-1)); end
try
    [Og,~,~] = OS_GroupByObjective(Population,'K',K_eff);
    O_groups = Og;
catch
    O_groups = os_group_by_objective_builtin(Y, K_eff);
end
map = fitT_ridgeSVD(Y, X, lambda);

Keff = numel(O_groups);
energy = zeros(Keff,D);
for k = 1:Keff
    Ok = O_groups{k}; if isempty(Ok), continue; end
    Tk = map.T(Ok,:);
    energy(k,:) = sqrt(sum(Tk.^2,1));
end
[~, owner] = max(energy, [], 1);
S_groups = cell(1,Keff);
for k = 1:Keff
    idx = find(owner==k);
    if isempty(idx), S_groups{k} = []; continue; end
    e = energy(k,idx); [~,ord] = sort(e,'descend');
    topK = max(1, round(topFrac*numel(idx)));
    S_groups{k} = idx(ord(1:topK));
end
end

function map = fitT_ridgeSVD(Y, X, lambda)
muY = mean(Y,1); sigY = std(Y,0,1); sigY(sigY==0)=1;
muX = mean(X,1); sigX = std(X,0,1); sigX(sigX==0)=1;
Yz  = (Y - muY)./sigY;
[U,S,V] = svd(Yz,'econ'); sig = diag(S);
G = V * diag(sig./(sig.^2 + lambda)) * U';
Xz = (X - muX)./sigX;
T  = G * Xz;
map.T=T; map.muY=muY; map.sigY=sigY; map.muX=muX; map.sigX=sigX;
end

function O_groups = os_group_by_objective_builtin(Y, K)
[~,M] = size(Y);
if M==1 || K<=1, O_groups = {1:M}; return; end
K = min(K, max(1, M-1));
Yz = zscore(Y,0,1); Yz(:,any(isnan(Yz),1))=0;
nrm = sqrt(sum(Yz.^2,1)); nrm(nrm==0)=1; U = Yz ./ nrm;
S = U.'*U;  S(1:M+1:end)=0;
A = max(S,0); A = A - diag(diag(A));
d = sum(A,2); Dm = diag(d + eps);
L = eye(M) - Dm^(-1/2)*A*Dm^(-1/2); L=(L+L.')/2;
[V,E] = eig(L); [~,ord]=sort(diag(E),'ascend');
H = V(:,ord(1:K)); Hn = H./max(sqrt(sum(H.^2,2)),eps); Hn(~isfinite(Hn))=0;
opts = statset('MaxIter',200,'Display','off');
repl = max(5, min(10, M-1));
lbl = kmeans(Hn,K,'Replicates',repl,'Options',opts);
O_groups = cell(1,K); for k=1:K, O_groups{k}=find(lbl==k).'; end
end

%% ========================= AE(PCA)+DE 三策略生成 =========================
function Y_new = ae_pca_generate_DE(Yk, Y_all, mode, kcap, kfrac, prm)
if nargin < 6 || isempty(prm), prm = struct(); end
if ~isfield(prm,'F') || isempty(prm.F), prm.F = 0.6; end
if ~isfield(prm,'CR') || isempty(prm.CR), prm.CR = 0.9; end
if ~isfield(prm,'pbestFrac') || isempty(prm.pbestFrac), prm.pbestFrac = 0.2; end
F  = prm.F; 
CR = prm.CR;

[Yz, mu, sg] = safe_zscore(Yk);
r  = max(1, min([size(Yk,2), max(1, ceil(size(Yk,2)*kfrac)), kcap]));
W  = pca_basis(Yz, r);
Z  = Yz * W;                 % N×r
N  = size(Z,1); k = size(Z,2);

switch lower(mode)
    case 'con'  % current-to-pbest/1/bin
        p = max(0.05, min(0.5, prm.pbestFrac));
        pbest_idx = pick_pbest_indices(Y_all, p);
        if numel(pbest_idx) >= N
            sel = pbest_idx(randperm(numel(pbest_idx), N));
            Zpb = Z(sel,:);
        else
            rep = ceil(N/numel(pbest_idx));
            sel = repmat(pbest_idx(:), rep, 1); sel = sel(1:N);
            Zpb = Z(sel,:);
        end
        Znew = de_current_to_pbest_1_bin(Z, Zpb, F, CR);

    case 'exp'  % best/1/bin
        zbest = pick_bestZ_by_sumY(Yk, Z);
        Znew  = de_best_1_bin(Z, zbest, F, CR);

    case 'div'  % rand/2/bin
        Znew  = de_rand_2_bin(Z, F, CR);

    otherwise
        Znew  = Z;
end

Y_new = (Znew * W') .* sg + mu;
end

function idx = pick_pbest_indices(Y, p)
N = size(Y,1);
score = sum(Y,2);           % 简洁代理：越小越好（可替换为层号）
[~,ord] = sort(score,'ascend');
np = max(2, round(p*N));
idx = ord(1:np);
end

function zbest = pick_bestZ_by_sumY(Yk, Z)
[~,best] = min(sum(Yk,2)); zbest = Z(best,:);
end

% --- 三个DE算子（潜空间） ---
function Znew = de_current_to_pbest_1_bin(Z, Zpb, F, CR)
[N,k] = size(Z); Znew = Z; idx = 1:N;
for i = 1:N
    r = idx; r(i)=[];
    if numel(r)<2, Znew(i,:)=Z(i,:); continue; end
    r = r(randperm(numel(r),2));
    v = Z(i,:) + (Zpb(i,:)-Z(i,:)) + F*(Z(r(1),:)-Z(r(2),:));
    jrand = randi(k); mask = (rand(1,k)<CR); mask(jrand)=true;
    u = Z(i,:); u(mask)=v(mask); Znew(i,:) = u;
end
end

function Znew = de_best_1_bin(Z, zbest, F, CR)
[N,k] = size(Z); Znew = Z; idx = 1:N;
for i = 1:N
    r = idx; r(i)=[];
    if numel(r)<2, Znew(i,:)=Z(i,:); continue; end
    r = r(randperm(numel(r),2));
    v = zbest + F*(Z(r(1),:)-Z(r(2),:));
    jrand = randi(k); mask = (rand(1,k)<CR); mask(jrand)=true;
    u = Z(i,:); u(mask)=v(mask); Znew(i,:) = u;
end
end

function Znew = de_rand_2_bin(Z, F, CR)
[N,k] = size(Z); Znew = Z; idx = 1:N;
for i = 1:N
    r = idx; r(i)=[];
    if numel(r)<5, Znew(i,:)=Z(i,:); continue; end
    r = r(randperm(numel(r),5));
    v = Z(r(1),:) + F*(Z(r(2),:)-Z(r(3),:)) + F*(Z(r(4),:)-Z(r(5),:));
    jrand = randi(k); mask = (rand(1,k)<CR); mask(jrand)=true;
    u = Z(i,:); u(mask)=v(mask); Znew(i,:) = u;
end
end

%% ========================= VA/写回/选择 =========================
function [S_con,S_plus,S_div] = va_split_global_on_set(Problem,Population,S_set,map,nSel,nPer,mode3)
S_con=[]; S_plus=[]; S_div=[];
if isempty(S_set), return; end
N = Problem.N;

for idx = 1:numel(S_set)
    i = S_set(idx);
    Sample = randi(N,1,nSel);
    vote = zeros(1,nSel);  % 1=收敛 2=扩张 0=多样
    for j = 1:nSel
        baseDec = repmat(Population(Sample(j)).decs,nPer,1);
        Y0 = repmat(Population(Sample(j)).objs,nPer,1);
        Ym = Y0 - Y0.*(rand(nPer,Problem.M))*0.25;   % 收缩候选
        Yp = Y0 + Y0.*(rand(nPer,Problem.M))*0.25;   % 扩张候选
        Yr = Y0 + (rand(nPer,Problem.M)-0.5).*0.1;   % 随机候选

        Xi_m = apply_map_col(map, Ym, i);
        Xi_p = apply_map_col(map, Yp, i);
        Xi_r = apply_map_col(map, Yr, i);
        Xi_m = min(max(Xi_m,Problem.lower(i)),Problem.upper(i));
        Xi_p = min(max(Xi_p,Problem.lower(i)),Problem.upper(i));
        Xi_r = min(max(Xi_r,Problem.lower(i)),Problem.upper(i));

        Dec_m = baseDec; Dec_m(:,i) = Xi_m;
        Dec_p = baseDec; Dec_p(:,i) = Xi_p;
        Dec_r = baseDec; Dec_r(:,i) = Xi_r;

        Obj_m = Problem.Evaluation(Dec_m).objs;
        Obj_p = Problem.Evaluation(Dec_p).objs;
        Obj_r = Problem.Evaluation(Dec_r).objs;

        m_m = sum(mean(Obj_m,1).^2);
        m_p = sum(mean(Obj_p,1).^2);
        m_r = sum(mean(Obj_r,1).^2);

        if m_m <= m_r && (~mode3 || m_m <= m_p)
            vote(j)=1;
        elseif mode3 && (m_p < m_m && m_p <= m_r)
            vote(j)=2;
        else
            vote(j)=0;
        end
    end

    if sum(vote==1) >= max(sum(vote==0),sum(vote==2))
        S_con  = [S_con, i];
    elseif mode3 && sum(vote==2) >= max(sum(vote==0),sum(vote==1))
        S_plus = [S_plus, i];
    else
        S_div  = [S_div, i];
    end
end
end

function OffDec = writeback_DE(Problem, OffDec, X, Y, Ok, Sk, S_con, S_plus, S_div, map, Y_con, Y_exp, Y_div, eta)
N = size(X,1);
% 收敛：半步融合
if ~isempty(S_con)
    Scon = intersect(Sk,S_con);
    if ~isempty(Scon)
        Xhat = apply_map_sub(map, Y_con, Ok, Scon);
        Xhat = min(max(Xhat, repmat(Problem.lower(Scon),N,1)), repmat(Problem.upper(Scon),N,1));
        OffDec(:,Scon) = (1-eta)*OffDec(:,Scon) + eta*Xhat;
    end
end
% 扩张：半步融合
if ~isempty(S_plus)
    Splus = intersect(Sk,S_plus);
    if ~isempty(Splus)
        Xhat = apply_map_sub(map, Y_exp, Ok, Splus);
        Xhat = min(max(Xhat, repmat(Problem.lower(Splus),N,1)), repmat(Problem.upper(Splus),N,1));
        OffDec(:,Splus) = (1-eta)*OffDec(:,Splus) + eta*Xhat;
    end
end
% 多样性：底噪 + 探索差分（相对当前 Y 的增量）
if ~isempty(S_div)
    Sdiv = intersect(Sk,S_div);
    if ~isempty(Sdiv)
        base  = sqrt(OffDec(randperm(N),Sdiv).*OffDec(randperm(N),Sdiv));
        base2 = OffDec(:,Sdiv); a = randperm(N);
        base(base > OffDec(a,Sdiv)) = base2(base > OffDec(a,Sdiv));
        dY = (Y_div - Y(:,Ok));                          % 探索增量
        dX = apply_map_sub(map, dY, Ok, Sdiv);
        Xcand = base + eta*dX;
        Xcand = min(max(Xcand, repmat(Problem.lower(Sdiv),N,1)), repmat(Problem.upper(Sdiv),N,1));
        OffDec(:,Sdiv) = Xcand;
    end
end
end

% —— 标准化/反标准化映射（子块 & 单列） —— 
function Xhat_Sk = apply_map_sub(map, Yk_new, Ok, Sk)
Yz = bsxfun(@rdivide, bsxfun(@minus, Yk_new, map.muY(Ok)), map.sigY(Ok));
Xz = Yz * map.T(Ok,Sk);
Xhat_Sk = bsxfun(@plus, bsxfun(@times, Xz, map.sigX(Sk)), map.muX(Sk));
end
function Xi = apply_map_col(map, Ycand, i)
Yz = bsxfun(@rdivide, bsxfun(@minus, Ycand, map.muY), map.sigY);
Xz_i = Yz * map.T(:,i);
Xi   = Xz_i .* map.sigX(i) + map.muX(i);
end

% —— 环境选择（非支配 + 角度截断） —— 
function Population = env_select_local(Population, N)
PopObj = Population.objs;
[FrontNo, MaxFNo] = NDSort(PopObj, N);
Next = FrontNo < MaxFNo;
Last = find(FrontNo == MaxFNo);
K = N - sum(Next);
if K > 0
    Choose = truncation_angle(PopObj(Last,:), K);
    Next(Last(Choose)) = true;
end
Population = Population(Next);
end
function Choose = truncation_angle(PopObj, K)
fmax = max(PopObj,[],1); fmin = min(PopObj,[],1);
span = fmax - fmin; span(span==0)=1;
P = (PopObj - fmin) ./ span;
nrm = sqrt(sum(P.^2,2)); nrm(nrm==0)=1; U = P ./ nrm;
Cosine = U*U.'; Cosine(1:size(Cosine,1)+1:end)=0;
Choose = false(1,size(P,1));
[~,extreme] = max(P,[],1); Choose(extreme)=true;
if sum(Choose) > K
    sel = find(Choose); Choose=false(1,size(P,1));
    Choose(sel(randperm(numel(sel),K))) = true;
else
    while sum(Choose) < K
        unSel = find(~Choose);
        [~,x] = min(max(Cosine(~Choose,Choose),[],2));
        Choose(unSel(x)) = true;
    end
end
end

% —— 线性AE工具 —— 
function [Z,mu,sg] = safe_zscore(X)
mu = mean(X,1); sg = std(X,0,1); sg(sg==0)=1; Z = (X - mu)./sg;
end
function W = pca_basis(X, k)
[~,~,V] = svd(X,'econ'); W = V(:,1:k);
end
