classdef OS_AE_T < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% OS_AE_T : Objective-space Grouping + AE (PCA) + Linear T write-back
% - 仅用线性 T 做“变量指派 + 写回”，写回时只改各自 S_k 列（避免串扰）
% - 每代重拟合 T（岭回归 SVD 解），可选 EMA 平滑
%
% 示例：
%   main('-algorithm',@OS_AE_T,'-problem',@DTLZ2, ...
%        '-M',3,'-D',12,'-N',92,'-evaluation',5e4);

    methods
        function main(obj, Problem)
            % 参数（可命令行覆盖）
            %   K        : 目标簇数（自动钳制 < M）
            %   topFrac  : 每组写回变量占比（唯一指派后取前 topFrac）
            %   lambda   : 拟合 T 的岭值
            %   eta      : 半步写回步长
            %   kcap     : 组内 AE 的潜空间最大维度
            %   kfrac    : 组内 AE 的潜空间维度占比
            %   F,Cr     : 潜空间 DE 参数
            %   period   : 每隔多少代重分组/重指派
            %   emaA     : T 的 EMA 平滑系数（0 关闭）
            [K,topFrac,lambda,eta,kcap,kfrac,F,Cr,period,emaA] = ...
                obj.ParameterSet(3,0.2,1e-5,0.5,4,1/3,0.7,0.9,5,0.0);

            % 初始化
            Population = Problem.Initialization();
            gen = 1;  T_prev = [];  %#ok<NASGU>

            % 首次：分组+T+指派
            [O_groups, S_groups, T] = group_and_assign_T(Problem, Population, K, topFrac, lambda);
            if emaA>0, T_prev = T; end

            while obj.NotTerminated(Population)
                X = Population.decs;   % N×D
                Y = Population.objs;   % N×M
                OffDec = X;

                % 周期性重分组/指派 + 每代更新 T
                if gen==1 || mod(gen-1,period)==0
                    [O_groups, S_groups, T_new] = group_and_assign_T(Problem, Population, K, topFrac, lambda);
                else
                    T_new = fitT_ridgeSVD(Y, X, lambda);   % 仅更新 T
                end
                if emaA>0 && ~isempty(T_prev)
                    T = (1-emaA)*T_prev + emaA*T_new;
                    T_prev = T;
                else
                    T = T_new;  T_prev = T_new;
                end

                % —— 逐目标组：AE(PCA) 生成 + 用子块 T(Ok,Sk) 半步写回
                for k = 1:numel(O_groups)
                    Ok = O_groups{k};      % 该组目标索引
                    Sk = S_groups{k};      % 该组写回的决策索引
                    if isempty(Ok) || isempty(Sk), continue; end

                    % 组内 AE: PCA + 一轮 DE
                    Yk     = Y(:,Ok);
                    Yk_new = ae_pca_generate(Yk, kcap, kfrac, F, Cr);  % N×|Ok|

                    % 线性写回（只用子块）
                    Xhat_Sk      = Yk_new * T(Ok,Sk);                 % N×|Sk|
                    OffDec(:,Sk) = (1-eta)*OffDec(:,Sk) + eta*Xhat_Sk;
                end

                % 裁剪 + 评估 + 环境选择
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(OffDec);
                Population = env_select_local([Population, Offspring], Problem.N);

                gen = gen + 1;
            end
        end
    end
end

%% ========================= 内联工具 =========================

function [O_groups, S_groups, T] = group_and_assign_T(Problem, Population, K, topFrac, lambda)
% 目标分组（谱聚类，稳健小 M/NaN） + 拟合 T + 按能量唯一指派
X = Population.decs; Y = Population.objs;
[~,D] = size(X);  M = size(Y,2);

% K 钳制
if isempty(K) || ~isscalar(K) || K<1, K = min(3,M); end
if M==1, K_eff=1; else, K_eff = min(K, max(1, M-1)); end

% 分组：优先外部函数，否则内置
try
    [Og,~,~] = OS_GroupByObjective(Population,'K',K_eff);
    O_groups = Og;
catch
    O_groups = os_group_by_objective_builtin(Y, K_eff);
end

% 拟合 T
T = fitT_ridgeSVD(Y, X, lambda);

% 唯一指派并组内取 topFrac
Keff = numel(O_groups);
energy = zeros(Keff,D);
for k = 1:Keff
    Ok = O_groups{k}; if isempty(Ok), continue; end
    Tk = T(Ok,:);                         % |Ok|×D
    energy(k,:) = sqrt(sum(Tk.^2,1));     % 可选： ./ sqrt(numel(Ok))
end
[~, owner] = max(energy, [], 1);

S_groups = cell(1,Keff);
for k = 1:Keff
    idx = find(owner==k);
    if isempty(idx), S_groups{k} = []; continue; end
    e = energy(k,idx);
    [~,ord] = sort(e,'descend');
    topK = max(1, round(topFrac*numel(idx)));
    S_groups{k} = idx(ord(1:topK));
end
end

function T = fitT_ridgeSVD(Y, X, lambda)
% 岭回归 (Y'Y+λI)^{-1}Y'X 的 SVD 稳定解（列标准化后求解）
Yz = zscore(Y,0,1);  Xz = zscore(X,0,1);
Yz(:,any(isnan(Yz),1)) = 0;  Xz(:,any(isnan(Xz),1)) = 0;
[U,S,V] = svd(Yz,'econ'); sig = diag(S);
G = V * diag(sig./(sig.^2 + lambda)) * U';   % (Y'Y+λI)^{-1}Y'
T = G * Xz;                                   % M×D （标准化域下；用于能量与写回一致性足够）
end

function O_groups = os_group_by_objective_builtin(Y, K)
% 仅用 Y 的谱聚类（非负 cosine），含 NaN 清理与 K 钳制
[~,M] = size(Y);
if M==1 || K<=1, O_groups = {1:M}; return; end
K = min(K, max(1, M-1));

Yz = zscore(Y,0,1); Yz(:,any(isnan(Yz),1))=0;
nrm = sqrt(sum(Yz.^2,1)); nrm(nrm==0)=1;
U = Yz ./ nrm;
S = U.'*U;  S(1:M+1:end) = 0;

A = max(S,0);  A = A - diag(diag(A));
d = sum(A,2);  Dm = diag(d + eps);
Lsym = eye(M) - Dm^(-1/2)*A*Dm^(-1/2);
Lsym = (Lsym + Lsym.')/2;

[V,E] = eig(Lsym);
[~,ord] = sort(diag(E),'ascend');
H  = V(:,ord(1:K));
Hn = H ./ max(sqrt(sum(H.^2,2)), eps);
Hn(~isfinite(Hn)) = 0;

opts = statset('MaxIter',200,'Display','off');
repl = max(5, min(10, M-1));
lbl = kmeans(Hn, K, 'Replicates', repl, 'Options', opts);

O_groups = cell(1,K);
for k = 1:K, O_groups{k} = find(lbl==k).'; end
end

function Yk_new = ae_pca_generate(Yk, kcap, kfrac, F, Cr)
% 线性 AE (PCA) + 一轮 DE
[Ykz, mu, sg] = safe_zscore(Yk);
r  = max(1, min([size(Yk,2), max(1, ceil(size(Yk,2)*kfrac)), kcap]));
W  = pca_basis(Ykz, r);
Z  = Ykz * W;
Z2 = latent_DE(Z, F, Cr);
Yk_new = (Z2 * W') .* sg + mu;
end

function [Z,mu,sg] = safe_zscore(X)
mu = mean(X,1);
sg = std(X,0,1); sg(sg==0) = 1;
Z = (X - mu) ./ sg;
end

function W = pca_basis(X, k)
[~,~,V] = svd(X,'econ');
W = V(:,1:k);
end

function Znew = latent_DE(Z, F, Cr)
[N,k] = size(Z);
Znew = Z;
if N<4 || k==0, return; end
idx = 1:N;
for i = 1:N
    r = idx; r(i) = [];
    if numel(r)<3, Znew(i,:) = Z(i,:); continue; end
    r = r(randperm(numel(r),3));
    v = Z(r(1),:) + F*(Z(r(2),:) - Z(r(3),:));
    jrand = randi(k);
    mask = (rand(1,k) < Cr); mask(jrand) = true;
    u = Z(i,:); u(mask) = v(mask);
    Znew(i,:) = u;
end
end

function Population = env_select_local(Population, N)
% 非支配排序 + 角度截断
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
% 归一化 + 基于 cosine 的多样性截断
fmax = max(PopObj,[],1); fmin = min(PopObj,[],1);
span = fmax - fmin; span(span==0) = 1;
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
