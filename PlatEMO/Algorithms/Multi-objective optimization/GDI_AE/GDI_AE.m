classdef GDI_AE < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation>
% GDI_AE : Group (G/D/I) + AE-in-decision-subspace generation (PCA as AE)
% - 逐变量打标签 G/D/I（内联 GDI_Grouping_local）
% - 在每个组的决策子空间用 PCA 当 AE，潜空间做一轮 DE 采样
% - 仅回写该组的列，评估再环境选择（内联 env_select_local）
%
% 用法示例：
%   main('-algorithm',@GDI_AE,'-problem',@DTLZ2,'-M',3,'-D',12,'-N',92,'-evaluation',5e4);

    methods
        function main(obj, Problem)
            % 超参数（可在命令行覆盖）
            [nSel,nPer,lambda,relax,eta,kcap,kfrac,F,Cr,period] = ...
                obj.ParameterSet(5,30,1e-5,0.9,0.5,4,1/3,0.7,0.9,5);

            % 初始化
            Population = Problem.Initialization();
            gen = 1;

            % 首次分组
            [G,D,I] = GDI_Grouping_local(Problem, Population, nSel, nPer, lambda, relax);

            while obj.NotTerminated(Population)
                % 周期性重分组（或你也可以加代际距离触发）
                if gen == 1 || mod(gen-1,period)==0
                    [G,D,I] = GDI_Grouping_local(Problem, Population, nSel, nPer, lambda, relax);
                end

                % —— AE（PCA）按组生成并仅回写该组列
                X      = Population.decs;
                OffDec = X;
                groups = {G,D,I};
                for gi = 1:numel(groups)
                    S = groups{gi};
                    if isempty(S), continue; end
                    XS = X(:,S);
                    dS = size(XS,2);
                    if dS==0, continue; end

                    % 线性 AE: PCA 基 + 编码
                    [XSz, mu, sg] = safe_zscore(XS);
                    k  = max(1, min([dS, max(1, ceil(dS*kfrac)), kcap]));
                    Wk = pca_basis(XSz, k);        % dS×k
                    Z  = XSz * Wk;                 % N×k

                    % 潜空间 DE/rand/1/bin
                    Znew  = latent_DE(Z, F, Cr);

                    % 解码并反标准化
                    XShat = (Znew * Wk') .* sg + mu;

                    % 半步回写（只改当前组）
                    OffDec(:,S) = (1-eta)*XS + eta*XShat;
                end

                % 边界裁剪 + 评估 + 环境选择
                OffDec    = min(max(OffDec, Problem.lower), Problem.upper);
                Offspring = Problem.Evaluation(OffDec);
                Population = env_select_local([Population, Offspring], Problem.N);

                gen = gen + 1;
            end
        end
    end
end

% ======================= 内联工具函数 =======================

function [G,D,I] = GDI_Grouping_local(Problem, Population, nSel, nPer, lambda, relax)
% 逐变量贴 G/D/I 标签（EAGO-Algorithm4 风格）
X = Population.decs;   Y = Population.objs;
[N,D] = size(X); M = size(Y,2); %#ok<ASGLU>

% 岭回归拟合线性逆映射（稳）
T = (Y.'*Y + lambda*eye(M)) \ (Y.'*X);    % M×D

isD = false(1,D); isI = false(1,D);
for i = 1:D
    Sample = randi(Problem.N,1,nSel);
    votes  = zeros(1,nSel);   % 0:G, 1:D, 2:I
    for j = 1:nSel
        baseX = repmat(Population(Sample(j)).decs, nPer, 1);
        baseY = repmat(Population(Sample(j)).objs, nPer, 1);

        % G: 随机采样第 i 列
        Xg = baseX;
        lb = Problem.lower(i); ub = Problem.upper(i);
        Xg(:,i) = lb + (ub-lb)*rand(nPer,1);
        Yg = Problem.Evaluation(Xg).objs; Yg_bar = mean(Yg,1);

        % D: 目标减小 → 只映回第 i 列
        Yd   = baseY - baseY.*rand(nPer,M)*0.25;
        Xi_d = Yd * T(:,i);
        Xd   = baseX; Xd(:,i) = min(max(Xi_d, lb), ub);
        Yd   = Problem.Evaluation(Xd).objs; Yd_bar = mean(Yd,1);

        % I: 目标增大 → 只映回第 i 列
        Yi   = baseY + baseY.*rand(nPer,M)*0.25;
        Xi_i = Yi * T(:,i);
        Xi   = baseX; Xi(:,i) = min(max(Xi_i, lb), ub);
        Yi   = Problem.Evaluation(Xi).objs; Yi_bar = mean(Yi,1);

        sg = sum(Yg_bar.^2); sd = sum(Yd_bar.^2); si = sum(Yi_bar.^2);
        if (sd*relax <= sg) && (sd <= si)
            votes(j)=1;   % D
        elseif (si*relax <= sg) && (si < sd)
            votes(j)=2;   % I
        else
            votes(j)=0;   % G
        end
    end
    if sum(votes==1) >= sum(votes==2) && sum(votes==1) >= sum(votes==0), isD(i)=true;
    elseif sum(votes==2) > sum(votes==1) && sum(votes==2) >= sum(votes==0), isI(i)=true;
    end
end
D = find(isD); I = find(isI); G = find(~(isD|isI));
end

function [Z,mu,sg] = safe_zscore(X)
mu = mean(X,1);
sg = std(X,0,1);
sg(sg==0) = 1;
Z = (X - mu) ./ sg;
end

function Wk = pca_basis(X, k)
% X 已标准化；返回前 k 个主方向（列正交）
[~,~,V] = svd(X,'econ');
Wk = V(:,1:k);
end

function Znew = latent_DE(Z, F, Cr)
% 一次性在潜空间做 DE/rand/1/bin
[N,k] = size(Z);
Znew = Z;
idx = 1:N;
for i = 1:N
    r = idx; r(i) = [];
    r = r(randperm(N-1,3));
    r1 = r(1); r2 = r(2); r3 = r(3);
    v = Z(r1,:) + F*(Z(r2,:) - Z(r3,:));
    jrand = randi(k);
    cross = (rand(1,k) < Cr); cross(jrand) = true;
    u = Z(i,:); u(cross) = v(cross);
    Znew(i,:) = u;
end
end

function Population = env_select_local(Population, N)
% 非支配排序 + 角度截断（不依赖外部 ES 文件）
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
% 归一化 + 余弦相似度为基础的多样性截断
fmax = max(PopObj,[],1); fmin = min(PopObj,[],1);
span = fmax - fmin; span(span==0) = 1;
P = (PopObj - fmin) ./ span;
% 计算单位向量的点积矩阵（=cosine 相似度）
nrm = sqrt(sum(P.^2,2)); nrm(nrm==0)=1;
U = P ./ nrm;
Cosine = U*U.';
Cosine(1:size(Cosine,1)+1:end) = 0;

Choose = false(1,size(P,1));
% 每个目标的极端点优先
[~,extreme] = max(P,[],1); Choose(extreme) = true;

if sum(Choose) > K
    sel = find(Choose);
    Choose = false(1,size(P,1));
    Choose(sel(randperm(numel(sel),K))) = true;
else
    while sum(Choose) < K
        unSel = find(~Choose);
        % 选择与已选集合相似度最大的最小者（即最不相似的）
        [~,x] = min(max(Cosine(~Choose,Choose),[],2));
        Choose(unSel(x)) = true;
    end
end
end
