classdef OSAE_RCD_L < ALGORITHM
% <2025> <multi/many> <real/integer/label/binary/permutation> <large>
% OSAE_RCD_L : Row–Column Dual-Drive for O–S–D（内置 LMOCSO/RVEA-APD 环境选择，零外部依赖）
%
% - 环境选择：参考向量分区 + APD(Angle-Penalized Distance) + 可行性优先
% - 参考向量最小夹角 gamma 只算一次并缓存
% - theta 线性调度：FE/maxFE
% - C-phase：收敛导向 DE(rand/1/bin)
% - D-phase：多样性导向的全局高斯探索（带边界裁剪）
%
% 示例：
%   main('-algorithm',@OSAE_RCD_L,'-problem',@DTLZ2,'-M',10,'-D',200,...
%        '-N',275,'-evaluation',1e5);

    properties (Access = private)
        RVEA_V        % 参考向量 (NV x M)
        RVEA_gamma    % 最小夹角 (NV x 1)
    end

    methods
        function main(obj, Problem)
            %% 初始化
            Population = Problem.Initialization();

            %% 初始化参考向量与 gamma（只做一次）
            if isempty(obj.RVEA_V)
                [V,~] = UniformPoint(Problem.N, Problem.M);
                V     = V./sqrt(sum(V.^2,2));      % 单位化
                obj.RVEA_V = V;

                cosVV = 1 - pdist2(V,V,'cosine');
                cosVV(1:size(cosVV,1)+1:end) = 0;  % 对角置零
                cosVV = max(-1,min(1,cosVV));      % 数值裁剪
                obj.RVEA_gamma = min(acos(cosVV),[],2);
            end

            %% 进化循环
            while obj.NotTerminated(Population)
                % 两相生成（占位已实现：可直接跑；你也可以替换为自己的 C/D 逻辑）
                OffspringC = obj.c_phase_generate(Population, Problem);
                OffspringD = obj.d_phase_generate(Population, Problem);

                % θ（角度惩罚系数）线性调度
                maxFE = max(Problem.maxFE, eps);
                theta = min(1, Problem.FE / maxFE);

                % LMOCSO / RVEA-APD 风格环境选择（鲁棒版）
                Population = obj.env_select_lmocso([Population, OffspringC, OffspringD], ...
                                   Problem.N, obj.RVEA_V, obj.RVEA_gamma, theta);
            end
        end
    end

    %% =========================  两相生成  =========================
    methods (Access = private)
        function Offspring = c_phase_generate(~, Population, Problem)
            % 列相位 C-phase：收敛导向 —— DE(rand/1/bin)
            Offspring = local_de_rand_1_bin(Population, Problem, 0.7, 0.9);
        end

        function Offspring = d_phase_generate(~, Population, Problem)
            % 行相位 D-phase：多样性导向 —— 全局高斯探索（围绕种群均值 + 随机个体）
            Offspring = local_gaussian_explore(Population, Problem, 0.15);
        end
    end

    %% ====================  环境选择（LMOCSO/RVEA-APD 鲁棒版） ====================
    methods (Access = private)
        function Population = env_select_lmocso(~, PopAll, N, V, gamma, theta)
            % 环境选择：RVEA/APD（LMOCSO 风格）+ 可行性优先 + 每向量最多选 1（鲁棒实现）

            % 空候选直接返回
            if isempty(PopAll)
                Population = PopAll;
                return;
            end

            % 仅第一前沿；若无 rank-1，退而求 rank 最小层
            FrontNo = NDSort(PopAll.objs, 1);
            Pop     = PopAll(FrontNo == 1);
            if isempty(Pop)
                fmin = min(FrontNo);
                Pop  = PopAll(FrontNo == fmin);
            end

            % 若仍空，兜底：从 PopAll 取前 N 个
            if isempty(Pop)
                k = min(N, length(PopAll));
                Population = PopAll(1:k);
                return;
            end

            % 基本量
            Nf     = length(Pop);                % 用 length 兼容 1xN / Nx1
            PopObj = Pop.objs;
            if isempty(PopObj)
                Population = Pop(1:min(N, Nf));
                return;
            end
            [~, M] = size(PopObj);

            % 目标平移（列最小值到 0）
            mins   = min(PopObj,[],1);
            PopObj = PopObj - mins;

            % 约束违反度（可行优先）
            CV = sum(max(0, Pop.cons), 2);
            if isempty(CV) || numel(CV) ~= Nf
                CV = zeros(Nf,1);
            end

            % 计算解-向量夹角并归属（数值鲁棒）
            cos_xv = 1 - pdist2(PopObj, V, 'cosine');   % 可能出现 NaN（零向量）
            cos_xv = max(-1, min(1, cos_xv));
            cos_xv(isnan(cos_xv)) = -1;                 % 把 NaN 当作最坏角度
            Angle  = acos(cos_xv);                      % Nf x NV
            [~, associate] = min(Angle, [], 2);

            NV   = size(V,1);
            Next = zeros(1, NV);

            % 逐向量择优
            uniq = unique(associate(:))';
            for i = uniq
                idx    = find(associate==i);
                idxFea = idx(CV(idx)==0);
                idxInf = idx(CV(idx)~=0);

                if ~isempty(idxFea)
                    gi    = max(gamma(i), 1e-12);       % 防止除零
                    ang   = Angle(idxFea, i);
                    normf = sqrt(sum(PopObj(idxFea,:).^2, 2));
                    APD   = (1 + M*theta*(ang/gi)) .* normf;
                    [~,best] = min(APD);
                    Next(i)  = idxFea(best);
                elseif ~isempty(idxInf)
                    [~,best] = min(CV(idxInf));
                    Next(i)  = idxInf(best);
                end
            end

            pick = Next(Next~=0);

            % 与 N 对齐的补齐策略（先可行按 ||f||2，再不可行按 CV）
            if numel(pick) ~= N
                selected = false(1, Nf);
                selected(pick) = true;

                if numel(pick) < N
                    remain = find(~selected);
                    if ~isempty(remain)
                        feasibleRemain = remain(CV(remain)==0);
                        if ~isempty(feasibleRemain)
                            nf = sqrt(sum(PopObj(feasibleRemain,:).^2, 2));
                            [~,ord] = sort(nf, 'ascend');
                            feasibleRemain = feasibleRemain(ord);
                            need = min(N - numel(pick), numel(feasibleRemain));
                            if need > 0
                                addIdx = feasibleRemain(1:need);
                                pick = [pick, addIdx]; %#ok<AGROW>
                                selected(addIdx) = true;
                            end
                        end
                    end
                end

                if numel(pick) < N
                    remain = find(~selected);
                    if ~isempty(remain)
                        infeasibleRemain = remain(CV(remain)~=0);
                        if ~isempty(infeasibleRemain)
                            [~,ord] = sort(CV(infeasibleRemain), 'ascend');
                            infeasibleRemain = infeasibleRemain(ord);
                            need = min(N - numel(pick), numel(infeasibleRemain));
                            if need > 0
                                addIdx = infeasibleRemain(1:need);
                                pick = [pick, addIdx]; %#ok<AGROW>
                                selected(addIdx) = true;
                            end
                        end
                    end
                end
            end

            % 最终索引清洗 & 填充
            pick = pick(:)';
            pick = pick(isfinite(pick));                 % 去 NaN/Inf
            pick = pick(pick == floor(pick));            % 必须是整数
            pick = pick(pick > 0 & pick <= Nf);          % 去 0/越界
            pick = unique(pick,'stable');                % 去重

            if isempty(pick)
                pick = 1:min(N, Nf);
            elseif numel(pick) < N
                rep = pick;
                if isempty(rep), rep = 1:min(N, Nf); end
                while numel(pick) < N
                    pick(end+1) = rep(mod(numel(pick), numel(rep)) + 1); %#ok<AGROW>
                end
            elseif numel(pick) > N
                pick = pick(1:N);
            end

            % 再次范围防御
            pick(pick < 1)  = 1;
            pick(pick > Nf) = Nf;

            Population = Pop(pick);
        end
    end
end

%% =========================  本地算子：DE(rand/1/bin)  =========================
function Offspring = local_de_rand_1_bin(Population, Problem, F, Cr)
% 轻量 DE(rand/1/bin)，不依赖 PopDec/GA_Mutation
if nargin < 3, F = 0.7;  end
if nargin < 4, Cr = 0.9; end

Dec = local_get_dec(Population, Problem);
[N, D] = size(Dec);
lower = Problem.lower(:)';
upper = Problem.upper(:)';

% 极端小规模兜底：随机采样
if N < 4
    lb = repmat(lower, N, 1);
    ub = repmat(upper, N, 1);
    RandDec = lb + rand(N,D).*(ub - lb);
    Offspring = Problem.Evaluation(RandDec);
    return;
end

Off = zeros(N, D);
for i = 1:N
    % 3 个互异索引且不含 i
    perm = randperm(N);
    perm(perm==i) = [];
    r = perm(1:3);
    % 变异：v = x_r1 + F*(x_r2 - x_r3)
    v = Dec(r(1),:) + F*(Dec(r(2),:) - Dec(r(3),:));

    % 交叉（bin）
    jrand = randi(D);
    u = Dec(i,:);
    mask = (rand(1,D) <= Cr);
    mask(jrand) = true;
    u(mask) = v(mask);

    % 边界裁剪
    u = min(max(u, lower), upper);
    Off(i,:) = u;
end

Offspring = Problem.Evaluation(Off);
end

%% =========================  本地算子：全局高斯探索  =========================
function Offspring = local_gaussian_explore(Population, Problem, sigma)
% 多样性导向：围绕种群重心与随机父代的混合高斯扰动
if nargin < 3, sigma = 0.15; end

Dec = local_get_dec(Population, Problem);
[N, D] = size(Dec);
lower = Problem.lower(:)';
upper = Problem.upper(:)';

mu   = mean(Dec, 1);                       % 重心
mate = Dec(randi(N, N, 1), :);             % 随机配偶
alpha = rand(N,1);                         % 线性混合系数
base  = alpha.*mate + (1-alpha).*mu;

% 自适应尺度：按边界范围
range = (upper - lower);
range(range==0) = 1;                       % 防除零
noise = randn(N, D) .* (sigma .* range);

Off = base + noise;
Off = min(max(Off, lower), upper);

Offspring = Problem.Evaluation(Off);
end

%% =========================  安全获取决策矩阵  =========================
function Dec = local_get_dec(Population, Problem)
% 兼容不同 PlatEMO 版本，安全获取决策矩阵
try
    % 如果环境里有 PopDec.m，则直接调用
    Dec = PopDec(Population);
    if isempty(Dec)
        error('PopDec returned empty');
    end
catch
    % 没有 PopDec：尝试对象属性
    if isprop(Population, 'decs')
        Dec = Population.decs;
    elseif isprop(Population, 'dec')
        % INDIVIDUAL 数组：把 [1xN] 结构拼成 [N x D]
        Dec = vertcat(Population.dec);
    else
        % 最后兜底：随机初始化以保证不断代（不建议长期使用）
        N = numel(Population);
        D = Problem.D;
        lb = repmat(Problem.lower, N, 1);
        ub = repmat(Problem.upper, N, 1);
        Dec = lb + rand(N,D).*(ub - lb);
    end
end
end
