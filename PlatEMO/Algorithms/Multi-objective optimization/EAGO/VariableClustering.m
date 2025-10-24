function [PV,DV] = VariableClustering(Problem, Population, nSel, nPer)
% Detect the kind of each decision variable (LMEA 风格的变量聚类)
%
% 输入:
%   Problem     PlatEMO 的 Problem 对象（含 lower/upper/Evaluation 等）
%   Population  INDIVIDUAL 数组
%   nSel        抽样的基准个体个数
%   nPer        每个基准个体在单变量扰动下的样本数
%
% 输出:
%   PV          被判定为“多样性”变量（Passive / PV）
%   DV          被判定为“收敛”变量（Determining / DV）

    [N, D] = size(Population.decs);

    % 使用非支配前沿归一化
    ND    = NDSort(Population.objs,1) == 1;
    fmin  = min(Population(ND).objs,[],1);
    fmax  = max(Population(ND).objs,[],1);
    if any(fmax==fmin)
        fmax = ones(size(fmax));
        fmin = zeros(size(fmin));
    end
    span = fmax - fmin;  span(span==0) = 1;

    %% 统计量容器
    Angle  = zeros(D, nSel);
    RMSE   = zeros(D, nSel);

    %% 选 nSel 个基准个体
    Sample = randi(N, 1, nSel);

    for i = 1:D
        drawnow();

        % —— 构造 (nSel * nPer) 组单变量扰动解：对第 i 维重采样
        Decs      = repmat(Population(Sample).decs, nPer, 1);                          % [nSel*nPer × D]
        Decs(:,i) = unifrnd(Problem.lower(i), Problem.upper(i), size(Decs,1), 1);

        % —— 统一用 Problem.Evaluation 评估（自动计数）
        newPopu   = Problem.Evaluation(Decs);                                          % INDIVIDUAL 数组

        % —— 对每个基准体分别做 PCA-1 拟合，统计 Angle & RMSE
        for j = 1:nSel
            idx    = j:nSel:size(Decs,1);                                             % 取该基准体的 nPer 个点
            Points = newPopu(idx).objs;                                               % [nPer × M]

            % 归一化 + 去均值
            Pn = (Points - fmin) ./ span;
            Pn = Pn - mean(Pn,1);

            % 主方向（SVD 第一右奇异向量）
            [~,~,V] = svd(Pn,'econ');
            v1 = V(:,1)';  v1 = v1 ./ max(norm(v1),eps);

            % RMSE 到该直线
            proj  = sum(Pn .* v1, 2);
            resid = Pn - proj .* v1;
            RMSE(i,j) = sqrt(mean(sum(resid.^2,2)));

            % 与 (1,1,...,1) 超平面的夹角（度）
            normal = ones(1,size(v1,2));
            sine   = abs(sum(v1.*normal,2)) / (norm(v1)*norm(normal));
            Angle(i,j) = real(asin(min(max(sine,0),1)) / pi * 180);
        end
    end

    %% 判别：先按 RMSE 过滤，再对 Angle 聚类成两类，选更“陡”的那一类为 DV
    VariableKind = (mean(RMSE,2) < 1e-2)';                            % 候选“收敛变量”
    result       = kmeans(Angle, 2)';                                 % 两簇
    if any(result(VariableKind)==1) && any(result(VariableKind)==2)
        m1 = mean(mean(Angle(result==1 & VariableKind, :)));
        m2 = mean(mean(Angle(result==2 & VariableKind, :)));
        if m1 > m2
            VariableKind = VariableKind & (result==1);
        else
            VariableKind = VariableKind & (result==2);
        end
    end

    PV = find(~VariableKind);     % 多样性变量
    DV = find(VariableKind);      % 收敛变量
end
