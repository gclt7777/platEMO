function CA = UpdateCA_tree(CA, Union, MaxSize, zmin, zmax)
% CA（收敛+分布）：VaEA 环境选择
% - 先装满若干完整非支配层
% - 对最后一层：最大向量角优先 + 劣者淘汰（阈值 pi/2/(K+1)）

    if isempty(Union)
        CA = SOLUTION.empty(); return;
    end

    Pop = Union;
    Y   = Pop.objs;
    Yn  = TreeUtils.NormalizeObjs(Y, zmin, zmax);

    % 非支配分层
    [FrontNo, ~] = NDSort(Y, inf);
    CA = SOLUTION.empty();
    cur = 1; remain = MaxSize;

    while remain > 0 && any(FrontNo==cur)
        idx = find(FrontNo==cur);
        if numel(idx) <= remain
            CA = [CA, Pop(idx)];
            remain = remain - numel(idx);
            cur = cur + 1;
        else
            % —— 最后一层：VaEA 角度选择 + 劣者淘汰 —— 
            Psel   = find(FrontNo < cur);
            K      = remain;
            choose = VaEA_select_last(Yn, Psel, idx, K);
            CA     = [CA, Pop(choose)];
            break;
        end
    end
end

%% ========= 最后一层：最大角优先 + 劣者淘汰（安全删除） =========
function choose = VaEA_select_last(Yn, preSel, cand, K)
    S = preSel(:).';               % 已选索引（可能为空）
    R = cand(:).';                 % 候选池
    choose = [];

    if K <= 0 || isempty(R)
        choose = R(1:0); return;
    end

    thr = pi/(2*(K+1));            % 劣者淘汰触发阈值

    while numel(choose) < K && ~isempty(R)
        % 1) 角最大优先
        theta = angle_to_set(Yn, R, [S, choose]);
        [~,pos] = max(theta);
        rstar = R(pos);
        choose(end+1) = rstar;     %#ok<AGROW>
        R(pos) = [];

        if numel(choose) >= K || isempty(R)
            break;
        end

        % 2) 劣者淘汰：若最小角很小，则允许用“更收敛”的个体替换
        theta_rest = angle_to_set(Yn, R, [S, choose]);
        if isempty(theta_rest)
            break;
        end
        [tmin, mu] = min(theta_rest);
        if tmin < thr
            csum  = sum(Yn(R,:), 2);               % 收敛：归一化目标和
            [~, ibest] = min(csum);
            if csum(mu) > csum(ibest)
                choose(end+1) = R(ibest);          %#ok<AGROW>
                dels = sort([ibest, mu], 'descend');
                for d = dels
                    if d>=1 && d<=numel(R), R(d) = []; end
                end
                if numel(choose) >= K || isempty(R)
                    choose = choose(1:min(K,numel(choose)));
                    break;
                end
            end
        end
    end

    % 未凑满则顺序补齐
    if numel(choose) < K && ~isempty(R)
        need = K - numel(choose);
        choose = [choose, R(1:min(need, numel(R)))];
    end
end

%% ========= 与集合的最小夹角（弧度） =========
function theta = angle_to_set(Yn, candIdx, setIdx)
    C = Yn(candIdx,:); 
    C = max(C, 1e-12);
    C = C ./ vecnorm(C,2,2);

    if isempty(setIdx)
        % 与坐标轴的最小夹角
        cosv  = max(min(C,1),-1);
        theta = min(acos(cosv),[],2).';
        return;
    end

    S = Yn(setIdx,:);
    S = max(S, 1e-12);
    S = S ./ vecnorm(S,2,2);
    cosv = C * S.';
    cosv = max(min(cosv,1),-1);
    ang  = acos(cosv);
    theta = min(ang,[],2).';
end
