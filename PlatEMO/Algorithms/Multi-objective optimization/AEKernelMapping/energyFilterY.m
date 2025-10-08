function idx = energyFilterY(model, Ycand, opts)
% ENERGYFILTERY  依据“局部能量 ~ 重构误差”筛掉虚空点（可选）
%   idx = energyFilterY(model, Ycand, opts)
%   输入:
%     model : trainDAE_walkback 的返回结构体
%     Ycand : K x M 候选目标向量
%     opts  : .keepQuantile 保留的分位数阈值（默认 0.7 = 保留重构误差较低的前 70%）
%   输出:
%     idx   : 逻辑索引（true 表示保留）
%
% 说明:
%   对每个候选 y，先按模型噪声库腐蚀一次 ytil，再去噪得到 yrec，
%   用 MSE(yrec, y) 作为“局部能量/不可信度”，阈值化筛选。
arguments
    model struct
    Ycand double
    opts.keepQuantile (1,1) double = 0.7
end

K = size(Ycand,1);
errs = zeros(K,1);
for k = 1:K
    sigmas = model.noisePool(randi(size(model.noisePool,1)),:);
    y = Ycand(k,:);
    ytil = y + randn(1,size(Ycand,2)) .* sigmas;
    yrec = model.decodeFcn(ytil);
    errs(k) = mean((yrec - y).^2);
end

th = quantile(errs, opts.keepQuantile);
idx = errs <= th;
end
