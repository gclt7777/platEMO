
function Y = unnormalize_objectives(Yz, info)
% Inverse of normalize_objectives
Y = Yz;
if isfield(info,'mode')
    switch lower(info.mode)
        case 'zscore'
            Y = Y.*info.sigma + info.mu;
        case 'minmax'
            den = info.mx - info.mn; den(den==0)=1;
            Y = Y.*den + info.mn;
    end
end
if isfield(info,'maximize_idx') && ~isempty(info.maximize_idx)
    Y(:,info.maximize_idx) = -Y(:,info.maximize_idx);
end
end
