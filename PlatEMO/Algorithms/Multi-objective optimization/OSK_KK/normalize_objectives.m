
function [Yz, info] = normalize_objectives(Y, opts)
% Normalize objectives with optional direction unification.
% opts.mode: 'zscore' (default) or 'minmax'
% opts.maximize_idx: indices of objectives that are to be maximized (will flip sign)
if nargin<2 || isempty(opts), opts = struct(); end
if ~isfield(opts,'mode'), opts.mode = 'zscore'; end
if ~isfield(opts,'maximize_idx'), opts.maximize_idx = []; end

Y = double(Y);
info = struct();
info.maximize_idx = opts.maximize_idx(:)';
Yz = Y;
% Unify direction: maximize -> minimize by flipping sign
if ~isempty(info.maximize_idx)
    Yz(:,info.maximize_idx) = -Yz(:,info.maximize_idx);
end

switch lower(opts.mode)
    case 'zscore'
        mu = mean(Yz,1);
        sigma = std(Yz,0,1);
        sigma(sigma==0) = 1; % avoid div-by-zero
        Yz = (Yz - mu) ./ sigma;
        info.mode  = 'zscore';
        info.mu    = mu;
        info.sigma = sigma;
    case 'minmax'
        mn = min(Yz,[],1);
        mx = max(Yz,[],1);
        den = mx-mn; den(den==0) = 1;
        Yz = (Yz - mn) ./ den;
        info.mode = 'minmax';
        info.mn   = mn;
        info.mx   = mx;
    otherwise
        info.mode = 'none';
end
info.flipped = ~isempty(info.maximize_idx);
end
