
function K = rbf_kernel(X, X2, sigma)
if nargin < 3 || isempty(sigma)
    sigma = median_heuristic([X; X2]);
end
D2 = pdist2(X, X2, 'euclidean').^2;
K = exp(- D2 / (2*sigma^2));
end

function s = median_heuristic(Z)
if size(Z,1) > 2000
    idx = randperm(size(Z,1), 2000);
    Z = Z(idx,:);
end
D = pdist(Z, 'euclidean');
m = median(D(:));
s = max(m / sqrt(2), eps);
end
