
function K = rbf_kernel(A, B, sigma)
% RBF kernel matrix. If sigma is empty or <=0, use median heuristic.
A = double(A); B = double(B);
if nargin<3 || isempty(sigma) || sigma<=0
    % median heuristic on combined set
    C = [A; B];
    if size(C,1) > 2000
        idx = randperm(size(C,1), 2000);
        C = C(idx,:);
    end
    D = pdist2(C, C, 'euclidean');
    d = median(D(:));
    if d<=0, d = 1; end
    sigma = d/sqrt(2);
end
D2 = pdist2(A, B, 'euclidean').^2;
K = exp(-D2/(2*sigma^2));
end
