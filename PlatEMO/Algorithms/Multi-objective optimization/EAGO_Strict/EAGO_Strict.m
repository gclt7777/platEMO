classdef EAGO_Strict < ALGORITHM
% <multi> <real/integer> <large/none>
% EAGO_Strict  Objective-Space population generation (strict mode per paper)
% - Linear mapping: T = pinv(Y)*X (no zscore/no ridge), update every even gen
% - Three Y-ops: cross-mutation, decrease, increase
% - Two-phase loop (Algorithm 1): odd gen uses fixed T0 on CV/DV; even gen uses updated T with G/D/I

    methods
        function main(Algorithm,Problem)
            %% Params
            [ns, nd, cfl, cfh, eta_c, eta_m, nSel, nPer, mseThr] = ...
                Algorithm.ParameterSet(5,30,0.01,10,20,20,2,4,1e-2);

            %% Init
            Population = Problem.Initialization();
            N  = Problem.N; D = Problem.D;
            lb = Problem.lower; ub = Problem.upper;
            X  = Population.decs; Y  = Population.objs;

            % Initial T0 (strict)
            T0 = pinv(Y) * X;
            T  = T0;
            gen = 1;

            % LMEA variable clustering (CV/DV) —— 这里改为传 Problem
            [DV, CV] = EAGO_Strict.LMEA_VariableClustering(Population, Problem, nSel, nPer, mseThr);

            while Algorithm.NotTerminated(Population)
                X = Population.decs; Y = Population.objs;

                if mod(gen,2)==1
                    %% ========= Odd gen: use fixed T0 =========
                    % ConvergenceOptimize on CV (per-variable, cross-mutation)
                    for k = 1:numel(CV)
                        v = CV(k);
                        Ycm = EAGO_Strict.opY_cross_mutate(Y, eta_c, eta_m, cfl, cfh);
                        Xcm_v = Ycm * T0(:,v);
                        Xnew  = X;
                        Xnew(:,v) = 0.5*(X(:,v) + Xcm_v);
                        Xnew = EAGO_Strict.clipX(Xnew, lb, ub);
                        Off  = Problem.Evaluation(Xnew);
                        Population = EAGO_Strict.select_convergence(Population, Off);
                    end
                    % DiversityOptimize on DV
                    if ~isempty(DV)
                        Ycm = EAGO_Strict.opY_cross_mutate(Y, eta_c, eta_m, cfl, cfh);
                        Xcm = Ycm * T0(:,DV);
                        R   = rand(size(Y));
                        Xs  = (Y - Y.*R) * T0(:,DV);
                        Xdiff = 0.5*(Xcm - Xs);
                        Xnew = X;
                        Xnew(:,DV) = EAGO_Strict.scramble_average(X(:,DV));
                        Xnew(:,DV) = Xnew(:,DV) + Xdiff;
                        Xnew = EAGO_Strict.clipX(Xnew, lb, ub);
                        Off  = Problem.Evaluation(Xnew);
                        Population = EAGO_Strict.select_diversity(Population, Off);
                    end

                else
                    %% ========= Even gen: update T, analyze variables (G/D/I) =========
                    T = pinv(Y) * X;
                    [G, Dset, Iset] = EAGO_Strict.variableAnalysis_strict(Population, Problem, nd, T, lb, ub);

                    if ~isempty(Dset)
                        R = rand(size(Y));
                        Xs = (Y - Y.*R) * T(:,Dset);
                        Xnew = X; Xnew(:,Dset) = 0.5*(X(:,Dset) + Xs);
                        Xnew = EAGO_Strict.clipX(Xnew, lb, ub);
                        Off = Problem.Evaluation(Xnew);
                        Population = EAGO_Strict.select_convergence(Population, Off);
                    end
                    if ~isempty(Iset)
                        R = rand(size(Y));
                        Xp = (Y + Y.*R) * T(:,Iset);
                        Xnew = X; Xnew(:,Iset) = 0.5*(X(:,Iset) + Xp);
                        Xnew = EAGO_Strict.clipX(Xnew, lb, ub);
                        Off = Problem.Evaluation(Xnew);
                        Population = EAGO_Strict.select_convergence(Population, Off);
                    end
                    if ~isempty(G)
                        Ycm = EAGO_Strict.opY_cross_mutate(Y, eta_c, eta_m, cfl, cfh);
                        Xcm = Ycm * T(:,G);
                        R   = rand(size(Y));
                        Xs  = (Y - Y.*R) * T(:,G);
                        Xdiff = 0.5*(Xcm - Xs);
                        Xnew = X;
                        Xnew(:,G) = EAGO_Strict.scramble_average(X(:,G));
                        Xnew(:,G) = Xnew(:,G) + Xdiff;
                        Xnew = EAGO_Strict.clipX(Xnew, lb, ub);
                        Off  = Problem.Evaluation(Xnew);
                        Population = EAGO_Strict.select_diversity(Population, Off);
                    end
                end

                gen = gen + 1;
            end
        end
    end

    %% ================= helpers =================
    methods (Static, Access = private)

        %% ---- LMEA Variable Clustering (Algorithm 5) ----
        function [DV, CV] = LMEA_VariableClustering(Pop, Problem, nSel, nPer, mseThr)
            % Implements Algorithm 5 in LMEA (变量聚类)
            Xpop = Pop.decs; Ypop = Pop.objs;
            [N,D] = size(Xpop); M = size(Ypop,2);
            nSel = max(1, min(nSel, N));
            nPer = max(3, nPer);

            normalVec = ones(1,M); normalVec = normalVec / norm(normalVec);
            Angles = nan(D, nSel);
            MSEs   = nan(D, nSel);

            lb = Problem.lower(:)'; ub = Problem.upper(:)';

            for i = 1:D
                idxSel = randperm(N, nSel);
                for j = 1:nSel
                    base = Xpop(idxSel(j),:);
                    xi_vals = linspace(lb(i), ub(i), nPer);
                    Xs = repmat(base, nPer, 1);
                    Xs(:,i) = xi_vals;
                    % 这里用 Problem.CalObj（不再引用 Pop(1).Problem）
                    Ysp = Problem.CalObj(Xs);   % nPer x M

                    ymin = min(Ysp,[],1); ymax = max(Ysp,[],1);
                    rng  = max(ymax - ymin, 1e-12);
                    Ynorm = (Ysp - ymin) ./ rng;

                    Yc = bsxfun(@minus, Ynorm, mean(Ynorm,1));
                    [U, ~, ~] = svd(Yc, 'econ');
                    v = (Yc' * U(:,1));              % direction in objective space
                    v = v(:)'; if norm(v) < 1e-12, v = randn(1,M); end
                    v = v / norm(v);

                    ang = acos(max(-1,min(1, dot(v, normalVec))));
                    Angles(i,j) = ang;

                    proj1 = U(:,1) * (U(:,1)'*Yc);
                    resid = Yc - proj1;
                    MSEs(i,j) = mean(sum(resid.^2,2));
                end
            end

            meanMSE = mean(MSEs,2,'omitnan');
            cvMask = meanMSE < mseThr;

            feats = Angles;
            nanrow = any(isnan(feats),2);
            feats(nanrow,:) = pi/2;

            try
                [cid, C] = kmeans(feats, 2, 'Replicates',5, 'MaxIter',200, 'OnlinePhase','off');
            catch
                mu = mean(feats,2);
                med = median(mu);
                cid = ones(D,1); cid(mu>med) = 2;
                C = [mean(feats(cid==1,:),1); mean(feats(cid==2,:),1)];
            end

            Cmean = [mean(C(1,:)), mean(C(2,:))];
            if Cmean(1) <= Cmean(2)
                clusterCV = 1; clusterDV = 2;
            else
                clusterCV = 2; clusterDV = 1;
            end

            S1 = find(cid==1); S2 = find(cid==2);
            Cand = find(cvMask);
            if ~isempty(intersect(Cand, S1)) && ~isempty(intersect(Cand, S2))
                if clusterCV == 1, CV = intersect(Cand, S1);
                else,               CV = intersect(Cand, S2);
                end
            else
                if clusterCV == 1, CV = S1(:);
                else,               CV = S2(:);
                end
            end
            DV = setdiff((1:D)', CV);

            CV = CV(:)'; DV = DV(:)';
        end

        %% ---- selection rules ----
        function Pop = select_convergence(P, O)
            PopAll = [P,O];
            Y = PopAll.objs;
            [FrontNo,~] = NDSort(Y, length(P));
            ideal = min(Y,[],1);
            Dt = sum(abs(Y - ideal),2);
            key = [FrontNo(:), Dt(:)];
            [~,ord] = sortrows(key,[1 2]);
            Pop = PopAll(ord(1:length(P)));
        end

        function Pop = select_diversity(P, O)
            PopAll = [P,O];
            Y = PopAll.objs;
            N = length(P);
            [FrontNo,MaxF] = NDSort(Y, N);
            Next = FrontNo < MaxF;
            Rem  = find(FrontNo==MaxF);
            K = N - sum(Next);
            if K>0
                Ycand = Y(Rem,:);
                S = EAGO_Strict.angle_select_indices(Ycand, K);
                Next(Rem(S)) = true;
            end
            Pop = PopAll(Next);
        end

        function S = angle_select_indices(Y, K)
            if size(Y,1)<=K, S = (1:size(Y,1))'; return; end
            Yn = Y - min(Y,[],1);
            n = size(Yn,1);
            v = Yn ./ max(sqrt(sum(Yn.^2,2)),1e-12);
            taken = false(n,1);
            c = mean(v,1);
            d = v*c';
            [~,i0] = min(d); taken(i0)=true; S = i0;
            while numel(S) < K
                mindeg = -inf(n,1);
                for i=1:n
                    if taken(i), mindeg(i) = -inf; continue; end
                    angs = acos(max(-1,min(1, v(i,:)*v(S,:)')));
                    mindeg(i) = min(angs);
                end
                [~,ix] = max(mindeg);
                taken(ix)=true; S(end+1,1)=ix; %#ok<AGROW>
            end
        end

        %% ---- Y-space operators ----
        function Ycm = opY_cross_mutate(Y, eta_c, eta_m, cfl, cfh)
            [N,M] = size(Y);
            ymin = min(Y,[],1); ymax = max(Y,[],1);
            L = ymin*cfl; U = ymax*cfh;
            idx = randperm(N);
            p1 = Y(idx(1:floor(N/2)),:);
            p2 = Y(idx(end-floor(N/2)+1:end),:);
            nc = size(p1,1);
            u = rand(nc,M);
            beta = (u<=0.5).*(2*u).^(1/(eta_c+1)) + (u>0.5).*(1./(2*(1-u))).^(1/(eta_c+1));
            c1 = 0.5*((1+beta).*p1 + (1-beta).*p2);
            c2 = 0.5*((1-beta).*p1 + (1+beta).*p2);
            Ysbx = [c1; c2];
            if size(Ysbx,1) < N
                Ysbx = [Ysbx; Y(1:N-size(Ysbx,1),:)];
            else
                Ysbx = Ysbx(1:N,:);
            end
            Ycm = EAGO_Strict.poly_mutate(Ysbx, L, U, eta_m);
        end

        function Ymut = poly_mutate(Y, L, U, eta_m)
            [N,M] = size(Y);
            Ymut = Y;
            pm = 1/M;
            for i=1:N
                for j=1:M
                    if rand < pm
                        yl = L(j); yu = U(j); y = Ymut(i,j);
                        if yl==yu, continue; end
                        delta1 = (y-yl)/(yu-yl);
                        delta2 = (yu-y)/(yu-yl);
                        r = rand;
                        mut_pow = 1/(eta_m+1);
                        if r <= 0.5
                            xy = 1 - delta1;
                            val = 2*r + (1-2*r)*(xy^(eta_m+1));
                            deltaq = val^mut_pow - 1;
                        else
                            xy = 1 - delta2;
                            val = 2*(1-r) + 2*(r-0.5)*(xy^(eta_m+1));
                            deltaq = 1 - val^mut_pow;
                        end
                        y = y + deltaq*(yu-yl);
                        Ymut(i,j) = min(max(y,yl),yu);
                    end
                end
            end
        end

        %% ---- variable analysis (G / D / I) ----
        function [G, Dset, Iset] = variableAnalysis_strict(Pop, Problem, nd, T, lb, ub)
            X = Pop.decs; Y = Pop.objs;
            [N,D] = size(X); M = size(Y,2);
            ideal = min(Y,[],1);

            G = []; Dset = []; Iset = [];
            for i=1:D
                s = randi(N);
                yi = Y(s,:);

                S1 = zeros(nd,M); S2 = zeros(nd,M); S3 = zeros(nd,M);
                for k=1:nd
                    r1 = -1 + 2*rand(1,M);
                    r2 = rand(1,M);
                    r3 = rand(1,M);

                    X1 = EAGO_Strict.clipX((yi - yi.*r1) * T, lb, ub);
                    X2 = EAGO_Strict.clipX((yi - yi.*r2) * T, lb, ub);
                    X3 = EAGO_Strict.clipX((yi + yi.*r3) * T, lb, ub);

                    % 这里也改为 Problem.CalObj
                    y1 = Problem.CalObj(X1);
                    y2 = Problem.CalObj(X2);
                    y3 = Problem.CalObj(X3);

                    S1(k,:) = y1; S2(k,:) = y2; S3(k,:) = y3;
                end
                y1m = mean(S1,1); y2m = mean(S2,1); y3m = mean(S3,1);
                E1 = norm(y1m - ideal); E2 = norm(y2m - ideal); E3 = norm(y3m - ideal);

                if E1<=E2 && E1<=E3
                    G(end+1) = i; %#ok<AGROW>
                elseif E2<=E1 && E2<=E3
                    Dset(end+1) = i; %#ok<AGROW>
                else
                    Iset(end+1) = i; %#ok<AGROW>
                end
            end
        end

        %% ---- utils ----
        function Xavg = scramble_average(Xseg)
            N = size(Xseg,1);
            p = randperm(N);
            Xavg = 0.5*(Xseg + Xseg(p,:));
        end

        function Xc = clipX(X, lb, ub)
            [N,D] = size(X);
            l = lb; u = ub;
            if isscalar(l), l = repmat(l,1,D); end
            if isscalar(u), u = repmat(u,1,D); end
            if iscolumn(l), l = l'; end
            if iscolumn(u), u = u'; end
            if numel(l)~=D, l = repmat(l(1),1,D); end
            if numel(u)~=D, u = repmat(u(1),1,D); end
            Xc = min(repmat(u,N,1), max(repmat(l,N,1), X));
        end
    end
end
