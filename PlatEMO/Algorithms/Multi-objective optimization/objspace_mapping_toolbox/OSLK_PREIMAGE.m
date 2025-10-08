classdef OSLK_PREIMAGE < ALGORITHM
% <2025> <multi/many> <real/integer>
% Objective-space Linear/Kernel mapping + Pre-image injection for MOEA
%
% modeFlag   --- 1 --- 0: linear (ridge in Y), 1: kernel (KRR in Y)
% alpha      --- 0.6 --- Step toward predicted Y (0~1 interp, >1 extrapol.)
% lambdaY    --- 1e-3 --- Regularization for Y-space mapping
% lambdaInv  --- 1e-3 --- Regularization for inverse KRR (Y->X)
% lambdaF    --- 1e-3 --- Regularization for forward KRR (X->Y, refine)
% refineIts  --- 60  --- fmincon iterations for pre-image refinement (0=off)
% injFrac    --- 0.5 --- Fraction of population to inject each generation
% sigmaY     --- -1  --- RBF sigma in Y (<=0 -> auto/median heuristic)
% sigmaInv   --- -1  --- RBF sigma for inverse KRR (<=0 -> auto)
% sigmaF     --- -1  --- RBF sigma for forward KRR (<=0 -> auto)

    methods
        function main(Algorithm,Problem)
            [modeFlag,alpha,lambdaY,lambdaInv,lambdaF,refineIts,injFrac,sigmaY,sigmaInv,sigmaF] = ...
                Algorithm.ParameterSet(1,0.6,1e-3,1e-3,1e-3,60,0.5,-1,-1,-1);

            if sigmaY  <= 0, sigmaY  = []; end
            if sigmaInv<= 0, sigmaInv= []; end
            if sigmaF  <= 0, sigmaF  = []; end

            Population = Problem.Initialization();
            LastPop    = [];
            gen        = 1;

            while Algorithm.NotTerminated(Population)
                drawnow('limitrate');

                % 常规 GA 子代
                OffDec    = OperatorGA(Problem,Population.decs);
                Offspring = Problem.Evaluation(OffDec);

                % 目标空间生新解 -> 预像回 X（需有上一代）
                Inj = [];
                if ~isempty(LastPop)
                    X     = Population.decs;
                    Y     = Population.objs;
                    histX = LastPop.decs;
                    histY = LastPop.objs;

                    params = struct( ...
                        'mode',      OSLK_PREIMAGE.ternary(modeFlag==1,'kernel','linear'), ...
                        'alpha',     alpha, ...
                        'lambdaY',   lambdaY, ...
                        'sigmaY',    sigmaY, ...
                        'norm',      struct('mode','zscore','maximize_idx',[]), ...
                        'lambdaInv', lambdaInv, ...
                        'sigmaInv',  sigmaInv, ...
                        'lambdaF',   lambdaF, ...
                        'sigmaF',    sigmaF, ...
                        'refineIters', refineIts, ...
                        'lb',        Problem.lower, ...
                        'ub',        Problem.upper);

                    try
                        Xcand = objective_space_linear_kernel_preimage(X,Y,histY,histX,params);
                        N     = size(X,1);
                        k     = max(1, min(N, round(injFrac*N)));
                        idx   = randperm(N, k);
                        Inj   = Problem.Evaluation(Xcand(idx,:));
                    catch ME
                        warning('OSLK_PREIMAGE:InjectionSkipped', ...
                                'Injection skipped this gen: %s', ME.message);
                        % disp(getReport(ME,'basic')); % 如需调试
                        Inj = [];
                    end
                end

                % 合并并环境选择
                if isempty(Inj)
                    Combined = [Population, Offspring];
                else
                    Combined = [Population, Offspring, Inj];
                end
                Population = OSLK_PREIMAGE.NDSelect(Combined, Problem.N);

                % 更新历史
                LastPop = Population;
                gen = gen + 1;
            end
        end
    end

    methods(Static, Access=private)
        function Pop = NDSelect(PopAll, N)
            Objs = PopAll.objs;
            [FrontNo,MaxFNo] = NDSort(Objs, N);
            Next = FrontNo < MaxFNo;
            Pop  = PopAll(Next);
            if sum(Next) < N
                Last  = find(FrontNo == MaxFNo);
                CD    = OSLK_PREIMAGE.CrowdingDistance(Objs(Last,:));
                [~,ord] = sort(CD,'descend');
                Pop  = [Pop, PopAll(Last(ord(1:(N - sum(Next)))) )];
            end
        end

        function CD = CrowdingDistance(Objs)
            n = size(Objs,1);
            M = size(Objs,2);
            if n==0, CD = []; return; end
            if n==1, CD = inf; return; end
            CD = zeros(n,1);
            Fmin = min(Objs,[],1);
            Fmax = max(Objs,[],1);
            Den  = max(Fmax - Fmin, 1e-12);
            for m = 1:M
                [~,ord] = sort(Objs(:,m));
                CD(ord(1))   = inf;
                CD(ord(end)) = inf;
                for i = 2:n-1
                    CD(ord(i)) = CD(ord(i)) + ...
                        (Objs(ord(i+1),m) - Objs(ord(i-1),m)) / Den(m);
                end
            end
        end

        function y = ternary(c,a,b)
            if c, y = a; else, y = b; end
        end
    end
end
