function EntropyQ = CalculateEntropy(UQpi,ConnectZNum,pi,N)

    if ConnectZNum == 0 || N == 0
        EntropyQ = 0;
        return;
    end

    EachZnum = zeros(1,ConnectZNum);
    for i = 1 : ConnectZNum
        EachZnum(i) = sum(pi == UQpi(i));
    end

    EntropyNow = 0;
    for i = 1 : ConnectZNum
        ratio = EachZnum(i) / N;
        if ratio > 0
            EntropyNow = EntropyNow - ratio * log2(ratio);
        end
    end

    bestZnum   = floor(N/ConnectZNum);
    remainZnum = N - bestZnum*ConnectZNum;
    EntropyBest = 0;
    if bestZnum > 0
        EntropyBest = EntropyBest - (ConnectZNum-remainZnum)*(bestZnum/N)*log2(bestZnum/N);
    end
    if remainZnum > 0
        EntropyBest = EntropyBest - remainZnum*((bestZnum+1)/N)*log2((bestZnum+1)/N);
    end

    if EntropyBest == 0
        EntropyQ = 0;
    else
        EntropyQ = EntropyNow/EntropyBest;
    end
end
