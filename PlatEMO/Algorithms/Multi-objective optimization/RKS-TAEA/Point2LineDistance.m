function d = Point2LineDistance(P,A,B)

    d = zeros(size(P,1),1);
    for i = 1 : size(P,1)
        pa = P(i,:) - A;
        ba = B - A;
        denom = dot(ba,ba);
        if denom == 0
            d(i,1) = norm(pa,2);
        else
            t = dot(pa,ba)/denom;
            d(i,1) = norm(pa - t*ba,2);
        end
    end
end
