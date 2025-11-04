function  [dis_y_par,dis_y_chi] = MED(Population,par,child)
% Calculate the multiplicative Euclidean distance used in LSMaODE.

    if isempty(Population)
        dis_y_par = 0;
        dis_y_chi = 0;
        return;
    end

    Population_obj = Population.objs;
    child_obj      = child.objs;
    par_obj        = par.objs;

    dist_child = sqrt(sum((Population_obj - child_obj).^2,2));
    dist_par   = sqrt(sum((Population_obj - par_obj).^2,2));

    dist_child = sort(dist_child(:));
    dist_par   = sort(dist_par(:));

    sum_child = sum(dist_child);
    sum_par   = sum(dist_par);

    if isempty(dist_child)
        near_child = 0;
    else
        near_child = dist_child(1);
    end
    if isempty(dist_par)
        near_par = 0;
    else
        near_par = dist_par(1);
    end

    dis_y_chi = sum_child * near_child;
    dis_y_par = sum_par * near_par;
end
