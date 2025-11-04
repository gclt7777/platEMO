    function  [dis_y_par,dis_y_chi] = MED(Population,par,child)
  
        Population_obj=Population.objs;
        Population_dec=Population.decs;
        child_obj=child.objs;
        child_dec=child.decs;
        N=length(Population);

        %%%%%%child
        dist_y =  pdist2(Population_obj, child_obj);
        [dis_k_y,~]=sort(dist_y);
        sum_y=sum(dis_k_y);
        neardist_y_chi=dis_k_y(1);
        dis_y_chi=sum_y*neardist_y_chi;
        %%%%%%par
        par_obj=par.objs;       
        dist_y1 =  pdist2(Population_obj, par_obj);
        [dis_k_y1,~]=sort(dist_y1);
        sum_y1=sum(dis_k_y1);
        neardist_y_par=dis_k_y1(1);
        dis_y_par=sum_y1*neardist_y_par;
    end