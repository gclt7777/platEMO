function Y = load_objs(file)
    V = load(file);
    cand = {'objs','Y','Pop','Population','result'};
    Y = [];
    if isfield(V,'objs'),          Y = V.objs;  end
    if isempty(Y) && isfield(V,'Y'),           Y = V.Y;     end
    if isempty(Y) && isfield(V,'Population'),  Y = V.Population.objs; end
    if isempty(Y) && isfield(V,'Pop') && isa(V.Pop,'SOLUTION')
        Y = V.Pop.objs; 
    end
    if isempty(Y) && isfield(V,'result')
        R = V.result{end};
        if isa(R,'SOLUTION'), Y = R.objs;
        elseif isstruct(R) && isfield(R,'objs'), Y = R.objs; end
    end
    if isempty(Y), error('在 %s 里找不到目标矩阵（objs/Y/Population/result）。', file); end
end
