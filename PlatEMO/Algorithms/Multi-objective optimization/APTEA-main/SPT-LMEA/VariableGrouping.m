function group = VariableGrouping(D,nvg,tvg)
%  Copyright (C) 2023 Songbai Liu
%  Songbai Liu <songbai@szu.edu.cn>   
% ----------------------------------------------------------------------- 
    
    switch tvg
        case 1 %random grouping
            randList = randperm(D);
			len = D/nvg;
			pointer = 1;
			group = cell(1,nvg);
			for i = 1:(nvg-1)
				for j = 1:len
					group{i}(j) = randList(pointer);
					pointer = pointer+1;
				end
			end
			for j = pointer:D
				group{nvg}(j-pointer+1) = randList(j);
			end
		case 2 %linear grouping
            len = D/nvg;
			pointer = 1;
			group = cell(1,nvg);
			for i = 1:(nvg-1)
				for j = 1:len
					group{i}(j) = pointer;
					pointer = pointer+1;
				end
			end
			for j = pointer:D
				group{nvg}(j-pointer+1) = j;
			end
    end
end