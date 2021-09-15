function u = addFixedDOFs(u_reduced, fixedDOFs)
% Add zeros to the solution vector
% NB: 1) indices start from 1
%     2) fixedDOFs listed from higher to lower (A shape is modified at each iteration)

totalDOFs = sum(size(u_reduced)) + sum(size(fixedDOFs)) - 2;
u = zeros(totalDOFs, 1);
i_reduced = 1;
for i = 1:totalDOFs
    if any(fixedDOFs(:) == i)
        u(i) = 0;
    else
        u(i) = u_reduced(i_reduced);
        i_reduced = i_reduced + 1;
    end
    
end


end