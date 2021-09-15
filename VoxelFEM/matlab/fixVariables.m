function A = fixVariables(A, fixedDOFs)
% Drop rows and columns corresponding to the provided DOFs
% NB: 1) indices start from 1
%     2) fixedDOFs listed from higher to lower (A shape is modified at each iteration)

if size(A, 2) > 1        % matrix
    for i = fixedDOFs
        A(i, :) = [];
        A(:, i) = [];
    end
elseif size(A, 2) == 1   % vector
    for i = fixedDOFs
        A(i) = [];
    end
end

end