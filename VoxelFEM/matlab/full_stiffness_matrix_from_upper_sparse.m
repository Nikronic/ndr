function K = full_stiffness_matrix_from_upper_sparse(A)
if sum(size(A)) > 8000
    disp("Big matrix! Execution aborted")
    return;
end
F = full(A);
U = triu(F, 1);
D = diag(F);
K = diag(D) + U + U.';
end