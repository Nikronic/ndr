function A_swapped = anotherRowsColsSwap(A)
% Swap rows and columns such that resulting local stiffness matrix is
% comuputed starting from strains ordered by colMajor instead of rowMajor

assert(size(A, 1) == size(A, 2));

indices = [1, 2, 5, 6, 3, 4, 7, 8];
A_swapped_cols = A(:, indices);
A_swapped = A_swapped_cols(indices, :);
end