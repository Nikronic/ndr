function A_swapped = pairwiseRowsColsSwap(A)
% Swap odd and even rows of the matrix; the same for columns.

assert(size(A, 1) == size(A, 2));

indices = zeros(size(A, 1), 1);
for i = 1:2:size(A, 1)
    indices(i) = i+1;
    indices(i+1) = i;
end
A_swapped_cols = A(:, indices);
A_swapped = A_swapped_cols(indices, :);
end