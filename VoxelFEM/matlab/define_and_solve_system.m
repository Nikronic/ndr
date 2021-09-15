clear all
close all

% loadDOFs = [5, 7];   % traction
loadDOFs = [6, 8];   % flexion
fixedDOFs = [1, 2, 3, 4];

% Stiffness matrix
A = read_sparse_matrix_binary("../debug/FEMMatrix");
K = full_stiffness_matrix_from_upper_sparse(A);

% Try to swap rows and cols in stiffness matrix to find C++ bug
K = anotherRowsColsSwap(K);

% Eliminate fixed DOFs
K_reduced = fixVariables(K, sort(fixedDOFs,'descend'));

% Try to swap rows and cols in stiffness matrix to find C++ bug
% K_reduced = pairwiseRowsColsSwap(K_reduced);

% Load vector
f = zeros(size(K, 1), 1);
f = applyLoads(f, loadDOFs);
f_reduced = fixVariables(f, sort(fixedDOFs,'descend'));

% Solve
u_reduced = K_reduced \ f_reduced;
u = addFixedDOFs(u_reduced, fixedDOFs)

% f_x = f(1:2:end-1);
% f_y = f(2:2:end);
% u_x = u(1:2:end-1);
% u_y = u(2:2:end);
% x = [0;0;1;1;2;2];
% y = [0; 1; 0; 1; 0; 1];
% 
% figure
% quiver(x, y, f_x, f_y)
% 
% figure
% quiver(x, y, u_x, u_y)



