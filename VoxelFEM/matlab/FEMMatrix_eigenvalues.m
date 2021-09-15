clear all
close all

% path = "../debug/FEMMatrix";
path = "../debug/FEMMatrixFixedDOFs";

A = read_sparse_matrix_binary(path);
K = full_stiffness_matrix_from_upper_sparse(A);

eigs(K, 4, 'smallestreal')

fclose('all');