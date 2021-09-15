clear all
close all

fixedDOFs = [1, 3, 10];

A = read_sparse_matrix_binary("../debug/FEMMatrix");
K = full_stiffness_matrix_from_upper_sparse(A);
Kreduced = fixVariables(K, sort(fixedDOFs,'descend'));

eigs(Kreduced, 4, 'smallestreal')
