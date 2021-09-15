function f = applyLoads(f, loadDOFs)
% Distribute a unitary load on the specified node and direction

loadPerNode = 1 / (sum(size(loadDOFs))-1);
for DOF = loadDOFs
    f(DOF) = loadPerNode;
end

end