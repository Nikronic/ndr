std::vector<Real> integrals1Vars ={1./1, 1./2, 1./3, 1./4, 1./5, 1./6, 1./7, 1./8, 1./9, 1./10};
std::vector<std::array<Real, 1>> varDegrees1Vars ={{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
std::vector<std::function<Real(Real)>> functions1Vars = 
{[](Real x) { return 1; },
 [](Real x) { return x; },
 [](Real x) { return x*x; },
 [](Real x) { return x*x*x; },
 [](Real x) { return x*x*x*x; },
 [](Real x) { return x*x*x*x*x; },
 [](Real x) { return x*x*x*x*x*x; },
 [](Real x) { return x*x*x*x*x*x*x; },
 [](Real x) { return x*x*x*x*x*x*x*x; },
 [](Real x) { return x*x*x*x*x*x*x*x*x; }};
