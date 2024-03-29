std::vector<Real> integrals2Vars ={1./1, 1./2, 1./3, 1./4, 1./5, 1./6, 1./7, 1./8, 1./9, 1./10, 1./2, 1./4, 1./6, 1./8, 1./10, 1./12, 1./14, 1./16, 1./18, 1./20, 1./3, 1./6, 1./9, 1./12, 1./15, 1./18, 1./21, 1./24, 1./27, 1./30, 1./4, 1./8, 1./12, 1./16, 1./20, 1./24, 1./28, 1./32, 1./36, 1./40, 1./5, 1./10, 1./15, 1./20, 1./25, 1./30, 1./35, 1./40, 1./45, 1./50, 1./6, 1./12, 1./18, 1./24, 1./30, 1./36, 1./42, 1./48, 1./54, 1./60, 1./7, 1./14, 1./21, 1./28, 1./35, 1./42, 1./49, 1./56, 1./63, 1./70, 1./8, 1./16, 1./24, 1./32, 1./40, 1./48, 1./56, 1./64, 1./72, 1./80, 1./9, 1./18, 1./27, 1./36, 1./45, 1./54, 1./63, 1./72, 1./81, 1./90, 1./10, 1./20, 1./30, 1./40, 1./50, 1./60, 1./70, 1./80, 1./90, 1./100};
std::vector<std::array<Real, 2>> varDegrees2Vars ={{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9}, {2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {2, 9}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {3, 8}, {3, 9}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6}, {4, 7}, {4, 8}, {4, 9}, {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6}, {5, 7}, {5, 8}, {5, 9}, {6, 0}, {6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}, {6, 7}, {6, 8}, {6, 9}, {7, 0}, {7, 1}, {7, 2}, {7, 3}, {7, 4}, {7, 5}, {7, 6}, {7, 7}, {7, 8}, {7, 9}, {8, 0}, {8, 1}, {8, 2}, {8, 3}, {8, 4}, {8, 5}, {8, 6}, {8, 7}, {8, 8}, {8, 9}, {9, 0}, {9, 1}, {9, 2}, {9, 3}, {9, 4}, {9, 5}, {9, 6}, {9, 7}, {9, 8}, {9, 9}};
std::vector<std::function<Real(Real, Real)>> functions2Vars = 
{[](Real x, Real y) { return 1; },
 [](Real x, Real y) { return y; },
 [](Real x, Real y) { return y*y; },
 [](Real x, Real y) { return y*y*y; },
 [](Real x, Real y) { return y*y*y*y; },
 [](Real x, Real y) { return y*y*y*y*y; },
 [](Real x, Real y) { return y*y*y*y*y*y; },
 [](Real x, Real y) { return y*y*y*y*y*y*y; },
 [](Real x, Real y) { return y*y*y*y*y*y*y*y; },
 [](Real x, Real y) { return y*y*y*y*y*y*y*y*y; },
 [](Real x, Real y) { return x; },
 [](Real x, Real y) { return x*y; },
 [](Real x, Real y) { return x*(y*y); },
 [](Real x, Real y) { return x*(y*y*y); },
 [](Real x, Real y) { return x*(y*y*y*y); },
 [](Real x, Real y) { return x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x; },
 [](Real x, Real y) { return x*x*y; },
 [](Real x, Real y) { return x*x*(y*y); },
 [](Real x, Real y) { return x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x; },
 [](Real x, Real y) { return x*x*x*y; },
 [](Real x, Real y) { return x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x; },
 [](Real x, Real y) { return x*x*x*x*y; },
 [](Real x, Real y) { return x*x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x; },
 [](Real x, Real y) { return x*x*x*x*x*y; },
 [](Real x, Real y) { return x*x*x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x; },
 [](Real x, Real y) { return x*x*x*x*x*x*y; },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x; },
 [](Real x, Real y) { return x*x*x*x*x*x*x*y; },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x; },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*y; },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*(y*y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x; },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*y; },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y*y*y*y*y*y); },
 [](Real x, Real y) { return x*x*x*x*x*x*x*x*x*(y*y*y*y*y*y*y*y*y); }};
