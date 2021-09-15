////////////////////////////////////////////////////////////////////////////////
// test_tp_gauss_quadrature.cc
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Test the tensor product Gauss quadrature rules on all multivariate
//  polynomials up to degree 9 in 1, 2, and 3 dimensions. We validate that
//  every quadrature rule that should be able to integrate a given polynomial
//  exactly does indeed get the correct answer (as precomputed in the
//  integrals*Vars arrays) to machine precision.
//
//  We also exhaustively test the TensorProductPolynomialInterpolant class by
//  converting the function into an interpolant of the appropriate degree and
//  integrating the interpolant.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Company:  New York University
//  Created:  11/08/2017 15:34:31
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <cmath>
using Real = double;
#include "../TensorProductQuadrature.hh"
#include "tp_quadrature_1var_test.inl"
#include "tp_quadrature_2var_test.inl"
#include "tp_quadrature_3var_test.inl"

#include "../TensorProductPolynomialInterpolant.hh"

size_t numTests = 0, numPassed = 0;

// Whether the tensor product quadrature rule with degrees "Degrees" can
// integrate the polynomial with degrees "degs" exactly.
template<size_t... Degrees, size_t... Idxs>
bool canIntegrateExactly(const std::array<Real, sizeof...(Degrees)> &degs, Future::index_sequence<Idxs...>) {
    std::array<bool, sizeof...(Degrees)> degreeSufficient{(degs[Idxs] <= Degrees)...};
    bool result = true;
    for (bool b : degreeSufficient) result &= b;
    return result;
}

template<size_t... Degrees, typename ITable, typename DTable, typename FTable>
void test_nd_quadrature(const ITable &exactIntegrals, const DTable &variableDegrees, const FTable &functions) {
    const size_t nTestFuncs = exactIntegrals.size();
    constexpr size_t Dim = sizeof...(Degrees);
    for (size_t i = 0; i < nTestFuncs; ++i) {
        // Only test the polynomials this rule should integrate exactly
        if (!canIntegrateExactly<Degrees...>(variableDegrees[i], Future::make_index_sequence<Dim>())) continue;
        ++numTests;
        Real numeric = TensorProductQuadrature<Degrees...>::integrate(functions[i]);
        Real error = std::abs(exactIntegrals[i] - numeric);
        if (error > 5e-16) {
            std::cerr << Dim << "D test polynomial " << i << " integrated with error " << error << std::endl;
        }
        else { ++numPassed; }

        // Note: the following are unfortunately slow to compile and run for high degree. We may need
        // to optimize our interpolant class for higher degree evaluations, but we should need
        // more than degree 2 or so in practice.
#if 0
        auto interp = make_tp_interpolant<Degrees...>(functions[i]);
        numeric = TensorProductQuadrature<Degrees...>::integrate([&](const EvalPt<Dim> &p) { return interp(p); });
        error = std::abs(exactIntegrals[i] - numeric);
        ++numTests;
        if (error > 5e-16) {
            std::cerr << Dim << "D test polynomial interpolant " << i << " integrated with error " << error << std::endl;
        }
        else { ++numPassed; }
#endif
    }
}

// 'if's below are to prevent running tests more than once (since, e.g.,
// test_1d_quadrature runs the same tests regardless of d2 and d3).
template<size_t d1, size_t d2, size_t d3> void test_1d_quadrature() { if ((d2 == 0) && (d3 == 0)) test_nd_quadrature<d1        >(integrals1Vars, varDegrees1Vars, functions1Vars); }
template<size_t d1, size_t d2, size_t d3> void test_2d_quadrature() { if (             (d3 == 0)) test_nd_quadrature<d1, d2    >(integrals2Vars, varDegrees2Vars, functions2Vars); }
template<size_t d1, size_t d2, size_t d3> void test_3d_quadrature() {                             test_nd_quadrature<d1, d2, d3>(integrals3Vars, varDegrees3Vars, functions3Vars); }

template<size_t MaxDegree, size_t Dims, size_t Counter, typename>
struct AllTestsImpl;

template<size_t MaxDegree, size_t Dims, size_t Counter, size_t... Degrees>
struct AllTestsImpl<MaxDegree, Dims, Counter, Future::index_sequence<Degrees...>> {
    static void run() {
        AllTestsImpl<MaxDegree, Dims    , Counter - 1, Future::index_sequence<Degrees...>>::run();
        AllTestsImpl<MaxDegree, Dims - 1,   MaxDegree, Future::index_sequence<Degrees..., Counter>>::run();
    }
};

// Counter = 0 base case: done with iteration for this dimension
template<size_t MaxDegree, size_t Dims, size_t... Degrees>
struct AllTestsImpl<MaxDegree, Dims, 0, Future::index_sequence<Degrees...>> {
    static void run() { AllTestsImpl<MaxDegree, Dims - 1, MaxDegree, Future::index_sequence<Degrees..., 0>>::run(); }
};

// Dims = 0 base case: done with all iteration, actually run the test
template<size_t MaxDegree, size_t Counter, size_t... Degrees>
struct AllTestsImpl<MaxDegree, 0, Counter, Future::index_sequence<Degrees...>> {
    static void run() {
        test_1d_quadrature<Degrees...>();
        test_2d_quadrature<Degrees...>();
        test_3d_quadrature<Degrees...>();
    }
};

template<size_t MaxDegree, size_t Dims>
struct AllTests : public AllTestsImpl<MaxDegree, Dims, MaxDegree, Future::index_sequence<>> { };

int main(int , const char *[]) {
    // Example usages for 3D quadrature:
    std::cout << TensorProductQuadrature<1, 1, 1>::integrate([](Real x, Real y, Real z) { return x * y * z; }) << std::endl;
    std::cout << TensorProductQuadrature<1, 1, 1>::integrate([](const EvalPt<3> &p) { return std::get<0>(p) *
                                                                                             std::get<1>(p) *
                                                                                             std::get<2>(p); }) << std::endl;

    auto interp = make_tp_interpolant<1, 1, 1>([](Real x, Real y, Real z) { return x * y * z; });
    std::cout << TensorProductQuadrature<1, 1, 1>::integrate([&](const EvalPt<3> &p) { return interp(p); }) << std::endl;

    // Run exhaustive tests on all multivariate polynomials with dimension up to 3 and degree up to 9.
    AllTests<9, 3>::run();
    std::cout << numPassed << " / " << numTests << " tests passed." << std::endl;
    return 0;
}
