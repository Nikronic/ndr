////////////////////////////////////////////////////////////////////////////////
// TensorProductBasisPolynomial
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Tensor Product Basis polynomial class
//
//  Let denote by phi the Lagrange Basis Polynomials
//  A basis of polynomials in multiple dimension is obtained from the 1D Lagrange
//  Polynomials by a tensor product of a 1D polynome of adequate degree in each
//  dimension: phi_{i,j,k}(X) = phi_i(x1) * phi_j(x2) * phi_k(x3)
//
//  Degrees is a list of the 1D polynomials degree
//      in our examples above, a list of (2, 1, 3)
//      means that phi_i is of degree 2, phi_j of
//      degree 1 and phi_k of degree 3
//  Values of indices i,j,k... are template paramenters
//  to the eval function
*/
////////////////////////////////////////////////////////////////////////////////
#ifndef TENSORPRODUCTBASISPOLYNOMIAL_HH
#define TENSORPRODUCTBASISPOLYNOMIAL_HH

#include "LagrangePolynomial.hh"
#include <MeshFEM/Future.hh>

// Base case (0 dimensional)
template<size_t... degrees>
struct TensorProductBasisPolynomial {
    template<size_t... idxs>
    static constexpr Real eval() { return 1; }
};

// Recursive case
template<size_t d, size_t... degrees>
struct TensorProductBasisPolynomial<d, degrees...> {
    // Evaluate the basis function labeled phi_<i, ...>
    // called as eval<i, ...>(ci, ...)
    template<size_t i, size_t... idxs, typename... Args>
    static Real eval(Real ci, Args... cRest) {
        static_assert(sizeof...(idxs) == sizeof...(degrees), "Invalid number of indices");
        static_assert(sizeof...(idxs) == sizeof...(cRest),   "Invalid number of coordinates");
        return TensorProductBasisPolynomial<degrees...>::template eval<idxs...>(cRest...) *
               LagrangeBasisPolynomial<d,i>::eval(ci);
    }

    template<size_t... idxs>
    static Real eval(const VectorND<sizeof...(idxs)> &evalPt) {
        return m_eval_helper<idxs...>(evalPt, Future::make_index_sequence<sizeof...(idxs)>());
    }

private:
    // Helper function to expand the values from the "evalPoint" tuple into the arguments of eval
    template <size_t... idxs, size_t... TupleExpandSeq>
    static Real m_eval_helper(const VectorND<sizeof...(idxs)> &evalPt, const Future::index_sequence<TupleExpandSeq...>) {
        return eval<idxs...>((evalPt[TupleExpandSeq])...);
    }
};

// Generate a tensor product of N Lagrange basis polynomials, all of degree "Deg".
// Replicate "Deg" N times, constructing "DegList". Then pass "DegList" as
// the template parameter for TensorProductBasisPolynomial.
template<size_t N, size_t Deg, size_t... DegList>
struct NDTensorProductBasisPolynomial : public NDTensorProductBasisPolynomial<N - 1, Deg, Deg, DegList...> { }; // Prepend "Deg" to DegList
template<size_t Deg, size_t... DegList>
struct NDTensorProductBasisPolynomial<0, Deg, DegList...> : public TensorProductBasisPolynomial<DegList...> { };

#endif /* end of include guard: TENSORPRODUCTBASISPOLYNOMIAL_HH */
