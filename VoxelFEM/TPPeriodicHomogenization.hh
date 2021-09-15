#ifndef TPPERIODICHOMOGENIZATION_HH
#define TPPERIODICHOMOGENIZATION_HH

#include <vector>
#include <string>

#include <MeshFEM/GaussQuadrature.hh>
#include <MeshFEM/InterpolantRestriction.hh>
#include <MeshFEM/GlobalBenchmark.hh>

#include <MeshFEM/ElasticityTensor.hh>
#include <MeshFEM/Parallelism.hh>


// #define FD_SD_DEBUG
#ifdef FD_SD_DEBUG
#include <MeshFEM/MSHFieldWriter.hh>
#endif

namespace TPPeriodicHomogenization {

////////////////////////////////////////////////////////////////////////////////
/*! Solve the linear elasticity periodic homogenization cell problems for
//  each constant strain e^ij:
//       -div E : [ strain(w^ij) + e^ij ] = 0 in omega
//        n . E : [ strain(w^ij) + e^ij ] = 0 on omega's boundary
//        w^ij periodic
//        w^ij = 0 on arbitrary internal node ("pin" no rigid translation constraint)
//  @param[out]   w_ij   Fluctuation displacements (cell problem solutions)
//  @param[inout] sim    Linear elasticity simulator for omega.
//  Warning: this function mutates sim by applying periodic and pin
//           constraints.
*///////////////////////////////////////////////////////////////////////////////
template<class _Sim>
void solveCellProblems(std::vector<typename _Sim::VField> &w_ij, _Sim &sim,
                       Real cellEpsilon = 1e-7,
                       bool ignorePeriodicMismatch = false,
                       std::unique_ptr<PeriodicCondition<_Sim::N>> pc = nullptr) {
    BENCHMARK_START_TIMER_SECTION("Cell problems");
    typedef typename _Sim::VField  VField;
    typedef typename _Sim::SMatrix SMatrix;
    constexpr size_t numStrains = SMatrix::flatSize();

    sim.applyPeriodicConditions(cellEpsilon, ignorePeriodicMismatch, std::move(pc));
    // sim.applyNoRigidMotionConstraint();
    sim.setUsePinNoRigidTranslationConstraint(true);

    w_ij.reserve(numStrains), w_ij.clear();
    for (size_t i = 0; i < numStrains; ++i) {
        BENCHMARK_START_TIMER("Constant Strain Load");
        VField rhs(sim.constantStrainLoad(-SMatrix::CanonicalBasis(i)));
        BENCHMARK_STOP_TIMER("Constant Strain Load");
        w_ij.push_back(sim.solve(rhs));
    }
    BENCHMARK_STOP_TIMER_SECTION("Cell problems");
}

template<class _Sim>
std::vector<typename _Sim::VField> solveCellProblems(_Sim &sim, Real cellEpsilon = 1e-7) {
    std::vector<typename _Sim::VField> w_ij;
    solveCellProblems(w_ij, sim, cellEpsilon);
    return w_ij;
}

////////////////////////////////////////////////////////////////////////////////
/*! Compute homogenized elasticity tensor (stress-like version):
//     Eh_ijkl = 1/|Y| int_omega [rho * E : strain(w_ij)]_kl + rho * E_ijkl dV
//  where |Y| = periodic cell (grid bounding box) volume
//  @param[in] w_ij           Fluctuation displacements
//  @param[in] sim            Linear elasticity simulator for omega.
//  @param[in] baseCellVolume |Y| (defaults to sim.volume())
//  @return    Homogenized elasticity tensor
*///////////////////////////////////////////////////////////////////////////////
template<class _Sim>
typename _Sim::ETensor homogenizedElasticityTensor(
        const std::vector<typename _Sim::VField> &w_ij, const _Sim &sim,
        Real baseCellVolume = 0.0) {
    BENCHMARK_START_TIMER("Homogenized tensor computation");

    if (baseCellVolume == 0.0) baseCellVolume = sim.volume();

    assert(w_ij.size() == _Sim::SMatrix::flatSize());

    typename _Sim::ETensor Eh;
    typename _Sim::SMatrix  avg_strain_ij;

    for (size_t ei = 0; ei < sim.numElements(); ++ei) {
        typename _Sim::ETensor Econtrib;
        for (size_t i = 0; i < w_ij.size(); ++i) {
            sim.elementAverageStrain(ei, w_ij[i], avg_strain_ij);
            Econtrib.DRowAsSymMatrix(i) =
                sim.elementElasticityTensor(ei).doubleContract(avg_strain_ij);
        }
        // Elasticity tensor is always constant on each element.
        Econtrib += sim.elementElasticityTensor(ei);
        Econtrib *= sim.elementVolume(ei) * sim.elementDensity(ei);
        Eh += Econtrib;
        
    }
    Eh /= baseCellVolume;

    BENCHMARK_STOP_TIMER("Homogenized tensor computation");
    return Eh;
}

////////////////////////////////////////////////////////////////////////////////
/*! Computes the derivative of the homogenized elasticity tensor with respect
 * to each density value.
 * d/dh|h=0 (C_{ijkl}(alpha + h gamma) = 1/|Y| int_Y gamma [strain(w_ij) + e_ij] : E : [strain(w_kl) + e_kl] dV
//  @param[in]  w       fluctuation displacements (cell problem solutions)
//  @param[in]  sim     linear elasticity solver
//  @return     sfield  derivative with respect to each element's density
*///////////////////////////////////////////////////////////////////////////////
template<class _Sim>
std::vector<typename _Sim::ETensor>
homogenizedElasticityTensorGradient(
        const std::vector<typename _Sim::VField> &w, const _Sim &sim) {
    BENCHMARK_START_TIMER("Gradient computation");

    typedef typename _Sim::ETensor ETensor;
    typedef typename _Sim::SMatrix SMatrix;
    using Strain = typename _Sim::Strain;
    constexpr size_t numStrains = SMatrix::flatSize();

    assert(w.size() == numStrains);

    Real volume = sim.volume();

    std::vector<typename _Sim::ETensor> gradient(sim.numElements());
    std::vector<Strain> we(numStrains), ws(numStrains);

    for (size_t ei = 0; ei < sim.numElements(); ++ei) {
        ETensor C = sim.elementElasticityTensor(ei);
        for (size_t ij = 0; ij < numStrains; ++ij) {
            auto eij = SMatrix::CanonicalBasis(ij);
            sim.elementStrain(ei, w[ij], we[ij]);
            for (size_t n = 0; n < Strain::size(); ++n) {
                we[ij][n] += eij;
                ws[ij][n] = C.doubleContract(we[ij][n]);
            }
        }

        for (size_t ij = 0; ij < numStrains; ++ij) {
            for (size_t kl = ij; kl < numStrains; ++kl) {
                gradient[ei].D(ij, kl) = _Sim::StiffnessMatrixQuadrature::integrate(
                             [&](const VectorND<_Sim::N> &p) {
                                return we[ij](p).doubleContract(ws[kl](p));
                             }
                );

            }
        }
        gradient[ei] *= sim.elementVolume(ei) / volume;
    }

    BENCHMARK_STOP_TIMER("Gradient computation");

    return gradient;
}



/// Updates the densities in each element of the similator to optimize the elasticity tensor toward the specified objective
/// This function is intended for testing, for real application a more powerfull solver is required
/// \param objective: objective elasticity tensor
/// \param alpha: step size
/// \param tol: tolerance toward the objective
/// \param maxiter: maximal number of iteration
template<class _Sim>
void gradientDescent(_Sim &sim, const typename _Sim::ETensor &objective,
                     std::vector<typename _Sim::VField> &w,
                     Real alpha, Real tol, size_t maxiter,
                     MSHFieldWriter &writer,
                     bool verbose = false) {

    auto currentETensor = homogenizedElasticityTensor(w, sim);
    auto grad = homogenizedElasticityTensorGradient(w, sim);

    size_t iter = 0;

    Real distance = (currentETensor- objective).frobeniusNormSq() / objective.frobeniusNormSq();

    // Gradient descent
    while( distance > tol && iter < maxiter ) {
        ScalarField<Real> gradients(grad.size());
        // Update the densities :
        for (size_t ei = 0; ei < grad.size(); ++ei) {
            Real objectiveGradient = 2*(currentETensor- objective).quadrupleContract(grad[ei]);
            sim.setElementDensity(ei, sim.elementDensity(ei) -alpha * objectiveGradient);
            gradients[ei] = objectiveGradient;
        }

        writer.addField("density " + std::to_string(iter), sim.getDensities(), DomainType::PER_ELEMENT);
        writer.addField("gradient " + std::to_string(iter), gradients, DomainType::PER_ELEMENT);

        // Get new solution to cell problem
        solveCellProblems(w, sim);

        currentETensor = homogenizedElasticityTensor(w, sim);
        grad = homogenizedElasticityTensorGradient(w, sim);

        distance = (currentETensor - objective).frobeniusNormSq() / objective.frobeniusNormSq();
        iter++;

        if (verbose) {
            std::cout << "Iteration: " << iter << " -----" <<std::endl;
            std::cout << "(Sq Rel Frob) Distance to Specified Tensor:\t" << distance << std::endl;
            std::cout << "Current tensor : " << std::endl;
            currentETensor.printOrthotropic(std::cout);
        }
    }
    return ;
}


} // namespace TPPeriodicHomogenization

#endif /* end of include guard: TPPERIODICHOMOGENIZATION_HH */
