#include <KTRSolver.h>
#include <KTRProblem.h>
#include <iostream>

#include "TPPeriodicHomogenization.hh"

// A type for the different possible implementation of the smoothing regularization term
enum class SmoothingType { Laplacian, Bilaplacian };

/// A problem class to be passed to the Knitro optimization algorithm
/// Since we do not know a priori in which order the algorithm will evaluate the objective function and objective function
/// gradient, members of the class are used to cache these values, and update them only if needed
template<typename _Sim>
class MicrostructureDesignProblem : public knitro::KTRProblem {
    using LI = LinearIndexer<typename _Sim::ETensor>;
    using ETensor = typename _Sim::ETensor;
    using Base = knitro::KTRProblem;
    // One constraint per elasticity tensor entry;
    static constexpr size_t numConstraints = LI::size();

public:
    // constructor: pass number of variables and constraints to base class
    MicrostructureDesignProblem(_Sim &sim, const ETensor &ETarget, MSHFieldWriter &writer,
                                Real w_integer, Real w_smoothness, Real w_volume,
                                Real density_lower_bound, std::string name)
            : KTRProblem(sim.numElements(), numConstraints),
              m_sim(sim), m_ETarget(ETarget), m_writer(writer),
              m_w_integer(w_integer), m_w_smoothness(w_smoothness), m_w_volume(w_volume),
              m_min_density(density_lower_bound), m_name(name)
    {
        // set problem properties
        setObjectiveProperties();
        setVariableProperties();
        setConstraintProperties();
    }

    // set the initial vector of the optimizer
    void setXInitial(const std::vector<double> &x) {
        m_x_init = x;
        Base::setXInitial(x);
    }

    // Objective and constraint evaluation function
    // overrides KTRProblem class
    double evaluateFC(const std::vector<double>& x,
            std::vector<double>& c,
            std::vector<double>& objGrad,
            std::vector<double>& jac) {
        const size_t numVars = x.size();
        assert(numVars == m_sim.numElements());
        assert(c.size() == numConstraints);

        // Update the cache values if needed
        m_evaluateCache(x);

        for (size_t i = 0; i < numConstraints; ++i)
            c[i] = LI::index(m_Eh, i) - LI::index(m_ETarget, i);

        Real val = 0;

        // Compute material volume from the density variables
        // Note that, for SIMP, the terminology is a bit misleading:
        // the optimization variables x are actually still densities, but
        // sim.setElementDensity() configures the element's (relative) Young's
        // modulus **not** density.
        // We compute this Young relative 's modulus from the density
        // (optimization variables) using the SIMP formula.
        for (size_t ei = 0; ei < numVars; ++ei)
            val += m_w_volume * x[ei] * m_sim.elementVolume(ei);

        // ---- Regularization terms:
        // Integer enforcement
        for (size_t ei = 0; ei < x.size(); ++ei)
            val += m_w_integer * (x[ei] - m_min_density) * (1.0 - x[ei]);

        // Smoothness
        // Either the discrete difference discretization of the Laplacian or bi-Laplacian operator
        // Create a NDVector of the variables to take advantage of the multi-index operations
        NDVector<Real> NDdensities (m_sim.NbElementsPerDimension());
        NDdensities.fill(x);
        NDVector<Real> diffToNeighborsAverageDensity = NDdensities.differenceToNeighborsAverage();
        for (size_t ei = 0; ei < x.size(); ++ei) {
            switch (m_smoothType) {
                case SmoothingType::Bilaplacian:
                    val += m_w_smoothness * diffToNeighborsAverageDensity[ei] * diffToNeighborsAverageDensity[ei];
                    break;
                case SmoothingType::Laplacian:
                    val += -m_w_smoothness * x[ei] * diffToNeighborsAverageDensity[ei];
                    break;
                default: assert(false);
            }
        }

        return val;
    }

    // Gradient and Jacobian evaluation function
    // overrides KTRProblem class
    int evaluateGA(const std::vector<double>& x,
            std::vector<double>& objGrad,
            std::vector<double>& jac) override {
        const size_t numVars = x.size();
        assert(numVars == m_sim.numElements());
        assert(objGrad.size() == numVars);
        assert(jac.size() == numConstraints * numVars);

        // Update the cache values if needed
        m_evaluateCache(x);

        objGrad.assign(numVars, 0);

        for (size_t j = 0; j < numVars; ++j)
            objGrad[j] += m_w_volume * m_sim.elementVolume(j);

        // ---- Regularization terms:
        // Integer enforcement
        for (size_t j = 0; j < numVars; ++j) {
            objGrad[j] += m_w_integer * m_p_integer * std::pow(4 * (x[j] - m_min_density) * (1.0 - x[j]) + m_eps_integer, m_p_integer - 1.0)* 4 * (-2*x[j] + 1.0 + m_min_density);
        }

        // Smoothness
        NDVector<Real> NDdensities (m_sim.NbElementsPerDimension());
        NDdensities.fill(x);
        NDVector<Real> diffToNeighborsAverageDensity = NDdensities.differenceToNeighborsAverage();
        NDVector<Real> diffDiffAverageDensity = diffToNeighborsAverageDensity.differenceToNeighborsAverage();
        for (size_t ei = 0; ei < x.size(); ++ei) {
            switch (m_smoothType) {
                case SmoothingType::Bilaplacian:
                    objGrad[ei] += m_w_smoothness * 2 * diffDiffAverageDensity[ei];
                    break;
                case SmoothingType::Laplacian:
                    objGrad[ei] += -m_w_smoothness * 2 * diffToNeighborsAverageDensity[ei];
                    break;
                default: assert(false);
            }
        }

        // ---- Constraint Jacobian
        for (size_t i = 0; i < numConstraints; ++i) {
            for (size_t j = 0; j < numVars; ++j) {
                jac[i * numVars + j] = LI::index(m_EhGradient[j], i);
            }
        }

        return 0;
    }

    // SIMP interpolation:
    // "density" = min_density + (max_density - min_density) * var^simp_power (cf comment in evaluateFC)
    void setSIMPPower(Real p) { m_simp_power = p; }

    double     simp_var_for_density(double  density) const { return m_disableSimp ?  density : pow((std::max(density, m_min_density) - m_min_density) / (1.0 - m_min_density), 1.0 / m_simp_power); }
    double     density_for_simp_var(double simp_var) const { return m_disableSimp ? simp_var : m_min_density + (1.0 - m_min_density) * pow(simp_var, m_simp_power); }
    double der_density_for_simp_var(double simp_var) const { return m_disableSimp ?      1.0 : (1.0 - m_min_density) * m_simp_power * pow(simp_var, m_simp_power - 1); }

    void setSmoothingType(SmoothingType t) { m_smoothType = t; }

private:
    // objective properties
    void setObjectiveProperties() {
        setObjType(KTR_OBJTYPE_GENERAL);
        setObjGoal(KTR_OBJGOAL_MINIMIZE);
    }

    // constraint properties
    void setConstraintProperties()
    {
        // set constraint types (general nonlinear)
        for (size_t i = 0; i < numConstraints; ++i) {
            setConTypes(i, knitro::KTREnums::ConstraintType::ConGeneral);
        }
        // set constraint lower bounds
        setConLoBnds(0.0);

        // set constraint upper bounds
        setConUpBnds(0.0);
    }

    // Variable bounds. All densities m_min_density <= x <= 1.
    void setVariableProperties() {
        setVarLoBnds(m_disableSimp ? m_min_density : 0.0); // With SIMP interpolation, variable range is [0, 1]
        setVarUpBnds(1);
    }

    // Checks if the cache ETensor gradient and objective function correspond to x
    // If not, compute new cache values according to x
    bool m_evaluateCache(const std::vector<double>& x) {
        double norm = 0;
        for (size_t i = 0; i < m_x_cache.size(); ++i)
            norm += (m_x_cache[i] - x[i]) * (m_x_cache[i] - x[i]);
        if ((m_x_cache.size() > 0) && (norm < 1e-16))
            return false;
        m_x_cache = x;

        // Update the densities
        for (size_t ei = 0; ei < x.size(); ++ei)
            m_sim.setElementDensity(ei, density_for_simp_var(x[ei]));

        try{
            TPPeriodicHomogenization::solveCellProblems(m_w, m_sim);
        } catch(...) {
            m_writer.addField("failure_density", m_sim.getDensities(), DomainType::PER_ELEMENT);
            throw;
        }

        // Distance to Etarget
        m_Eh = TPPeriodicHomogenization::homogenizedElasticityTensor(m_w, m_sim);
        Real dist = (m_Eh - m_ETarget).frobeniusNormSq() / m_ETarget.frobeniusNormSq();
        // Gradient with respect to density
        m_EhGradient = TPPeriodicHomogenization::homogenizedElasticityTensorGradient(m_w, m_sim);
        // Gradient with respect to SIMP var (chain rule for density_for_simp_var)
        for (size_t i = 0; i < x.size(); ++i)
            m_EhGradient[i] *= der_density_for_simp_var(x[i]);

        m_Eh.printOrthotropic(std::cout);
        std::cout << "Current distance (iterate " << m_iterCounter << "): " << dist << std::endl;

        m_writer.addField(m_name + " - density "  + std::to_string(m_iterCounter), m_sim.getDensities(), DomainType::PER_ELEMENT);
        ++m_iterCounter;
        return true;
    }

    // finite element simulator
    _Sim &m_sim;
    // target elasticity tensor
    ETensor m_ETarget, m_Eh;
    // Mesh writer to output the activations
    MSHFieldWriter &m_writer;

    // Variable to cache values of Elasticity tensor and Gradient
    std::vector<ETensor> m_EhGradient;
    std::vector<double> m_x_cache, m_x_init;

    // To hold the solution to the cell problem
    std::vector<typename _Sim::VField> m_w;

    // Counter of the number of steps done by the optimization algorithm
    size_t m_iterCounter = 0;

    // Weights of the regularization terms
    Real m_w_integer;
    Real m_w_smoothness;
    Real m_w_volume = 1.0;

    // Exponent to use in the integer regularization term, to get steeper slope toward 0 and 1
    Real m_p_integer = 1;

    // Small quantity to prevent division by 0 when setting m_p_integer to 1
    Real m_eps_integer = 1e-3;

    // Minimal activation value, to ensure the finite element matrix is positive definite
    Real m_min_density = 1e-4;

    // Value of the SIMP exponent
    Real m_simp_power = 3.0;

    // Smoothing formula to use to compute the smoothness regularization term
    // Choice between the finite difference discretization of the bi-Laplacian, or Laplacian, operator
    SmoothingType m_smoothType = SmoothingType::Laplacian;

    // Whether to use SIMP formulation or not
    bool m_disableSimp = false;

    // Prefix to add to the name of the density field written out.
    std::string m_name;
};


////////////////////////////////////////////////////////////////////////////////
/*! Solve the microstructure optimization problem using the knitro library solver
 *  problem:  min Regularization_terms(densities) subject to Homogenized ETensor = Etarget
//  @param[inout] sim       Linear elasticity simulator for omega.
//  Warning: this function mutates sim by applying periodic and pin
//           constraints.
//  @param[in] Etarget              Target elasticity tensor
//  @param[in] writer               Writer to output the activations
//  @param[in] smoothType           Designs the formulation to use to compute the smoothness regularization term
//  @param[in] w_integer            Weight of the integer regularization term
//  @param[in] w_smoothness         Weight of the smoothness regularization term
//  @param[in] w_volume             Weight of the volume regularization term
//  @param[in] simpPower            SIMP exponent
//  @param[in] density_lower_bound  Lower bound of the activations (stricly positive, to ensure a positive definite FEM matrix)
//  @param[in] prefix               prefix string used to differentiate the density fields from another previously output with writer
*///////////////////////////////////////////////////////////////////////////////
template<class _Sim>
void optimize_knitro(
        _Sim &sim,
        const typename _Sim::ETensor &Etarget,
        MSHFieldWriter &writer,
        size_t /* maxIter */,
        Real w_integer,
        Real w_smoothness,
        Real w_volume,
        SmoothingType smoothType,
        Real simpPower,
        Real density_lower_bound,
        std::string prefix)
{
    // Create a problem instance.
    MicrostructureDesignProblem<_Sim> problem(sim, Etarget, writer,
                                              w_integer, w_smoothness, w_volume,
                                              density_lower_bound, prefix);
    problem.setSIMPPower(simpPower);
    problem.setSmoothingType(smoothType);

    {
        std::vector<double> x_init;
        x_init.resize(sim.numElements());
        // Knitro doesn't like it if the variables are too close to their bounds.
        // (It rejects the initial point).
        for (size_t i = 0; i < sim.numElements(); ++i)
            x_init[i] = std::min(std::max(sim.elementDensity(i), 1e-3), 1 - 1e-3);
        // Convert density to SIMP vars
        for (size_t i = 0; i < sim.numElements(); ++i)
            x_init[i] = problem.simp_var_for_density(x_init[i]);
        problem.setXInitial(x_init);
    }

    // Create a solver - optional arguments:
    // exact first derivatives
    // BFGS approximate second derivatives
    knitro::KTRSolver solver(&problem, KTR_GRADOPT_EXACT, KTR_HESSOPT_LBFGS);
    solver.setParam(KTR_PARAM_LMSIZE, 15); // history size
    solver.setParam(KTR_PARAM_HONORBNDS, KTR_HONORBNDS_ALWAYS); // always respect bounds during optimization
    solver.setParam(KTR_PARAM_FEASTOL, 1e-2); // equality constraints relative tolerance
    solver.setParam(KTR_PARAM_PRESOLVE, KTR_PRESOLVE_NONE);
    solver.setParam(KTR_PARAM_ALGORITHM, KTR_ALG_ACT_CG);   // active set CG with BFGS Hessian approximation

    int solveStatus = solver.solve();

    if (solveStatus != 0) {
        std::cout << std::endl;
        std::cout << "KNITRO failed to solve the problem, final status = ";
        std::cout << solveStatus << std::endl;
    }
    else {
        std::cout << std::endl << "KNITRO successful";
    }

    const auto soln = solver.getXValues();

    // Set the solver solution in the simulator
    assert(soln.size() == sim.numElements());
    for (size_t i = 0; i < soln.size(); ++i)
        sim.setElementDensity(i, problem.density_for_simp_var(soln[i]));

    writer.addField("Final density", sim.getDensities(), DomainType::PER_ELEMENT);
}