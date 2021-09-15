////////////////////////////////////////////////////////////////////////////////
// OptimalityCriterion.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  An implementation of the optimality criterion method for solving an
//  optimization problem with a single equality constraint (or a single
//  inequality constraint that is known to be active at the optimum).
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  03/08/2021 17:48:46
////////////////////////////////////////////////////////////////////////////////
#ifndef OPTIMALITYCRITERION_HH
#define OPTIMALITYCRITERION_HH

template<class Problem>
struct ConstraintEvaluator {
    ConstraintEvaluator(Problem &p) : m_p(p) {
        auto gridSize = p.getSimulator().NbElementsPerDimension();
        x.Resize(gridSize);
        xscratch.Resize(gridSize);
    }
    template<class Vars>
    Real operator()(const Vars &steppedVars) {
        x.flattened() = steppedVars;
        return m_p.evaluateOCConstraintAtVars(x, xscratch);
    }
private:
    Problem &m_p;
    NDVector<Real> x, xscratch;
};

template<class Problem>
struct OCOptimizer {
    using VXd = Eigen::VectorXd;

    OCOptimizer(Problem &p) : m_p(p), constraint_evaluator(p) {
        lambda_min = 1;
        lambda_max = 2;
    }

    void step(Real m = 0.2, Real ctol = 1e-6) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("OC step");
        VXd dJ = m_p.evaluateObjectiveGradient();
        VXd dc = m_p.evaluateConstraintsJacobian().row(0);
        auto x0 = m_p.getVars();

        auto stepped_vars_for_lambda = [&](Real lambda) {
            return (x0.array() * (dJ.array() / (dc * lambda).array()).sqrt()).cwiseMax(x0.array() - m).cwiseMax(0.0)
                                                                              .cwiseMin(x0.array() + m).cwiseMin(1.0);
        };
        auto ceval = [&](Real lambda) { return constraint_evaluator(stepped_vars_for_lambda(lambda)); };

        // Note: constraint function "c" is continuous and monotonically increasing,
        // and we seek to find its unique root with a bisection method.
        // (c = (1 - vol/(N_e * target_vol_frac))

        // TODO: determine if we want to reset the Lagrange multiplier brackets:
        // lambda_min = 1, lambda_max = 2;
        BENCHMARK_START_TIMER_SECTION("Bisection");
        while (ceval(lambda_min) > 0) { lambda_max = lambda_min; lambda_min /= 2; }
        while (ceval(lambda_max) < 0) { lambda_min = lambda_max; lambda_max *= 2; }
        Real lambda_mid = 0.5 * (lambda_min + lambda_max);
        Real vol = ceval(lambda_mid);
        while (std::abs(vol) > ctol) {
            if (vol < 0) lambda_min = lambda_mid;
            if (vol > 0) lambda_max = lambda_mid;
            lambda_mid = 0.5 * (lambda_min + lambda_max);
            vol = ceval(lambda_mid);
        }
        BENCHMARK_STOP_TIMER_SECTION("Bisection");

        m_p.setVars(stepped_vars_for_lambda(lambda_mid));

        std::cout << "objective, constraint, lambda estimate: " << m_p.evaluateObjective() << "\t" << m_p.evaluateConstraints()[0] << "\t" << lambda_mid << std::endl;
    }

private:
    Real lambda_min, lambda_max; // bracket around current guess of Lagrange multipliers
    Problem &m_p;
    ConstraintEvaluator<Problem> constraint_evaluator;
};

#endif /* end of include guard: OPTIMALITYCRITERION_HH */
