#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONPROBLEM_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONPROBLEM_HH

#include <iostream>
#include <cmath>
#include <functional>
#include <memory>

#include "TensorProductSimulator.hh"
#include "TopologyOptimizationObjective.hh"
#include "TopologyOptimizationFilter.hh"
#include "TopologyOptimizationConstraint.hh"

// A general problem class for topology optimization.
// Provides methods for evaluating objective function, its derivatives, 
// constraints and derivatives of the constraints.
template<typename _Sim>
class TopologyOptimizationProblem {
public:
    using ObjectivePtr    = typename std::shared_ptr<Objective<_Sim>>;
    using FiltersList     = typename std::vector<std::shared_ptr<Filter>>;
    using ConstraintsList = typename std::vector<std::shared_ptr<Constraint>>;
    using VXd             = Eigen::VectorXd;

    TopologyOptimizationProblem(_Sim &simulator, const ObjectivePtr &objective, 
        const ConstraintsList &constraints, const FiltersList &filters = FiltersList()):
        m_sim(simulator), m_objective(objective), 
        m_filters(filters), m_constraints(constraints), 
        m_constrNumber(constraints.size()) {
            m_numVars = m_sim.numElements();
            initializeCache();
            initializeFiltersDimensions();
        }

    virtual ~TopologyOptimizationProblem() { };

    size_t numVars() const { return m_numVars; }
    Eigen::Map<const VXd> getVars() const { return Eigen::Map<const VXd>(m_cachedVars[0].data().data(), m_numVars); }

    // Compute all the needed fields given the value of the design variables
    // This method updates the state of the TOP object affecting the results of all the following function evaluations
    bool setVars(Eigen::Ref<const VXd> x, bool forceUpdate = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("setVars");

        // Notation to access design and physical variables
        NDVector<Real> &xDesign = m_cachedVars[0];
        NDVector<Real> &xPhys = m_cachedVars.back();

        // If the current densities correspond to the cached ones, return without any update
        if (!forceUpdate && (m_varsAreSet) && (xDesign.Size() > 0) && ((x - getVars()).norm() < 1e-16))
            return false;
        
        // Cache design variables
        xDesign.flattened() = x;

        // Filter variables and cache intermediate values
        for (size_t i = 0; i < m_filters.size(); i++)
            m_filters[i]->apply(m_cachedVars[i], m_cachedVars[i+1]);

        // Update Objective cache
        m_objective->updateCache(xPhys);

        m_varsAreSet = true;
        return true;
    }

    // For use in the binary search of the Optimality Criteria method
    // (seeking the Lagrange multiplier estimate that satisfies the volume constraint):
    // evaluate the volume constraint "c >= 0"; at the optimum we will have c == 0.
    // The following additional density arrays are needed to avoid memory allocations in each call:
    // @param[inout] x         input design/"blueprint" variables; will be overwritten with the filtered variables.
    // @param[inout] xscratch  scratch space for filtering the density variables; must be initialized to the proper size.
    Real evaluateOCConstraintAtVars(NDVector<Real> &x, NDVector<Real> &xscratch) const {
        if ((m_constraints.size() != 1) ||
            (nullptr == std::dynamic_pointer_cast<TotalVolumeConstraint>(m_constraints[0]))) {
            throw std::runtime_error("Applicable only for a topology optimization with a single (volume) constraint");
        }

        for (const auto &filt : m_filters) {
            filt->apply(x, xscratch);
            x.swap(xscratch);
        }

        return m_constraints[0]->evaluate(x);
    }

    // Compute value of objective function at current design variables
    Real evaluateObjective() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateObjective");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        const NDVector<Real> &xPhys = m_cachedVars.back();
        return m_objective->evaluate(xPhys);
    }

    // Compute gradient of objective function w.r.t. design variables
    // Return a vector of dimension m_numVars
    VXd evaluateObjectiveGradient() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateObjectiveGradient");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        // From dObjective/dPhysical to dObjective/dDesign
        size_t nFilters = m_filters.size();
        const NDVector<Real> &xPhys = m_cachedVars.back();

        NDVector<Real> gradient = m_objective->gradient();
        NDVector<Real> gradOld(m_sim.NbElementsPerDimension()); // derivative with respect to filtered densities
        for (size_t i = 0; i < nFilters; i++) {
            gradient.swap(gradOld);
            m_filters[(nFilters-1)-i]->backprop(gradOld, m_cachedVars[(nFilters-1)-i], gradient);
        }
        return gradient.flattened();
    }

    // Compute value of the constraints at current design variables
    // Return a vector of dimension m_constrNumber
    virtual VXd evaluateConstraints() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateConstraints");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        size_t nConstr = m_constraints.size();
        VXd constrValues(nConstr);
        const NDVector<Real> &xPhys = m_cachedVars.back();
        for(size_t i = 0; i < nConstr; i++)
            constrValues[i] = m_constraints[i]->evaluate(xPhys);  // constrValues[i] is imposed positive inside the optimizer
        return constrValues;
    }

    // Compute derivatives of all the constraints w.r.t. design variables
    // Return a matrix of dimensions [m_constrNumber, m_numVars]
    Eigen::MatrixXd evaluateConstraintsJacobian() const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("evaluateConstraintsJacobian");
        if (!m_varsAreSet) throw std::runtime_error("Must call setVars first!");

        // From dConstraint_n/dPhysical to dConstraint_n/dDesign
        Eigen::MatrixXd jacobian(m_constrNumber, m_numVars);
        size_t nFilters = m_filters.size();
        const NDVector<Real> &xPhys = m_cachedVars.back();
        NDVector<Real> derivatives(m_sim.NbElementsPerDimension()), derivativesOld(m_sim.NbElementsPerDimension());
        for (size_t n = 0; n < m_constrNumber; n++) {
            m_constraints[n]->backprop(xPhys, derivativesOld);  // dConstraint_n/dPhysical
            for (size_t i = 0; i < nFilters; i++) {
                m_filters[(nFilters-1)-i]->backprop(derivativesOld, m_cachedVars[(nFilters-1)-i], derivatives);
                derivativesOld.swap(derivatives);
            }
            // Cast jacobian to Eigen::Matrix format
            jacobian.row(n) = Eigen::Map<const Eigen::RowVectorXd>(derivativesOld.data().data(), m_numVars);
        }
        return jacobian;
    }

    VXd getDensities() const { return m_sim.getDensities(); }
    ObjectivePtr getObjective() const { return m_objective; }
    FiltersList getFilters() const { return m_filters; }
    ConstraintsList getConstraints() const { return m_constraints; }

    const _Sim &getSimulator() const { return m_sim; }

    void setObjective(ObjectivePtr objective) { m_objective = objective; }
    void setFilters(FiltersList filters) {
        m_filters = filters;
        initializeFiltersDimensions();
    }
    void setConstraints(ConstraintsList constraints) { m_constraints = constraints; }

protected:
    // Resize cache depending on the number of filters used
    void initializeCache() {
        // m_cachedVars[0] -> design vars; m_cachedVars.back() -> physical vars
        m_cachedVars.assign(m_filters.size() + 1, NDVector<Real>(m_sim.NbElementsPerDimension())); // design variables + all filtered
    }

    // Store grid dimensions in all filters
    void initializeFiltersDimensions() {
        for(auto & filter : m_filters)
            filter->setGridDimensions(m_sim.NbElementsPerDimension());
    }

    // Finite element simulator
    _Sim &m_sim;

    // Number of variables to be optimized (== number of elements in the simulator)
    size_t m_numVars;
    
    // Number of constraints in the problem instance
    size_t m_constrNumber;

    // Cached values of all variables (design, ..., filtered_i, ..., physical = filtered_last)
    std::vector<NDVector<Real>> m_cachedVars;

    // Check if the state has been set. True if setVars() was called at least once
    bool m_varsAreSet = false;

    // Objective function
    ObjectivePtr m_objective;

    // Chain of filters applied to the design variables
    // Filters implement a backprop method that allows computing sensitivities
    FiltersList m_filters;

    // Constraints collection. Methods for evaluating constraint and 
    // computing its derivative physical variables are provided
    ConstraintsList m_constraints;  
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONPROBLEM_HH
