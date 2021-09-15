#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONFILTER_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONFILTER_HH

#include "NDVector.hh"
#include "TensorProductSimulator.hh"
#include "TopologyOptimizationProblem.hh"

#include <functional>

using EigenNDIndex = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
template<typename _Sim> class TopologyOptimizationProblem;

struct Filter {
    template<typename _Sim>
    friend class TopologyOptimizationProblem;

    virtual ~Filter() {};

    /// Forward propagation.
    /// @param[in] in original variables
    /// @param[out] out fitered variables
    virtual void apply(const NDVector<Real> &in, NDVector<Real> &out) = 0;

    /// Backward propagation.
    /// @param[in] in chain rule partial result
    /// @param[in] vars value of variables, could be needed to evaluate current derivatives
    /// @param[out] out derivatives accounting for current filter contribution
    virtual void backprop(const NDVector<Real> &in, const NDVector<Real> &vars, NDVector<Real> &out) const = 0;

    EigenNDIndex getGridDimensions() const { return m_gridDims; }

    // Throw an error if filter is used or modified without specifying the grid dimensions
    void checkGridDimensionsAreSet() const {
        if(!gridDimsAreSet)
            throw std::runtime_error("Filter grid dimensions not set. "
                "Initialize a TopologyOpimizationProblem object with this filter before using it.");
    }

protected:
    // Initialize grid dimensions and eventually set members that depend on it
    // Called by friend class TopologyOptimizationProblem to ensure that grid dimensions match with problem ones.
    virtual void setGridDimensions(const EigenNDIndex &gridDims) { 
        m_gridDims = gridDims;
        gridDimsAreSet = true;
    }

    // Number of elements in each dimension of the domain
    // Needed by all the filters to allow reshaping and testing in Python
    EigenNDIndex m_gridDims;

    // True if a TopologyOptimizationProblem has been used to fully initialize the filter
    bool gridDimsAreSet = false;
};

struct ProjectionFilter : public Filter {
    ProjectionFilter() { }

    void apply(const NDVector<Real> &in, NDVector<Real> &out) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("applyProjectionFilter");
        out.flattened() = 0.5*(tanh(0.5*m_beta) + tanh(m_beta*(in.flattened().array() - 0.5))) / tanh(0.5*m_beta);
    }

    void backprop(const NDVector<Real> &in, const NDVector<Real> &vars, NDVector<Real> &out) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("backpropProjectionFilter");
        out.flattened() = in.flattened().array() * 0.5*m_beta*(1.0 - tanh(m_beta*(vars.flattened().array() - 0.5))*tanh(m_beta*(vars.flattened().array() - 0.5))) / tanh(0.5*m_beta);
    }

    Real getBeta() const { return m_beta; }
    void setBeta(Real beta) {
        if(beta <= 0)
            throw std::runtime_error("Beta parameter has to be positive (received beta = " + std::to_string(beta) + ")");
        m_beta = beta;
    }

private:
    // Beta defines the steepness of the Heaviside-like projection 
    // (for beta->inf, projection is the step function)
    Real m_beta = 1.0;
};

struct PythonFilter : public Filter {
    PythonFilter() { }

    using ApplyCallback    = std::function<void(Eigen::Ref<const Eigen::VectorXd> in, Eigen::Ref<      Eigen::VectorXd> out)>;
    using BackpropCallback = std::function<void(Eigen::Ref<const Eigen::VectorXd> in, Eigen::Ref<const Eigen::VectorXd> vars, Eigen::Ref<Eigen::VectorXd> out)>;

    void apply(const NDVector<Real> &in, NDVector<Real> &out) override {
        if (!apply_cb) throw std::runtime_error("Apply callback must be configured");
        Eigen::VectorXd result(out.flattened().size());
        apply_cb(in.flattened(), result);
        out.flattened() = result;
    }

    void backprop(const NDVector<Real> &in, const NDVector<Real> &vars, NDVector<Real> &out) const override {
        if (!backprop_cb) throw std::runtime_error("Backprop callback must be configured");
        Eigen::VectorXd result(out.flattened().size());
        backprop_cb(in.flattened(), vars.flattened(), result);
        out.flattened() = result;
    }

    ApplyCallback apply_cb;
    BackpropCallback backprop_cb;
};

struct SmoothingFilter : public Filter {

    SmoothingFilter() { }
    
    void apply(const NDVector<Real> &in, NDVector<Real> &out) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("applySmoothingFilter");
        m_A.applyRaw(in.data().data(), out.data().data());
    }

    void backprop(const NDVector<Real> &in, const NDVector<Real> &vars, NDVector<Real> &out) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("backpropSmoothingFilter");
        // Note: derivatives are independent of variable values (vars argument not used)
        m_A.applyRaw(in.data().data(), out.data().data(), /* transpose = */ true);
    }

    Real getRadius() const { return m_radius; }
    void setRadius(Real r) {
        m_radius = r;
        updateMatrix();
    }

private:
    void setGridDimensions(const EigenNDIndex &gridDims) override {
        m_gridDims = gridDims;
        gridDimsAreSet = true;
        updateMatrix();  // once the mesh shape is known, compute and cache filter derivatives
    }

    void updateMatrix() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("updateSmoothingFilter");
        checkGridDimensionsAreSet();
        const size_t n = m_gridDims.prod();
        const size_t d = m_gridDims.size();
        TripletMatrix<> A(n, n);
        size_t nonZerosUpperBound = n * std::pow(2 * m_radius + 1, d); // as if each element had cubic neighborhood
        A.reserve(nonZerosUpperBound);
        for (size_t i = 0; i < n; i++) {
            size_t start = A.nnz();
            visitStencil(i, [&](size_t k) { A.addNZ(i, k, 1.0); }); // Generate new entries
            size_t numInStencil = A.nnz() - start;
            for (size_t j = 0; j < numInStencil; ++j)
                A.nz[start + j].v = 1.0 / numInStencil;
        }
        assert(A.nnz() <= nonZerosUpperBound);  // Check for undesirable reallocation
        m_A.setFromTMatrix(A);
    }

    // Visit the neighborhood of an element and apply the callback to each of the neighbors
    template<typename Visitor>
    void visitStencil(size_t variableIndex, Visitor &&callback) const {
        NDVector<Real>::visitNeighbors(variableIndex, m_gridDims, m_radius, callback);
    }

    SuiteSparseMatrix m_A; // Matrix representing this linear filter.

    // Filter radius in terms of grid elements
    size_t m_radius = 1;
};

struct LangelaarFilter : public Filter {
    LangelaarFilter() { }
    
    void apply(const NDVector<Real> &in, NDVector<Real> &out) override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("applyLangelaarFilter");
        checkGridDimensionsAreSet();
        // Note: smax approximation overestimates max function and can lead to overshoot in Langelaar-filtered densities values, possibly > 1.
        //       The variable bounds in the optimizer will take care of keeping the density field in [0, 1]
        size_t N = m_gridDims.size();
        for (size_t layer = 0; layer < m_gridDims[N-1]; layer++) {
            if (layer == 0) { // layer closer to the baseplate
                visitLayer(layer, [&](size_t i) { 
                    m_cachedSmax[i] = 1; // cache smax for backpropagation (first layer supported by baseplate: smax of support is 1)
                    out[i] = in[i];      // all blueprint densities are actually manufacturable
                });
            }
            else { // all upper the layers
                visitLayer(layer, [&](size_t i) {
                    m_cachedSmax[i] = smax(out, NDVector<Real>::unflattenIndex(i, m_gridDims));
                    out[i] = smin(in[i], m_cachedSmax[i]);
                });
            }
        }
        m_cachedFiltered = out; // cache value of filtered variables for backpropagation
    }

    void backprop(const NDVector<Real> &in, const NDVector<Real> &vars, NDVector<Real> &out) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("backpropLangelaarFilter");
        checkGridDimensionsAreSet();
        NDVector<Real> lambdas(m_gridDims);
        computeLagrangeMultipliers(in, vars, lambdas);
        size_t N = m_gridDims.size();
        for (size_t layer = 0; layer < m_gridDims[N-1]; layer++)
            visitLayer(layer, [&](size_t i) { out[i] = lambdas[i]*dsmin_dx1(vars[i], m_cachedSmax[i]); });
    }

private:
    void setGridDimensions(const EigenNDIndex &gridDims) override
    {
        m_gridDims = gridDims;
        m_cachedSmax.Resize(gridDims);
        m_cachedFiltered.Resize(gridDims);
        gridDimsAreSet = true;
    }

    void computeLagrangeMultipliers(const NDVector<Real> &in, const NDVector<Real> &vars, NDVector<Real> &lambdas) const
    {
        BENCHMARK_SCOPED_TIMER_SECTION timer("multipliersLangelaarFilter");
        checkGridDimensionsAreSet();
        size_t N = m_gridDims.size();
        std::vector<int> idx(N);
        std::vector<size_t> varIdx(N), derIdx(N);
        for (int layer = m_gridDims[1]-1; layer >= 0; layer--) {
            visitLayer(layer, [&](size_t i) { lambdas[i] = in[i]; });
            if (layer < m_gridDims[1]-1) {                        // for every layer but the upper one
                visitLayer(layer+1, [&](size_t i) {               // i is the index of the variable w.r.t. derivative is evaluated
                    std::vector<size_t> varIdx = NDVector<Real>::unflattenIndex(i, m_gridDims);
                    visitSupportingRegion(varIdx, [&](size_t k) { // k is the index of the variable w.r.t. derivative is taken
                        std::vector<size_t> derIdx = NDVector<Real>::unflattenIndex(k, m_gridDims);
                        lambdas[k] += lambdas[i]*sminDerivative(vars, varIdx, derIdx);
                    });
                });
            }
        }
    }

    // Visit a full layer and apply the callback to each of the voxels
    void visitLayer(size_t layerIndex, const std::function<void(size_t)> &callback) const {
        NDVector<Real>::visitLayer(layerIndex, m_gridDims, callback);
    }

    // Visit the supporting region of an element and apply the callback to each of the supporting voxels
    void visitSupportingRegion(const std::vector<size_t> &variableIndices, const std::function<void(size_t)> &callback) const {
        NDVector<Real>::visitSupportingRegion(variableIndices, m_gridDims, callback);
    }

    // Evaluate smax function using as input the densities in support of voxel at location defined by indices
    Real smax(const NDVector<Real> &vars, const std::vector<size_t> &indices) const {
        Real sum = 0;
        visitSupportingRegion(indices, [&](size_t i) { sum += std::pow(vars[i], m_P); });
        return std::pow(sum, 1/m_Q);
    }

    // Evaluate smax derivative w.r.t. one of the directly supporting variables
    Real smaxDerivative(const NDVector<Real> &vars, const std::vector<size_t> &indices, const std::vector<size_t> &derivativeIndices) const {
        Real sum = 0;
        visitSupportingRegion(indices, [&](size_t i) { sum += std::pow(vars[i], m_P); });
        return m_P*std::pow(vars(derivativeIndices), m_P-1)/m_Q * std::pow(sum, 1/m_Q-1);
    }

    // Evaluate smin derivative w.r.t. one of the directly supporting variables
    Real sminDerivative(const NDVector<Real> &vars, const std::vector<size_t> &indices, const std::vector<size_t> &derivativeIndices) const {
        // Note: derivativeIndices identify a supporting voxel of the one identified by indices
        return dsmin_dx2(vars(indices), m_cachedSmax(indices)) * smaxDerivative(m_cachedFiltered, indices, derivativeIndices);
    }

    Real smin(Real x1, Real x2) const { return 0.5*(x1 + x2 - std::pow((x1-x2)*(x1-x2)+m_epsilon, 0.5) + std::pow(m_epsilon, 0.5)); }
    Real dsmin_dx1(Real x1, Real x2) const { return 0.5*(1 - (x1-x2)*std::pow((x1-x2)*(x1-x2)+m_epsilon, -0.5)); }
    Real dsmin_dx2(Real x1, Real x2) const { return 0.5*(1 + (x1-x2)*std::pow((x1-x2)*(x1-x2)+m_epsilon, -0.5)); }

    // Physical variables
    NDVector<Real> m_cachedFiltered;

    // Result of smax in supporting regions
    NDVector<Real> m_cachedSmax;

    // Coefficient used in the approximation of the min function 
    Real m_epsilon = 1e-4;
    
    // Value defining the P-norm that approximates the max function
    Real m_P = 40;

    // Exponent used to correct the P-norm overestimation
    Real m_Q = 40 - 1.58;
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONFILTER_HH
