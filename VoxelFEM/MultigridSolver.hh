#ifndef MULTIGRIDSOLVER_HH
#define MULTIGRIDSOLVER_HH

#include "NDVector.hh"
#include "TensorProductPolynomialInterpolant.hh"
#include "TensorProductSimulator.hh"
#include <memory>
#include <MeshFEM/ParallelAssembly.hh>

/// A class for multigrid solver
template<size_t... Degrees>
class MultigridSolver {
public:
    constexpr static size_t N = sizeof...(Degrees);
    using TPS          = TensorProductSimulator<Degrees...>;
    using Point        = PointND<N>;
    using VNd          = Eigen::Matrix<Real, N, 1>;
    using MNd          = Eigen::Matrix<Real, N, N>;
    using VField       = typename TPS::VField;
    using EigenNDIndex = typename TPS::EigenNDIndex;

    MultigridSolver(std::shared_ptr<TPS> fineSimulator, size_t numCoarseningLevels) {
        std::vector<size_t> numElemsPerDim(N);
        Eigen::Matrix<size_t, N, 1>::Map(&numElemsPerDim[0]) = fineSimulator->NbElementsPerDimension();
        BBox<VectorND<N>> domain = fineSimulator->domain();

        m_x.reserve(numCoarseningLevels + 1);
        m_b.reserve(numCoarseningLevels + 1);

        // Build multigrid hierarchy, l=0 corresponds to the finest level
        for (size_t l = 0; l <= numCoarseningLevels; ++l) {
            std::shared_ptr<TPS> tps = fineSimulator;
            if (l > 0) { // l = 1, 2, ... correspond to the progressively coarsened grid.
                // We assume for now that we can simply halve the grid resolution at each step.
                for (size_t d = 0; d < N; ++d) {
                    if (numElemsPerDim[d] % 2 == 1)
                        throw std::runtime_error("Grid size currently must be divisible by 2^numCoarseningLevels (nonuniform coarsening not yet implemented)");
                    numElemsPerDim[d] = numElemsPerDim[d] / 2;
                }
                tps = std::shared_ptr<TPS>(new TPS(domain, numElemsPerDim)); // make_shared doesn't call Eigen's aligned `new` overload :( :(
                tps->setETensor(fineSimulator->getETensor());

                // Coarsen the Dirichlet conditions.
                // Even if non-zero Dirichlet conditions are specified at the
                // fine level, they coarsen to zero-Dirichlet conditions at the
                // coarser levels (the coarsened levels solve for *corrections*
                // to an admissible solution).
                // If a fine Dirichlet node falls on the boundary face of a
                // coarse element, all coarse nodes on that boundary face are
                // assigned Dirichlet conditions. If a fine node falls in the
                // interior we would need to make **all** coarse nodes Dirichlet,
                // which seems undesirable--for now we throw an error in this case
                // (assuming Dirichlet conditions are applied only at the grid boundary).
                const auto &finer = *m_sims.back();
                auto &coarser = *tps;
                EigenNDIndex numElemNodes = coarser.NbNodesPerDimensionPerElement();
                for (size_t nfi = 0; nfi < finer.numNodes(); ++nfi) {
                    const auto &nf = finer.getNode(nfi);
                    if (!nf.hasDirichlet()) continue;
                    auto e_and_p = coarser.getElementAndReferenceCoordinates(nf.coordinates());
                    // Get an Nd index representing the coarse element boundary vertex/edge/face
                    // on which `p` appears. The d^th entry of this index is -1 if `p` is not
                    // on the min/max face of the d^th dimension. Otherwise it is the d^th
                    // index entry of the index of the coinciding course element boundary node.
                    Eigen::Array<int, N, 1> p_on_boundary;
                    for (size_t d = 0; d < N; ++d) {
                        Real c = e_and_p.second[d];
                        p_on_boundary[d] = std::abs(c) < 1e-9 ? 0 : (std::abs(c - 1.0) < 1e-9 ? (numElemNodes[d] - 1) : -1);
                    }
                    if ((p_on_boundary < 0).all()) throw std::runtime_error("Dirichlet constraints on internal nodes are not supported");

                    // Apply fine node's Dirichlet constraints to all coarse
                    // nodes on the same vertex/edge/face as `p` (necessary and
                    // sufficient to guarantee the coarsened field satisfies
                    // the Dirichlet conditions on the fine mesh).
                    coarser.visitLocalNodeNdIndices([&](const EigenNDIndex &lni) {
                        for (size_t d = 0; d < N; ++d) {
                            if (p_on_boundary[d] == -1) continue;
                            if (p_on_boundary[d] != lni[d]) return; // Local node `lni` is not on the same vertex/edge/face of the coarse element as `p`
                        }
                        auto &nc = coarser.getNode(coarser.elemNodeGlobalIndex(coarser.elementIndexForGridCell(e_and_p.first), lni));
                        nc.setDirichlet(nf.dirichletComponents(), Point::Zero());
                    });
                }
            }
            m_x.push_back(VField::Zero(int(tps->numNodes()), int(N)));
            m_b.push_back(VField::Zero(int(tps->numNodes()), int(N)));
            m_sims.push_back(tps);
        }
    }

    void setSymmetricGaussSeidel(const bool symmetric) {
        m_symmetricGaussSeidel = symmetric;
    }

          TPS &getSimulator(const size_t l)       { return *m_sims.at(l); }
    const TPS &getSimulator(const size_t l) const { return *m_sims.at(l); }

    // Interpolate values from a coarser grid simulator at each node of a finer simulator.
    // TODO: more efficient implementation looping over coarse nodes, then fine nodes inside
    // each incident coarse element.
    using Interpolant = TensorProductPolynomialInterpolant<VectorND<N>, Degrees...>;
    mutable std::vector<Interpolant> _interpolantScratch;
    template<typename ValMatrix>
    void _constructInterpolantScratch(const TPS &coarser, const ValMatrix &values) const {
        _interpolantScratch.resize(coarser.numElements());
        parallel_for_range(coarser.numElements(), [&](size_t ei) {
            size_t n = 0;
            coarser.visitElementNodes(ei, [&](size_t node_index) {
                _interpolantScratch[ei][n++].topRows(values.cols()) = values.row(node_index);
            });
        });
    }

    template<typename ValMatrix>
    void accum_interpolation(const TPS &finer, const TPS &coarser, const ValMatrix &values, ValMatrix &out) const {
        if (values.rows() != coarser.numNodes()) throw std::runtime_error("Invalid input size");
        if ((out.rows() != finer.numNodes()) || (out.cols() != values.cols())) throw std::runtime_error("Invalid output size");
        BENCHMARK_SCOPED_TIMER_SECTION timer("Interpolation");
        _constructInterpolantScratch(coarser, values);
        // accumulate to fine nodes.
        parallel_for_range(finer.numNodes(), [&](size_t i) {
            auto e_and_p = coarser.getElementAndReferenceCoordinates(finer.nodePosition(i));
            out.row(i) += _interpolantScratch[coarser.elementIndexForGridCell(e_and_p.first)](e_and_p.second).topRows(values.cols());
        });
    }

    // WARNING: only supports interpolation of data up to ND.
    template<typename ValMatrix>
    void interpolation(const TPS &finer, const TPS &coarser, const ValMatrix &values, ValMatrix &out) const {
        if (values.rows() != coarser.numNodes()) throw std::runtime_error("Invalid input size");

        BENCHMARK_SCOPED_TIMER_SECTION timer("Interpolation");
        _constructInterpolantScratch(coarser, values);
        // write to fine nodes.
        out.resize(finer.numNodes(), values.cols());
        parallel_for_range(finer.numNodes(), [&](size_t i) {
            auto e_and_p = coarser.getElementAndReferenceCoordinates(finer.nodePosition(i));
            out.row(i) = _interpolantScratch[coarser.elementIndexForGridCell(e_and_p.first)](e_and_p.second).topRows(values.cols());
        });
    }

    // Restrict values from fine grid to the coarse grid.
    // (Apply transposed interpolation operator I^T)
    template<typename ValMatrix>
    void restriction(const TPS &finer, const TPS &coarser, const ValMatrix &values, ValMatrix &result) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Restriction");
        if (values.rows() != finer.numNodes()) throw std::runtime_error("Invalid input size");
#if 1
        using Restriction = TensorProductPolynomialRestriction<Degrees...>;
        auto accumulatePerNodeContrib  = [&](size_t n_f, ValMatrix &out) {
            auto e_and_p = coarser.getElementAndReferenceCoordinates(finer.nodePosition(n_f));
            // get the values of phi(p)
            size_t n_c = 0;
            NDArray<Real, N, (Degrees + 1)...> coeffs = Restriction::evaluate(e_and_p.second);
            coarser.visitElementNodes(e_and_p.first, [&](size_t node_index) {
                    out.row(node_index) += coeffs.get1D(n_c++) * values.row(n_f);
            });
        };
        result.setZero(coarser.numNodes(), values.cols());
        assemble_parallel(accumulatePerNodeContrib, result, finer.numNodes());
#else
        // Values of coarse element shape functions on the fine nodes.
        const auto &phis = getCompressedInterpolationOperator();

        result.setZero(coarser.numNodes(), values.cols());
        assemble_parallel([&](size_t e_c, ValMatrix &out) {
            visitFineElementsInside(coarser, finer, e_c, [&](size_t fi, size_t e_f) {
                size_t ln_c = 0;
                coarser.visitElementNodes(e_c, [&](size_t n_c) {
                    size_t ln_f = 0;
                    finer.visitElementNodes(e_f, [&](size_t n_f) {
                        out.row(n_c) += phis[fi](ln_f++, ln_c) * values.row(n_f);
                    });
                    ++ln_c;
                });
            });
        }, result, coarser.numElements());
#endif
    }

    // A less branchy, slightly faster version of the Gauss-Seidel smoothing operation proposed in
    // [Wu 2016: eqs (11)-(13)]. We define S' = S + M u_old so that we needn't exclude the diagonal block `M`
    // when accumulating `S'`.
    // Then u_new[i] = (b[i] - S[i] - (M_lower * u_old - M_upper * u_new)[i]) / M_ii
    //               = (b[i] - S'[i] + (M * u_old)[i] - (M_lower * u_old - M_upper * u_new)[i]) / M_ii
    //               = (b[i] - S'[i] - (M_upper * (u_new - u_old))[i] + M_ii u_old[i]) / M_ii
    //               = (b[i] - S'[i] - (M_upper * (u_new - u_old))[i]) / M_ii + u_old[i]
    //               = (b[i] - S'[i] - (M * (u_new - u_old))[i]) / M_ii + u_old[i]
    //              := (b[i] - S'[i] - (M * u_diff)[i]) / M_ii + u_old[i]
    // where the second-to-last equality holds because u_diff[j] := u_new[j] - u_old[j] = 0 for j in i..N-1
    // (i.e., M_lower * u_diff == 0).
    void m_smoothNode(const TPS &sim, const EigenNDIndex &globalNode, VField &u, const VField &b, const bool forwardSweep) const {
        using PerElementStiffness = typename TPS::PerElementStiffness;
        const size_t n = sim.nodes().flatIndex(globalNode);
        VNd bMinusSprime = b.row(n); // Accumulates b - S'
        MNd M;
        
        if (!sim.hasBlockK()) {
            M.setZero();
            if (__builtin_expect(!sim.hasCachedElementStiffness(), 1)) {
                // build S and M
                const PerElementStiffness &K0 = sim.fullDensityElementStiffnessMatrix();
                sim.visitIncidentElements(globalNode, [&](size_t ek, size_t localIndex, const typename TPS::ENodesArray &enodes) {
                    Real E = sim.elementYoungModulus(ek);
#if 0
                    VNd S_local = K0.template block<N, N>(localIndex * N, 0) * u.row(enodes[0]).transpose();
                    for (size_t m = 1; m < TPS::numNodesPerElem; ++m)
                        S_local += K0.template block<N, N>(localIndex * N, m * N) * u.row(enodes[m]).transpose();
                    bMinusSprime -= E * S_local;
#else
                    Eigen::Matrix<Real, PerElementStiffness::ColsAtCompileTime, 1> u_local;
                    for (size_t m = 0; m < TPS::numNodesPerElem; ++m)
                        u_local.template segment<N>(N * m) = u.row(enodes[m]).transpose();
					// Leverage symmetry of K0 to access storage-contiguous columns instead of storage-discontiguous rows.
                    bMinusSprime -= E * (K0.template block<PerElementStiffness::ColsAtCompileTime, N>(0, localIndex * N).transpose() * u_local);
#endif
                    M += E * K0.template block<N, N>(localIndex * N, localIndex * N);
                });
            }
            else {
                // build S and M
                sim.visitIncidentElements(globalNode, [&](size_t ek, size_t localIndex, const typename TPS::ENodesArray &enodes) {
                    const PerElementStiffness &Ke = sim.cachedElementStiffnessMatrix(ek);
#if 0
                    Eigen::Matrix<Real, PerElementStiffness::ColsAtCompileTime, 1> u_local;
                    // accumulate contribution of element ek to S
                    for (size_t m = 0; m < TPS::numNodesPerElem; ++m)
                        u_local.template segment<N>(N * m) = u.row(enodes[m]).transpose();
                    bMinusSprime -= Ke.template block<N, PerElementStiffness::ColsAtCompileTime>(localIndex * N, 0) * u_local;
#else
                    // accumulate contribution of element ek to S
                    for (size_t m = 0; m < TPS::numNodesPerElem; ++m)
                        bMinusSprime -= Ke.template block<N, N>(localIndex * N, m * N) * u.row(enodes[m]).transpose();
#endif

                    // accumulate contribution of element ek to M
                    M += Ke.template block<N, N>(localIndex * N, localIndex * N);
                });
            }
        }
        else {
            const auto &K = sim.blockK();
            const size_t end = K.Ap[n + 1];
            // Loop over n^th row of `K` by looping over n^th col and transposing (exploiting symmetry)
            for (size_t ii = K.Ap[n]; ii < end; ++ii) {
                const size_t i = K.Ai[ii];
                bMinusSprime -= (u.row(i) * K.Ax[ii]).transpose(); // Apply tranpose block `Ax[ii].transpose()`
                if (__builtin_expect(i == n, 0)) M = K.Ax[ii];     // Diagonal blocks are symmetric--no transpose needed.
            }
        }

        // Compute displacement u
        const auto &dc = sim.getNode(n).dirichletComponents();
        VNd u_diff = VNd::Zero();
        if (forwardSweep) {
            for (size_t i = 0; i < N; ++i)
                u_diff[i] = (bMinusSprime[i] - M.row(i) * u_diff) * (double(!dc.has(i)) / M(i, i));
        }
        else {
            for (int i = N - 1; i >= 0; --i)
                u_diff[i] = (bMinusSprime[i] - M.row(i) * u_diff) * (double(!dc.has(i)) / M(i, i));
        }
        u.row(n) += u_diff;
    }

    // smoothing at level l via Gauss-Seidel
    // Pre/postcondition: u has Dirichlet conditions applied
    void smoothing(const size_t l, VField &u, const VField &b, const bool forwardSweep = true) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Smoothing");

        auto &sim = getSimulator(l);
        const size_t nn = sim.numNodes();
        const auto &nodes = sim.nodes();

        if ((u.rows() != nn) || (b.rows() != nn)) throw std::runtime_error("Invalid input size");

        for (size_t n_ = 0; n_ < nn; ++n_) {
            size_t n = (forwardSweep) ? n_ : nn - (n_ + 1);
            m_smoothNode(sim, TPS::eigenNDIndexWrapper(nodes.unflattenIndex(n)), u, b, forwardSweep);
        }
    }

    template<class F>
    void visitNodesMulticolored(const size_t l, F &&visitor, bool forwardSweep = true, bool parallel = true) const {
        using ENI = typename TPS::ElementNodeIndexer;
        const auto &sim = getSimulator(l);
        const EigenNDIndex NbNodesPerDimensionPerElement = sim.NbNodesPerDimensionPerElement();
        const EigenNDIndex NbNodesPerDimension           = sim.NbNodesPerDimension();

        // Process one color (local node of the reference element) at a time
        for (size_t i = 0; i < ENI::size(); ++i) {
            size_t lni = forwardSweep ? i : (ENI::size() - i - 1); // Process colors in reverse order for reverse sweep
            EigenNDIndex lni_nd = TPS::eigenNDIndexWrapper(ENI::unflattenIndex(lni));

            // For nodes on element boundary, we need to advance two elements
            // in that direction to reach a node of the same color (advancing
            // by 1 reaches a different color due to element overlaps).
            // For internal nodes, we advance by one element.
            EigenNDIndex nodeIncrements;
            for (size_t d = 0; d < N; ++d) {
                bool isBoundary = (lni_nd[d] == 0) || (lni_nd[d] == (NbNodesPerDimensionPerElement[d] - 1));
                // Adding (NbNodesPerDimensionPerElement[d] - 1) to the node index advances by one element in the d direction.
                nodeIncrements[d] = (1 + isBoundary) * (NbNodesPerDimensionPerElement[d] - 1);
            }

            // Solve for largest numNodesOfColorPerDim such that:
            //      lni_nd + nodeIncrements * (numNodesOfColorPerDim - 1) <= NbNodesPerDimension - 1
            EigenNDIndex numNodesOfColorPerDim = (NbNodesPerDimension - 1 - lni_nd) / nodeIncrements + 1;

            // Visit nodes of the current color in an arbitrary order.
            auto processNode = [&](size_t i) {
                EigenNDIndex colorNodeIdx;
                NDVector<Real>::unflattenIndex(i, numNodesOfColorPerDim, colorNodeIdx);
                visitor(lni_nd + (colorNodeIdx * nodeIncrements));
            };
            const size_t nn = numNodesOfColorPerDim.prod();
            if (parallel) {
                parallel_for_range(nn, processNode);
            }
            else {
                for (size_t i = 0; i < nn; ++i)
                    processNode(i);
            }
        }
    }

    Eigen::VectorXi debugMulticolorVisit() const {
        Eigen::VectorXi result(getSimulator(0).numNodes());
        size_t i = 0;
        const auto &nodes = getSimulator(0).nodes();
        visitNodesMulticolored(0, [&](const EigenNDIndex &globalNode) { result[nodes.flatIndex(globalNode)] = i++; }, /* forwardSweep */ true, /* parallel */ false);
        return result;
    }

    void smoothingMulticoloredGS(const size_t l, VField &u, const VField &b, const bool forwardSweep = true) const {
        const auto &sim = getSimulator(l);
        BENCHMARK_SCOPED_TIMER_SECTION timer("smoothingMulticoloredGS " + sim.description());
        visitNodesMulticolored(l, [&](const EigenNDIndex &globalNode) { m_smoothNode(sim, globalNode, u, b, forwardSweep); }, forwardSweep, /* parallel */ true);
    }

    // Compute K * u for the finest grid (ignoring Dirichlet conditions)
    VField applyK(const VField &u) { return applyK(0, u); }

    // Compute K * u for the l^th grid (ignoring Dirichlet conditions)
    VField applyK(const size_t l, const VField &u) {
        VField result(u.rows(), u.cols());
        applyK(l, u, result);
        return result;
    }

    // Compute K * u for the l^th grid (ignoring Dirichlet conditions)
    void applyK(const size_t l, const VField &u, VField &result) {
        const auto &sim = *m_sims.at(l);
        if ((l > 0) && !sim.hasCachedElementStiffness()) updateElementStiffnessMatrices();
        if (sim.hasBlockK()) sim.applyBlockK(u, result);
        else                 sim.applyK     (u, result);
    }

    // Zero out the variables in `u` with pin constraints for the finest grid
    void zeroOutDirichletComponents(VField &u) const { zeroOutDirichletComponents(0, u); }

    // Zero out the variables in `u` with pin constraints for the l^th grid
    void zeroOutDirichletComponents(const size_t l, VField &u) const {
        BENCHMARK_SCOPED_TIMER_SECTION timer("zeroOutDirichletComponents");
        const auto &sim = *m_sims[l];

        const size_t nn = m_sims[l]->numNodes();
        parallel_for_range(nn, [&](size_t ni) {
            const Node<N> &node = sim.getNode(ni);
            if (node.hasDirichlet()) {
                for (size_t d = 0; d < N; ++d) {
                    if (node.dirichletComponents().has(d))
                        u(ni, d) = 0;
                }
            }
        });
    }

    // Modify `u` so that it satisfies the Dirichlet conditions for the finest grid
    void enforceDirichletConditions(VField &u) const { enforceDirichletConditions(0, u, false); }

    // Modify `u` so that it satisfies the Dirichlet conditions for the l^th grid.
    // The imposed values are either the ones stored in the simulator's Dirichlet condition
    // data structure (if `zero` is false), or zeros (if `zero` is true).
    void enforceDirichletConditions(const size_t l, VField &u, bool zero) const {
        const auto &sim = *m_sims[l];
        const size_t nn = m_sims[l]->numNodes();
        parallel_for_range(nn, [&](size_t ni) {
            const Node<N> &node = sim.getNode(ni);
            if (node.hasDirichlet()) {
                for (size_t d = 0; d < N; ++d) {
                    if (node.dirichletComponents().has(d))
                        u(ni, d) = zero ? 0.0 : node.dirichletDisplacement()[d];
                }
            }
        });
    }

    // Compute the residual at level l from displacement u and force b
    VField computeResidual(const size_t l, const VField &u, const VField &b) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Compute residual");

        VField residual = b - applyK(l, u);

        // Apply Dirichlet conditions:
        // If there is a Dirichlet constraint on this displacement component
        // the residual should be zero (a single smoothing iteration will
        // enforce the Dirichlet condition exactly).
        zeroOutDirichletComponents(l, residual);

        return residual;
    }

    void updateElementStiffnessMatrices() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("updateElementStiffnessMatrices");

        // Caching the fine grid element stiffnesss matrices is actually a dramatic pessimization,
        // likely due to overfilling the on-CPU cache..
        // getSimulator(0).cacheElementStiffnessMatrices();

        const size_t numLevels = m_sims.size();
        for (size_t l = 1; l < numLevels; ++l)
            buildPESCoarse(l);
    }

    void updateBlockKs() {
        const size_t numLevels = m_sims.size() - 1; // We don't store a blockK at the finest level (a direct solve is used there...)
        if (!buildBlockStiffnessMatrices) {
            for (size_t l = 0; l < numLevels; ++l)
                getSimulator(l).clearBlockK();
        }

        BENCHMARK_SCOPED_TIMER_SECTION timer("updateBlockKs");

        if (!buildFinestBlockStiffnessMatrix)
            getSimulator(0).clearBlockK();

        for (size_t l = buildFinestBlockStiffnessMatrix ? 0 : 1; l < numLevels; ++l)
            getSimulator(l).updateBlockK();
    }

    // Do a complete V-Cycle to solve the multigrid system
    // if `zeroDirichlet` is true, we replace the Dirichlet constraint values with zero.
    // This is needed for solving residual systems.
    template<class Derived>
    VField solve(const Eigen::MatrixBase<Derived> &u, const VField &f, size_t numSteps, size_t numSmoothingSteps, bool stiffnessUpdated = false,
                 bool zeroDirichlet = false, std::function<void(size_t, const VField &)> it_callback = nullptr, bool fmg = false) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("MG Solver");
        if (!stiffnessUpdated)
            updateElementStiffnessMatrices();
        if (numSteps == 0) return u;

        m_x[0] = u;
        m_b[0] = f;

        if (fmg) {
            fullMultigrid(0, numSmoothingSteps, zeroDirichlet);
            if (it_callback) it_callback(0, m_x[0]);
            for (size_t i = 1; i < numSteps; ++i) {
                vcycle(0, numSmoothingSteps, zeroDirichlet);
                if (it_callback) it_callback(i, m_x[0]);
            }
        }
        else {
            for (size_t i = 0; i < numSteps; ++i) {
                vcycle(0, numSmoothingSteps, zeroDirichlet);
                if (it_callback) it_callback(i, m_x[0]);
            }
        }
        return m_x[0];
    }

    // Approximately solve the system K u = r.
    // We use an initial guess that is more appropriate for the preconditioner application.
    VField applyPreconditionerInv(const VField &r, size_t numSteps, size_t numSmoothingSteps, bool fmg = false) {
        if (numSmoothingSteps == 0) return r;
        return solve(VField::Zero(r.rows(), r.cols()), r, numSteps, numSmoothingSteps, /* stiffnessUpdated= */ true, /* zeroDirichlet= */ true, /* cb = */ nullptr, fmg);
    }

    // Run a full multigrid cycle starting at level `l` of the hierarchy using
    // `numSmoothingSteps` Gauss-Seidel iterations to solve the equation:
    //      A_l m_x[l] = m_b[l]
    // If `residualSystem` is true, we are solving a residual equation and therefore
    // the initial guess/Dirichlet condition values should be set to zero.
    void fullMultigrid(size_t l, size_t numSmoothingSteps, bool residualSystem) {
        const size_t coarsestLevel = m_sims.size() - 1;
        if (l == coarsestLevel) {
            // TODO: support nonzero Dirichlet values in `residualSystem = false` case.
            m_x[l] = getSimulator(l).solve(m_b[l]);
            return;
        }

        const auto &sim        = getSimulator(l);
        const auto &sim_coarse = getSimulator(l + 1);

        // Restrict RHS to coarser level
        restriction(sim, sim_coarse, m_b[l], m_b[l + 1]); 

        // Solve on the coarser grid.
        fullMultigrid(l + 1, numSmoothingSteps, residualSystem);

        // Interpolate the coarse grid's solution to this level.
        interpolation(sim, sim_coarse, m_x[l + 1], m_x[l]);

        // Run a V-cycle to reduce the error at this level.
        vcycle(l, numSmoothingSteps, residualSystem);
    }

    // Run a V-cycle iteration starting at level `l` of the hierarchy using
    // `numSmoothingSteps` Gauss-Seidel iterations to solve the equation:
    //      A_l m_x[l] = m_b[l]
    // **starting from the initial guess already stored in m_x[l]**.
    // If `residualSystem` is true, we are solving a residual equation and
    // therefore the Dirichlet condition values should be set to zero.
    void vcycle(size_t l, size_t numSmoothingSteps, bool residualSystem) {
        BENCHMARK_SCOPED_TIMER_SECTION timer("V Cycle " + m_sims[l]->description());
        const size_t coarsestLevel = m_sims.size() - 1;

        // Direct solve at the coarsest level.
        if (l == coarsestLevel) {
            m_x[l] = getSimulator(l).solve(m_b[l]);
            return;
        }

        const auto &sim        = getSimulator(l);
        const auto &sim_coarse = getSimulator(l + 1);

        enforceDirichletConditions(l, m_x[l], residualSystem);

        // Run Gauss-Seidel iterations at this level.
        for (size_t i = 0; i < numSmoothingSteps; ++i)
            smoothingMulticoloredGS(l, m_x[l], m_b[l], /* forwardSweep */ true);

        // Restrict the smoothed residual to the next coarser level to form the coarse RHS.
        restriction(sim, sim_coarse, computeResidual(l, m_x[l], m_b[l]), m_b[l + 1]);

        // Continue v-cycle on coarser grid, computing correction m_x[l + 1]
        m_x[l + 1].setZero(sim_coarse.numNodes(), N); // initial guess for correction is all zeros.
        vcycle(l + 1, numSmoothingSteps, /* residualSystem = */ true);

        // Interpolate and apply the coarse grid's correction to this level.
        accum_interpolation(sim, sim_coarse, m_x[l + 1], m_x[l]);

        // // Apply Dirichlet conditions (precondition for smoother, which is preserved by smoother).
        // // Note: this technically shouldn't be necessary as the interpolated coarse solution
        // // should have zeros on this grid's Dirichlet-constrained variables.
        // enforceDirichletConditions(l, m_x[l], residualSystem);

        // Re-run Gauss-Seidel iterations at this level.
        for (size_t i = 0; i < numSmoothingSteps; ++i)
            smoothingMulticoloredGS(l, m_x[l], m_b[l], /* forwardSweep */ !m_symmetricGaussSeidel);
    }

    // Build a compressed form of the interpolation operator `I`:
    // evaluate the shape functions' value on the nodes of all fine elements it contains.
    static constexpr size_t numFineElemsPerCoarse = 1 << N; // There are 2^N fine elements within each coarse element.
    using Phi = Eigen::Matrix<Real, TPS::numNodesPerElem, TPS::numNodesPerElem>;
    std::array<Phi, 1 << N> getCompressedInterpolationOperator() const {
        // Assumes coarse and fine elements are the same (i.e., scaled versions of each other),
        // so the level doesn't matter.
        const auto &finer = getSimulator(0); // arbitrary
        std::array<Phi, numFineElemsPerCoarse> phis; // phis[fine_e](fine_n, coarse_n) holds:
                                            //   coarse shape function `coarse_n` evaluated on node `fine_n`
                                            //   of fine element `fine_e`
        Point origin = finer.nodePosition(finer.flattenedFirstNodeOfElement1D(0));
        for (size_t fine_n = 0; fine_n < TPS::numNodesPerElem; ++fine_n) {
            // "Half-canonical coordinates" of fine_n (in [0, 0.5]^N)
            const Point fineNodeHalfCoords = ((finer.nodePosition(finer.elemNodeGlobalIndex(0, fine_n)) - origin).array() / (2.0 * finer.getStretchings().array())).matrix();
            for (size_t fi = 0; fi < numFineElemsPerCoarse; ++fi) {
                // Get canonical coordinates (in [0, 1]^N) within the coarse element.
                Point fineNodePosition;
                for (size_t d = 0; d < N; ++d)
                    fineNodePosition[d] = fineNodeHalfCoords[d] + ((fi & (1 << d)) ? 0.5 : 0.0);

                // Evaluate all coarse shape functions on this fine node.
                using Restriction = TensorProductPolynomialRestriction<Degrees...>;
                auto coeffs = Restriction::evaluate(fineNodePosition);
                phis[fi].row(fine_n) = Eigen::Map<const Eigen::Matrix<Real, coeffs.size(), 1>>(coeffs.data());
            }
        }
        return phis;
    }
    // There are up to 2^N fine elements contained in coarse element e_c.
    // These are found by doubling e_c's ND index and adding offsets in
    // {0, 1} to each index component.
    // Call `visitor(fi, e_f)` for each, where `fi` is a local linear index
    // in 0..2^N and `e_f` is the global element index in `finer.
    template<class Visitor>
    void visitFineElementsInside(const TPS &coarser, const TPS &finer, size_t e_c, const Visitor &visitor) const {
        auto e_c_ndIndex = coarser.ndIndexForElement(e_c);
        for (size_t fi = 0; fi < numFineElemsPerCoarse; ++fi) {
            EigenNDIndex e_f_ndIndex;
            for (size_t d = 0; d < N; ++d)
                e_f_ndIndex[d] = 2 * e_c_ndIndex[d] + ((fi & (1 << d)) ? 1 : 0);
            if ((e_f_ndIndex >= finer.NbElementsPerDimension()).any()) continue; // out of bounds
            size_t e_f = finer.elementIndexForGridCell(e_f_ndIndex);
            visitor(fi, e_f);
        }
    }

    // Compute per element stiffness matrices of coarse grid from the Ke's of the fine grid
    // and stores it in the coarse TensorProductSimulator
    void buildPESCoarse(const size_t l_coarse) {
        using PerElementStiffness = typename TPS::PerElementStiffness;

        if (l_coarse < 1) throw std::runtime_error("Level l should be at least 1. Current l: " + std::to_string(l_coarse));
        if (l_coarse >= m_sims.size()) throw std::runtime_error("Level l too large: " + std::to_string(l_coarse) + "vs" + std::to_string(m_sims.size()));

        const auto &finer = getSimulator(l_coarse - 1);
        auto     &coarser = getSimulator(l_coarse);

        const auto &phis = getCompressedInterpolationOperator();

        aligned_std_vector<PerElementStiffness> KeVec(coarser.numElements());

        // Build a compressed form of the interpolation operator I:
        // cache the coarse shape functions' value on the fine nodes it contains.
        using FlattenKe      = Eigen::Map<      Eigen::Matrix<Real, TPS::KeSize * N, TPS::KeSize / N>>;
        using FlattenKeConst = Eigen::Map<const Eigen::Matrix<Real, TPS::KeSize * N, TPS::KeSize / N>>;
        using FlattenAccum   = Eigen::Map<      Eigen::Matrix<Real, TPS::KeSize * N, 1              >>;

        // add I^T Ke_f I to the coarse element's stiffness matrix,
        // where I[N * i + c, N * j + d] = phi(i, j) 𝛅_cd for c, d in range(N)
        // Note for any matrix `M` of same shape as Ke_f:
        //      [M I]_{ab} = sum_k M[a, k] I[k, b]
        //      [M I][:, b] = sum_k M[:, k] phi(k / N, b / N) 𝛅_{b % N, k % N}
        //      ==> Flatten(M I) = Flatten(M) * phi
        // Likewise,  I^T Ke_f = (Ke_f I)^T ==> Flatten((Ke_f I)^T) = Flatten(Ke_f) * phi
        auto accumulateCoarsenedStiffnessMatrix = [&](size_t fi, const PerElementStiffness &Ke_f, auto &Ke_c) {
            PerElementStiffness It_Ke_f;

            const Phi &phi = phis[fi];
            FlattenKe(It_Ke_f.data()) = FlattenKeConst(Ke_f.data()) * phi;
            It_Ke_f.transposeInPlace();
            FlattenKe(Ke_c.data()) += FlattenKeConst(It_Ke_f.data()) * phi;
        };

        if (!finer.hasCachedElementStiffness()) {
            // Note at the first (most expensive) level of coarsening, we accumulate
            // a scalar multiple of I_f^T K0 I_f, which for all coarse elements is the same for
            // a given nested fine element `f`. Therefore, we can cache these 2^N coarsened
            // stiffness matrices.
            std::array<PerElementStiffness, numFineElemsPerCoarse> coarsenedK0s;
            for (size_t fi = 0; fi < numFineElemsPerCoarse; ++fi) {
                coarsenedK0s[fi].setZero();
                accumulateCoarsenedStiffnessMatrix(fi, finer.fullDensityElementStiffnessMatrix(), coarsenedK0s[fi]);
            }

            parallel_for_range(coarser.numElements(), [&](size_t e_c) {
                auto &Ke_c = KeVec[e_c];
                Ke_c.setZero();
                visitFineElementsInside(coarser, finer, e_c, [&, e_c](size_t fi, size_t e_f) {
                    Ke_c += finer.elementYoungModulus(e_f) * coarsenedK0s[fi];
                });
            });
        }
        else {
            parallel_for_range(coarser.numElements(), [&](size_t e_c) {
                auto &Ke_c = KeVec[e_c];
                Ke_c.setZero();
                visitFineElementsInside(coarser, finer, e_c, [&, e_c](size_t fi, size_t e_f) {
                    accumulateCoarsenedStiffnessMatrix(fi, finer.cachedElementStiffnessMatrix(e_f), Ke_c);
                });
            });
        }

        coarser.cacheCustomElementStiffnessMatrices(std::move(KeVec));
    }

    // Apply PCG to solve K x = b starting from initial guess `x = u`.
    // Uses our multigrid solver as a preconditioning operator `M^{-1}`.
    // If `fmg` is true, the fullMultigrid solver is used; otherwise plain V-cycles.
    // The stopping criterion is when the force residual becomes small relative to the applied forces:
    //      ||K x - b|| / ||b|| < tol.
    // This is different from the stopping criterion in [Shewchuk 94]; it is
    // strongly preferred for its physical meaning and because it **independent
    // of the initial guess u** and of the preconditioner employed.
    VField preconditionedConjugateGradient(const VField &u, const VField &b, const size_t maxIter,
                                           const Real tol, std::function<void(size_t, const VField &, const VField &)> it_callback = nullptr,
                                           size_t mgIterations = 1,
                                           size_t mgSmoothingIterations = 1,
                                           bool fmg = false) {
        if (u.rows() != b.rows()) throw std::runtime_error("x and b should have the same size");
        if (u.rows() != m_sims[0]->numNodes()) throw std::runtime_error("size of input and number of nodes don't correspond");

        VField x(u);
        enforceDirichletConditions(x);

        updateElementStiffnessMatrices(); // must happen before computeResidual...
        updateBlockKs();

        BENCHMARK_SCOPED_TIMER_SECTION timer("CG Iterations");
        Real b_norm_sq = b.squaredNorm();

        VField r = computeResidual(0, x, b);
        Real r_Minv_r = 0; // after first iteration, will hold ||r||^2 in M^{-1} norm

        VField Ad, s, d;

        // We use a slightly restructured variant of the standard PCG algorithm
        // (e.g., [Shewchuk 94]) that eliminates some duplicate code and
        // crucially avoids a needless and costly application of the
        // preconditioner at the end of the final iteration.
        //
        // Loop invariant:
        //      r holds current residual
        //      d holds previous search direction (empty on first iteration)
        //      After the first iteration, r_Minv_r holds the squared M-inverse norm of `r`, where M is the preconditioner.
        size_t i;
        while ((i++ < maxIter) && (r.squaredNorm() > tol * tol * b_norm_sq)) {
            // Compute new search direction by making preconditioned residual conjugate to previous direction
            s = applyPreconditionerInv(r, mgIterations, mgSmoothingIterations, fmg); // s = M^{-1} r
            zeroOutDirichletComponents(s);
            Real r_Minv_r_old = r_Minv_r;
            r_Minv_r = r.cwiseProduct(s).sum();
            if (d.size()) d = s + (r_Minv_r / r_Minv_r_old) * d; // Make s conjugate to previous direction `d`
            else          d = s;                                 // Previous direction doesn't exist; d = M^{-1} r directly.

            applyK(0, d, Ad);
            zeroOutDirichletComponents(Ad);

            Real alpha = r_Minv_r / d.cwiseProduct(Ad).sum();   // optimal step length
            x += alpha * d;
            r -= alpha * Ad;
            if (it_callback) {
                BENCHMARK_SCOPED_TIMER_SECTION cbtimer("Callback");
                it_callback(i, x, r);
            }
        }
        return x;
    }

    const VField &debug_get_x(int l) { return m_x.at(l); }
    const VField &debug_get_b(int l) { return m_b.at(l); }

    // Whether to build and use an explicit sparse-block representation of the
    // stiffness matrix for the implementations of `applyK` and `m_smoothNode`.
    // If `false`, these operations are implemented in a more memory efficient
    // "matrix-free" way by accumulating contributions from a node's incident
    // elements on-the-fly. This on-the-fly approach ends up performing many
    // more floating point operations, but it is generally *faster* at the finest
    // level (where a scaling of a single reference per-element-stiffness
    // matrix can be used). At the coarser levels, the block stiffness matrix
    // involves fewer FLOPs and less memory access.
    bool buildBlockStiffnessMatrices = true;
    // Whether to also build a block stiffness matrix for the finest level when
    // `buildBlockStiffnessMatrices` is `true`. As discussed above, this is
    // generally undesirable.
    bool buildFinestBlockStiffnessMatrix = false;

private:    
    std::vector<std::shared_ptr<TPS>> m_sims;

    std::vector<VField> m_x;
    std::vector<VField> m_b;

    bool m_symmetricGaussSeidel = true;
};

// Metafunction to get multigrid solver for a particular TensorProductSimulator
// instantiation.
template<typename T>
struct MGSolverForTPS;

template<size_t... Degrees>
struct MGSolverForTPS<TensorProductSimulator<Degrees...>> {
    using type = MultigridSolver<Degrees...>;
};

#endif
