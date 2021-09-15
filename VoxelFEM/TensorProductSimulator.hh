#ifndef MESHFEM_TENSORPRODUCTSIMULATOR_HH
#define MESHFEM_TENSORPRODUCTSIMULATOR_HH


#include "NDVector.hh"
#include "TensorProductPolynomialInterpolant.hh"
#include "TensorProductQuadrature.hh"

#include <MeshFEM/SparseMatrices.hh>
#include <MeshFEM/Materials.hh>
#include <MeshFEM/Parallelism.hh>
#include <MeshFEM/ParallelAssembly.hh>
#include <MeshFEM/Fields.hh>
#include <MeshFEM/Geometry.hh>

#include <MeshFEM/MeshIO.hh>
#include <MeshFEM/MSHFieldWriter.hh>
#include <MeshFEM/MSHFieldParser.hh>
#include <MeshFEM/BoundaryConditions.hh>
#include <MeshFEM/util.h>

#include <string>
#include <cmath>
#include <bitset>


/// A struct for the Node in the mesh
/// \tparam N: dimension of the problem
template <size_t N>
struct Node {
    Node<N>() = default;

    Node<N>(VectorND<N> const &coords)
        : m_coordinates(coords) { }

    // Node coordinates setter and getter
    void setCoordinates(const VectorND<N> &coords) { m_coordinates = coords; }
    const VectorND<N> &coordinates() const { return m_coordinates; }

    bool hasDirichlet() const { return m_dirichletComponents.hasAny(N); }
    const ComponentMask &dirichletComponents()   const { return m_dirichletComponents; }
          ComponentMask &dirichletComponents()         { return m_dirichletComponents; }
    const VectorND<N>   &dirichletDisplacement() const { return m_dirichletDisplacement; }
          VectorND<N>   &dirichletDisplacement()       { return m_dirichletDisplacement; }
    size_t const dirichletRegionIdx() const { return m_dirichletRegionIdx; }

    void setDirichlet(VectorND<N> const &value) { m_dirichletDisplacement = value; }

    void setDirichlet(ComponentMask mask, const VectorND<N> &val) {
        for (size_t c = 0; c < N; ++c) {
            if (!mask.has(c)) continue;
            // If a new component is being constrained, merge
            if (!m_dirichletComponents.has(c)) {
                m_dirichletComponents.set(c);
                m_dirichletDisplacement[c] = val[c];
            }
            // Otherwise, make sure there isn't a conflict
            else {
                if (std::abs(m_dirichletDisplacement[c] - val[c]) > 1e-10)
                    throw std::runtime_error("Conflicting dirichlet displacements.");
            }
        }
    }

    void setDirichletRegion(size_t idx) {
        if (m_dirichletRegionIdx != 0 && m_dirichletRegionIdx != idx) {
            std::cerr << "WARNING: region traction currently unsupported for vertices "
                      << "belonging to multiple regions" << std::endl;
        }
        m_dirichletRegionIdx = idx;
    }

    // Forces (Neumann-like, they can be imposed also on internal nodes)
    void hasForce(bool value) { m_hasForce = value; }
    void setForce(const VectorND<N> &value) { m_force = value; }

    const bool hasForce() const { return m_hasForce; }
    const VectorND<N> & force() const { return m_force; }

private:
    VectorND<N> m_coordinates;

    // Displacement imposed on the node (Dirichlet-like condition)
    ComponentMask m_dirichletComponents;  // Directions of the imposed displacement
    VectorND<N> m_dirichletDisplacement;  // Displacement amount
    size_t m_dirichletRegionIdx = 0;      // != 0 if any displacement is imposed

    // Force imposed on the node (Neumann-like condition)
    VectorND<N> m_force = VectorND<N>::Zero();   // Nodal force value
    bool m_hasForce = false;                     // True if a Neumann-like condition is imposed
};

/// A struct for a mesh element
/// \tparam Degrees: The degrees of the Lagrange Polynomials of the FEM basis, in each dimension
template<size_t... Degrees>
struct Element_T {
    constexpr static size_t N = sizeof...(Degrees);
    // Number of nodes in the element
    static constexpr size_t nNodes = NumBasisFunctions<Degrees...>::value;
    // Number of vectorial basis function
    static constexpr size_t nVecPhi = N * NumBasisFunctions<Degrees...>::value;

    using ETensor = ElasticityTensor<Real, N>;
    using StiffnessMatrix = Eigen::Matrix<Real, nVecPhi, nVecPhi>;
    using SMatrix = SymmetricMatrixValue<Real, N>;
    using ElementLoad = Eigen::Matrix<Real, N,  nNodes>;
    using Strain = TensorProductPolynomialInterpolant<SMatrix, Degrees...>;
    using VField = Eigen::Matrix<Real, Eigen::Dynamic, N, Eigen::RowMajor>; // compatibility with numpy's row-major default; flattens as x1, y1, z1, x2, ...

    using StiffnessMatrixQuadrature = TensorProductQuadrature<2*Degrees...>;

    // Set the stretching of the element: constant along the axis - no shear or weird deformation)
    // Also caches the corresponding strains
    void setStretchings(const VectorND<N> &Stretchings) {
        m_element_stretch = Stretchings;
        m_strains = Strains<Degrees...>::getStrains(m_element_stretch);
        m_volume = m_element_stretch.prod();
    }

    Real stretching(size_t dimension) const { return m_element_stretch.at(dimension); }

    // Compute the Element stiffness matrix Ke = int_elem [Strain : E : Strain] dV
    // Assumes the element stretchings have been set
    // Ke: Element stiffness matrix
    // E_tensor: Element Elasticity tensor
    // density: Element density
    void Stiffness(StiffnessMatrix &Ke, const ETensor &E_tensor, const Real density) const {
        Real volume = Volume();
        for (size_t i = 0; i < m_strains.size(); ++i) {
            for (size_t j = i; j < m_strains.size(); ++j) {
                Ke(i, j) = StiffnessMatrixQuadrature::integrate(
                        [&](const VectorND<N> &p) {
                            return m_strains[i](p).doubleContract(E_tensor.doubleContract(m_strains[j](p)));
                        }
                );
            }
        }

        Ke *= density * volume; // volume is the norm of the Jacobian of the transformation to the reference element
    }

    // Computes the element stress load
    // l: element stress, load, assigned in the function
    // cstress:  element stress "C : e^{ij}"
    void constantStressLoad(ElementLoad &l, const SMatrix &cstress) const {
        Real volume = Volume();
        // Loop over the nodes
        for (size_t j = 0; j < nNodes; ++j) {
            // Loop over the strain components for the node
            for (size_t i = 0; i < N; ++i) {
                l(i,j) = TensorProductQuadrature<Degrees...>::integrate(
                        [&](const VectorND<N> &p) {
                            return m_strains[N*j + i](p).doubleContract(cstress);
                        }
                );
            }
        }

        l *= volume;
    }

    // Computes the element strainload under contant unit strain e
    // l: Element load, assigned during computation
    // E_tensor: Element elasticity tensor
    // density: element density
    // cstrain: constant unit strain e^{ij}
    void constantStrainLoad(ElementLoad &l,
                            const ETensor &E_tensor,
                            const Real density,
                            const SMatrix &cstrain) const {
        constantStressLoad(l, E_tensor.doubleContract(cstrain));
        l *= density;
    }

    // assigns to e the strain of the element "ni" of the simulator,
    // NodeIndexGetter returns the index of the nodes of element ni
    // in the indexing of the simulator
    template<class NodeIndexGetter>
    void strain(const NodeIndexGetter &ni, const VField &u, Strain &e) const {
        e.coeffs.fill(SMatrix()); // SMatrix zero-initializes
        for (size_t j = 0; j < nNodes; ++j) {
            for (size_t c = 0; c < N; ++c) {
                Real u_comp = u(ni(j), c);
                for (size_t n = 0; n < Strain::size(); ++n)
                    e[n] += u_comp * m_strains[N * j + c][n];
            }
        }
    }

    template<class NodeIndexGetter>
    void averageStrain(const NodeIndexGetter &ni, const VField &u, SMatrix &e) const {
        e.clear();
        for (size_t j = 0; j < nNodes; ++j) {
            for (size_t c = 0; c < N; ++c) {
                // Note: we could optimize this with an "average" or
                // "integrate" method on our TensorProductPolynomialInterpolant
                e += u(ni(j), c) *
                    TensorProductQuadrature<Degrees...>::integrate(
                        [&](const VectorND<N> &p) {
                            return m_strains[N*j + c](p);
                        });
            }
        }
    }

    Real Volume() const {
        return m_volume;
    }

private:
    VectorND<N> m_element_stretch;
    std::vector<typename Strains<Degrees...>::Strains_> m_strains;
    Real m_volume;
};


/// A class for the finite element simulator
/// \tparam Degrees: The degrees of the Lagrange Polynomials of the FEM basis, in each dimension
template<size_t... Degrees>
class TensorProductSimulator  {
public:
    using Element = Element_T<Degrees...>;
    static constexpr size_t N               = sizeof...(Degrees);
    static constexpr size_t numNodesPerElem = Element::nNodes;

    using TMatrix = TripletMatrix<Triplet<Real>>;
    using VField = Eigen::Matrix<Real, Eigen::Dynamic, N, Eigen::RowMajor>; // compatibility with Eigen's row-major default
    using Point = PointND<N>;
    using SMatrix             = typename Element::SMatrix;
    using Strain              = typename Element::Strain;
    using MNd                 = Eigen::Matrix<Real, N, N>;
    using PerElementStiffness = typename Element::StiffnessMatrix;
    using BlockSuiteSparseMatrix = CSCMatrix<SuiteSparse_long, MNd>;
    static constexpr size_t KeSize = PerElementStiffness::RowsAtCompileTime;

    using StiffnessMatrixQuadrature = typename Element::StiffnessMatrixQuadrature;

    using ETensor = ElasticityTensor<Real, N>;

    using EigenNDIndex = Eigen::Array<size_t, N, 1>;
    template<class STLNDIndex>
    static auto eigenNDIndexWrapper(const STLNDIndex &idxs) -> decltype(Eigen::Map<const EigenNDIndex>(idxs.data())) {
        assert(size_t(idxs.size()) == N);
        return Eigen::Map<const EigenNDIndex>(idxs.data());
    }

    using ElementNodeIndexer   = NDArrayIndexer<N, (    Degrees + 1)...>;
    using ElementVertexIndexer = NDArrayIndexer<N, (0 * Degrees + 2)...>;

    // Construct a TPS given the domain (left-lower and right-upper points)
    // and the number of elements per dimension
    TensorProductSimulator(const BBox<VectorND<N>> &domain,
                           const std::vector<size_t> &numElemsPerDim) {
        if (numElemsPerDim.size() != N)
            throw std::runtime_error("Dimension mismatch: " + std::to_string(numElemsPerDim.size()) + " vs " + std::to_string(N));
        m_NbElementsPerDimension = eigenNDIndexWrapper(numElemsPerDim);
        
        // Compute number of nodes belonging to an element, in each dimension
        m_NbNodesPerDimensionPerElement = eigenNDIndexWrapper(std::array<size_t, N>({{(Degrees+1)... }}));

        // Initialize domain bounding box
        m_domain = domain;

        // Compute total number of nodes along each dimension:
        // For each element, count all nodes except those it shares with its "positive neighbors"
        // Then add in the "positive border" nodes
        m_NbNodesPerDimension = m_NbElementsPerDimension.array() * (m_NbNodesPerDimensionPerElement.array() - 1) + 1;

        // Set NDVector of Nodes
        m_nodes.Resize(m_NbNodesPerDimension);
        m_nodes.getFlatIndexIncrements(m_NodeGlobalIndexIncrement);

        // Compute nodes coordinates and initialize m_nodes
        VectorND<N> boxDimensions = m_domain.dimensions();
        VectorND<N> nodeSpacings = boxDimensions.array() / (m_NbNodesPerDimension.template cast<Real>().array() - 1.0); // distance (in the coordinate directions) of adjacent nodes
        for (size_t ni = 0; ni < m_nodes.Size(); ni++) {
            auto gridIndices = eigenNDIndexWrapper(m_nodes.unflattenIndex(ni)).eval();   // type of nodes storage (row or column major) is handled by the flattening function
            m_nodes[ni].setCoordinates(m_domain.minCorner.array() + gridIndices.template cast<Real>().array() * nodeSpacings.array());
        }

        // Set NDVector of elements and densities
        m_densities.Resize(m_NbElementsPerDimension);
        m_numElements = m_densities.Size();

        // Initialize the element aspect ratio (uniform grid) using the distance of adjacent nodes
        // (stretchings are used for mapping to reference element when intergrating)
        setStretchings(boxDimensions.array() / m_NbElementsPerDimension.template cast<Real>().array());

        // Generate the array of global node indices for the 0th element's nodes;
        // the nth element's nodes are just this array offset by the first node's index.
        {
            EigenNDIndex local_nNd = EigenNDIndex::Zero();

            // Update the globalNodeIndex associated with each element node as we enumerate them.
            // (To avoid the overhead of calling m_nodes.flatIndex(...).
            // Start at the lowest-indexed node ("bottom left" corner)
            size_t globalNodeIndex = 0;
            size_t back = 0;

            while (true) {
                m_referenceElementNodes[back++] = globalNodeIndex;

                // Increment N-digit counter
                // WARNING: this assumes the current local node ordering; it must
                // match ElementNodeIndexer flattening conventioning!
                ++local_nNd[N - 1];
                ++globalNodeIndex;
                for (int d = N - 1; local_nNd[d] == m_NbNodesPerDimensionPerElement[d]; --d) {
                    if (d == 0) return; // "most significant" digit has overflowed; we are done
                    globalNodeIndex += m_NodeGlobalIndexIncrement[d - 1] - m_NbNodesPerDimensionPerElement[d] * m_NodeGlobalIndexIncrement[d];
                    local_nNd[d] = 0;
                    ++local_nNd[d - 1];
                }
            }
        }
    }

    // If no domain bounding box is specified, use unit squares/cubes
    TensorProductSimulator(const std::vector<size_t> & numElemsPerDim)
        : TensorProductSimulator(BBox<VectorND<N>>(VectorND<N>::Zero().eval(),
                                                   eigenNDIndexWrapper(numElemsPerDim).template cast<Real>().matrix().eval()),
                                 numElemsPerDim) { }

    BBox<VectorND<N>> domain() { return m_domain; }

    void readMaterial (const std::string &materialFilepath) {
        // Set the material tensor
        Materials::Constant<N> mat(materialFilepath);
        setETensor(mat.getTensor());
    }

    ETensor getETensor() const {
        return m_E_tensor;
    }

    void setETensor(const ETensor &et) {
        m_E_tensor = et;
        m_updateK0();
    }

    void applyPeriodicConditions(Real /* epsilon */ = 1e-7,
                                 bool /*ignoreMismatch*/ = false,
                                 std::unique_ptr<PeriodicCondition<N>> /*pc*/ = nullptr) {
        periodic_BC = true;
        m_solver.reset();
        m_hessianSparsityPattern.clear();
    }

    void applyDisplacementsAndLoadsFromFile(const std::string &path) {
        bool noRigidMotion;
        auto conds = readBoundaryConditions<N>(path, m_domain, noRigidMotion);
        applyDisplacementsAndLoads(conds);
    }

    // Translate constraint conditions read from file into conditions on every single node.
    // Constraints can be Dirichlet-like (displacements) or Neumann-like (forces) but can be applied also to internal nodes.
    // Note: forces are expressed in Newtons, they are not defined per unit area as usual Neumann conditions.
    void applyDisplacementsAndLoads(const std::vector<CondPtr<N>> &conds) {
        // Set up evaluator environment
        ExpressionEnvironment env;
        env.setVectorValue("mesh_size_", m_domain.dimensions());
        env.setVectorValue("mesh_min_", m_domain.minCorner);
        env.setVectorValue("mesh_max_", m_domain.maxCorner);

        size_t dirichletRegionIdx = 0;
        for (const auto &cond : conds) {
            env.setVectorValue("region_size_", cond->region->dimensions());
            env.setVectorValue("region_min_",  cond->region->minCorner);
            env.setVectorValue("region_max_",  cond->region->maxCorner);
            std::runtime_error illegalCondition("Illegal constraint type, only \"dirichlet\" and \"force\" accepted");
            std::runtime_error unimplementedForceType("Illegal force type, only \"force\" accepted");

            if (auto nc = dynamic_cast<const NeumannCondition<N> *>(cond.get())) {  // Force constraint
                // Identify the nodes involved in the constraint
                std::vector<size_t> nodesInRegion;
                for (size_t ni = 0; ni < m_nodes.Size(); ni++)
                    if (nc->containsPoint(m_nodes[ni].coordinates()))
                        nodesInRegion.push_back(ni);
                if (nodesInRegion.size() == 0)
                    throw std::runtime_error("Force constraint region unmatched");

                // Store force conditions on the nodes
                for (size_t ni : nodesInRegion) {
                    env.setXYZ(m_nodes[ni].coordinates());
                    if (nc->type == NeumannType::Force) {  // Force is distributed uniformly among all nodes in the region
                        m_nodes[ni].hasForce(true);
                        m_nodes[ni].setForce(nc->traction(env)/nodesInRegion.size());
                    }
                    else throw unimplementedForceType;
                }
            }
            else if (auto dc = dynamic_cast<const DirichletCondition<N> *>(cond.get())) {  // Displacement constraint
                ++dirichletRegionIdx;
                size_t numNodesInRegion = 0;
                for (auto & node : m_nodes) {
                    if (dc->containsPoint(node.coordinates())) {
                        ++numNodesInRegion;
                        env.setXYZ(node.coordinates());
                        node.setDirichlet(dc->componentMask, dc->displacement(env));
                        node.setDirichletRegion(dirichletRegionIdx);
                    }
                }
                if (numNodesInRegion == 0)
                    throw std::runtime_error("Dirichlet region unmatched");
                m_solver.reset(); // Changing dirichlet conditions invalidates the symbolic factorization
            }
            else throw illegalCondition;
        }
    }

    // Assemble a vector of size numNodes() encoding the condition on displacement imposed for each direction of the node
    using ArrayXNb = Eigen::Array<bool, Eigen::Dynamic, N>;
    ArrayXNb getDirichletMask() const {
        const size_t nn = numNodes();
        ArrayXNb mask(nn, size_t(N));
        for (size_t ni = 0; ni < nn; ni++)
            mask.row(ni) = m_nodes[ni].dirichletComponents().template getArray<N>();
        return mask;
    }

    // Assemble a vector of size numNodes() encoding the condition on displacement imposed for each direction of the node
    void setDirichletMask(Eigen::Ref<const ArrayXNb> mask) {
        const size_t nn = numNodes();
        if ((mask.rows() != nn) || (mask.cols() != N)) throw std::runtime_error("Size mismatch");
        for (size_t ni = 0; ni < nn; ni++)
            m_nodes[ni].dirichletComponents().setArray(mask.row(ni));
    }

    VField getDirichletValues() const {
        const size_t nn = numNodes();
        Eigen::Matrix<Real, Eigen::Dynamic, N> result(nn, size_t(N));
        for (size_t ni = 0; ni < nn; ni++)
            result.row(ni) = m_nodes[ni].dirichletDisplacement();
        return result;
    }

    void setDirichletValues(Eigen::Ref<const VField> values) {
        const size_t nn = numNodes();
        if ((values.rows() != nn) || (values.cols() != N)) throw std::runtime_error("Size mismatch");
        for (size_t ni = 0; ni < nn; ni++)
            m_nodes[ni].dirichletDisplacement() = values.row(ni);
    }

    // Assemble a vector of size numNodes() encoding the force imposed for each direction of the node
    Eigen::Matrix<std::array<bool, N>, Eigen::Dynamic, 1> getForceMask() const {
        Eigen::Matrix<std::array<bool, N>, Eigen::Dynamic, 1> indicators(m_nodes.Size());
        for (size_t ni = 0; ni < m_nodes.Size(); ni++) {
            indicators[ni].fill(false);
            if (m_nodes[ni].force()[0] != 0)               indicators[ni][0] = true; // X force
            if (m_nodes[ni].force()[1] != 0)               indicators[ni][1] = true; // Y force
            if ((N == 3) && (m_nodes[ni].force()[2] != 0)) indicators[ni][2] = true; // Z force
        }
        return indicators;
    }

    void setUniformDensities(Real density) {
        if ((density > 1.0) || (density < 0))
            throw std::runtime_error("Density value (" + std::to_string(density) + ") has to be in between 0 and 1");
        m_densities.fill(density);
        m_numericFactorizationUpToDate = false;
    }

    void setDensity(size_t flatIndex, Real density) {
        if ((density > 1.0) || (density < 0))
            throw std::runtime_error("Density value (" + std::to_string(density) + ") has to be in between 0 and 1");
        m_densities[flatIndex] = density;
        m_numericFactorizationUpToDate = false;
    }

    void readDensities (const std::string &materialPath, const std::string &FieldName = "density") {
        if (fileExtension(materialPath) != ".msh")
            throw std::runtime_error("Material file extension" + fileExtension(materialPath) + " is not supported");

        MSHFieldParser<N> fieldParser(materialPath);

        ScalarField<Real> density = fieldParser.scalarField(FieldName,DomainType::PER_ELEMENT);

        const auto &elems = fieldParser.elements();
        const auto &verts = fieldParser.vertices();

        if (elems.size() != numElements())
            throw std::runtime_error("The number of elements in the mesh : " + std::to_string(elems.size())
                    + " and the number of elements of the simulator : " + std::to_string(numElements())
                    + " must be equal.");

        BBox<Point3D> bbox(verts);

        for (size_t ei = 0; ei < elems.size(); ++ei) {
            m_densities[elementIndexFromMeshIO(elems[ei], verts, bbox)] = density[ei];
        }
        m_numericFactorizationUpToDate = false;
    }

    // Get the grid index of a MeshIO element.
    size_t elementIndexFromMeshIO(const MeshIO::IOElement &e, const std::vector<MeshIO::IOVertex> &vertices, const BBox<Point3D> &bbox) const {
        Point3D center(Point3D::Zero());
        for (size_t v : e)
            center += vertices.at(v).point;
        center *= 1.0 / e.size();

        // Transform center to the simulator coordinate system from "bbox".
        // Get the center of the elements in the simulator coordinate system
        center = bbox.interpolationCoordinates(center);
        PointND<N> centerND;
        for (size_t i = 0; i < N; ++i)
            centerND[i] = center[i] * m_element_stretch[i] * m_NbElementsPerDimension[i];
        // return the corresponding index
        return getElementIndex(centerND);
    }

    const Node<N> &getNode(const size_t &n) const { return m_nodes[n]; }
          Node<N> &getNode(const size_t &n)       { return m_nodes[n]; }

    void imposeDisplacement(const size_t &n, const VectorND<N> displacement) {
        ComponentMask mask;
        for (size_t d = 0; d < N; ++d) {
            mask.set(d);
        }
        m_nodes[n].setDirichlet(mask, displacement);
    }

    // Zero out the Dirichlet constraints
    void zeroOutDirichlet() {
        for (size_t n = 0; n < m_nodes.Size(); ++n) {
            if (m_nodes[n].hasDirichlet())
                m_nodes[n].setDirichlet(VectorND<N>::Zero());
        }
    }

    // Fills vertices and elements with the vertices and elements of the simulator
    void getMesh(std::vector<MeshIO::IOVertex> &vertices,
                 std::vector<MeshIO::IOElement> &elements) const {
        if (!(N==2 || N == 3))
            throw std::runtime_error("Field writer not supported for dimension " + std::to_string(N));

        vertices.clear();
        elements.clear();

        for (size_t i = 0; i < numNodes(); ++i)
            vertices.push_back(nodePosition(i));

        size_t numVertPerElem = (N==2) ? 4 : 8;

        for (size_t ei = 0; ei < m_numElements; ++ei) {
            MeshIO::IOElement Ei;

            for (size_t pair = 0; pair < numVertPerElem/2; ++pair) {
                size_t v1 = elemVertexGlobalIndex(ei, pair*2);
                size_t v2 = elemVertexGlobalIndex(ei, pair*2+1);

                // Gmsh ordering : trigonometric on lines
                //                 counterclockwise (trigonometric) on quads
                //                 counterclockwise on both quad faces in 3D
                // see: http://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
                if (pair % 2 == 0) {
                    Ei.push_back(v1);
                    Ei.push_back(v2);
                } else {
                    Ei.push_back(v2);
                    Ei.push_back(v1);
                }
            }
            elements.push_back(Ei);
        }
    }

    Eigen::VectorXd getDensities() {
        const auto &v = m_densities.data();
        return Eigen::Map<const Eigen::VectorXd>(v.data(), v.size());
    }

    // Writes the complete K Triplet matrix to outFile in binary format
    void writeFEMMatrix( const std::string &outFile, TMatrix &K) const {
        K.dumpBinary(outFile);
    }

    // Set the stretching in the simulator (m_element_stretch), as well as in all elements (stretching is identical for all elements)
    void setStretchings(const VectorND<N> &s) {
        m_element_stretch = s;
        m_element.setStretchings(s);
        m_updateK0();
    }

    // Assemble the global stiffnessMatrix into K, which should already be
    // initialized with the correct size. If K is a SuiteSparseMatrix, it
    // should also be initized with the correct sparsity pattern.
    // If `sparsity = true` we ensure nonzero values are written to every entry
    // that could possibly become nonzero.
    template<typename SparseMatrix_>
    void m_assembleStiffnessMatrix(SparseMatrix_ &K, bool sparsity = false) const {
        BENCHMARK_START_TIMER("Assembling stiffness matrix");
        const size_t n = N * numDoFs();

        auto accumToSparseMatrix = [&](size_t ei, const PerElementStiffness &Ke, SparseMatrix_ &_K) {
            constexpr size_t nNodes = Element::nNodes;
            for (size_t j = 0; j < nNodes; ++j) {
                int dj = DoF(elemNodeGlobalIndex(ei, j));
                for (size_t i = 0; i < nNodes; ++i) {
                    int di = DoF(elemNodeGlobalIndex(ei, i));
                    if (di > dj) continue; // accumulate only upper triangle.
                    for (size_t cj = 0; cj < N; ++cj) {
                        _K.addNZStrip(N * di, N * dj + cj,
                                Ke.col(N * j + cj).segment(N * i, std::min((N * dj + cj) - (N * di) + 1, size_t(N))));
                    }
                }
            }
        };

        if (hasCachedElementStiffness() && !sparsity) {
            for (size_t ei = 0; ei < m_numElements; ++ei)
                accumToSparseMatrix(ei, cachedElementStiffnessMatrix(ei), K);
        }

        else {
            // Assembly (optimized exploiting regularity of the mesh)
            PerElementStiffness Ke, k0 = fullDensityElementStiffnessMatrix();
            for (size_t ei = 0; ei < m_numElements; ++ei) {
                Ke = elementYoungModulus(ei) * k0; // Each element contributes differently to the global stiffness matrix depending only on its Young modulus
                if (sparsity) Ke.setOnes();
                accumToSparseMatrix(ei, Ke, K);
            }
        }

        BENCHMARK_STOP_TIMER("Assembling stiffness matrix");
    }

    void m_cacheSparsityPattern() const {
        TMatrix Ktrip(N * numDoFs(), N * numDoFs());
        Ktrip.symmetry_mode = TMatrix::SymmetryMode::UPPER_TRIANGLE;
        Ktrip.reserve(KeSize * KeSize * numElements()); // upper bound; difficult to predict exactly in periodic case.
        m_assembleStiffnessMatrix(Ktrip, true);
        Ktrip.sumRepeated();
        m_hessianSparsityPattern = SuiteSparseMatrix(Ktrip);
        m_hessianSparsityPattern.fill(1.0);
    }

    // Check if *any* block stiffness matrix has been cached.
    // WARNING: the user must manually ensure the cached matrix is kept up-to-date when densities are
    // updated (by calling `updateBlockK`).
    bool hasBlockK() const { return m_blockK.m != 0; }

    // Clear out the cached block stiffness matrix.
    void clearBlockK() { m_blockK.m = m_blockK.n = 0; }

    // Update the block-sparse-matrix representation of stiffness matrix K.
    // Note that this representation does *not* leverage symmetry and explicitly
    // stores the upper and lower triangle. This is to simplify and accelerate
    // the GS smoothing iterations and matvecs.
    void updateBlockK() {
        if (periodic_BC) throw std::runtime_error("Periodic case unimplemented");

        const size_t nn = numNodes();
        auto &Ax = m_blockK.Ax;
        // Generate the "scalar-valued" sparsity pattern if it has not yet been created.
        if (m_blockK.m == 0) {
            m_blockK.m = nn;
            m_blockK.n = nn;

            auto &Ai = m_blockK.Ai;
            auto &Ap = m_blockK.Ap;

            Ap.reserve(nn + 1);
            Ap.assign(1, 0); // first column start
            Ai.clear();
            Ai.reserve(std::pow(2, N) * numNodesPerElem * nn); // overestimate assuming no shared nodes.

            std::vector<size_t> adj;
            for (size_t ni = 0; ni < nn; ++ni) {
                adj.clear();
                EigenNDIndex n = m_nodes.template unflattenIndex<N>(ni);
                visitIncidentElements(n, [&adj](const size_t /* ei */, const size_t /* local_n */, const ENodesArray &enodes) {
                        for (int i = 0; i < enodes.size(); ++i)
                            adj.push_back(enodes[i]);
                    });
                std::sort(adj.begin(), adj.end());
                adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
                Ai.reserve(Ai.size() + adj.size());
                for (size_t nj : adj) Ai.push_back(nj);
                Ap.push_back(Ai.size()); // column end
            }
            Ai.shrink_to_fit();
            Ax.resize(Ai.size());
        }
        // Zero out the Hessian.
        m_blockK.setZero();

        // Basic sanity size checks for the existing sparsity pattern.
        assert((m_blockK.m == nn) && (m_blockK.n == nn) && (m_blockK.Ap.size() == nn + 1) && (m_blockK.Ap.back() == m_blockK.Ai.size()));

        constexpr bool LOWER_TRI = false; // Whether to assemble only the lower triangle and then reflect; this seems slightly slower...
        auto accumToSparseMatrix = [&](size_t ei, const PerElementStiffness &Ke, BlockSuiteSparseMatrix &K) {
            const size_t enode_offset = flattenedFirstNodeOfElement1D(ei);
            for (size_t j = 0; j < Element::nNodes; ++j) {
                int gj = m_referenceElementNodes[j] + enode_offset;
                SuiteSparse_long hint = std::numeric_limits<SuiteSparse_long>::max();
                for (size_t i = LOWER_TRI ? (j + 1) : 0; i < Element::nNodes; ++i) {
                    int gi = m_referenceElementNodes[i] + enode_offset;
                    hint = K.addNZ(gi, gj, Ke.template block<N, N>(N * i, N * j), hint);
                }
            }
        };

        // Note: due to the small amount of work for each element,
        // parallel assembly is actually slower than a plain for loop :(
        // Assemble the block sparse matrix from the per-element stiffness matrices.
        if (hasCachedElementStiffness()) {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("Assemble Cached Ke");
            for (size_t ei = 0; ei < m_numElements; ++ei)
                accumToSparseMatrix(ei, cachedElementStiffnessMatrix(ei), m_blockK);
        }
        else {
            BENCHMARK_SCOPED_TIMER_SECTION timer2("Assemble Ke");
            // Assembly (exploits the identicalness of all elements)
            for (size_t ei = 0; ei < m_numElements; ++ei)
                accumToSparseMatrix(ei, (elementYoungModulus(ei) * m_K0).eval(), m_blockK);
        }

        if (LOWER_TRI)
            m_blockK.reflectLowerTriangleInPlace();
    }

    const BlockSuiteSparseMatrix &blockK() const { return m_blockK; }

    // SIMP formula to comupte the Young modulus of an element given its density
    Real elementYoungModulus(size_t ei) const {
        return m_E_min + std::pow(m_densities[ei], m_gamma) * (m_E_0 - m_E_min);
    }

    // Evaluate the gradient of compliance given the equilibrium displacement `u`.
    NDVector<Real> complianceGradient(const VField &u) const {
        // The TPS routine returns an upper triangular matrix: copy upper triangular into the lower triangular part
        PerElementStiffness k0 = fullDensityElementStiffnessMatrix();

        const size_t nodesPerElement = NbNodesPerElement(); // coincides with KeSize / N;
        const size_t ne = numElements();

        NDVector<Real> g(NbElementsPerDimension());

        parallel_for_range(numElements(), [&](size_t i) {
            Eigen::Matrix<Real, KeSize, 1> u_i;
            // Fill the flattened displacements vector of element i
            for (size_t nLocal = 0; nLocal < nodesPerElement; ++nLocal) {
                size_t nGlobal = elemNodeGlobalIndex(i, nLocal);
                u_i.template segment<N>(nLocal * N) = u.row(nGlobal);
            }
            // Compute derivative of compliance w.r.t. projected density field
            g[i] = -0.5 * m_gamma * std::pow(m_densities[i], m_gamma - 1.0) * (m_E_0 - m_E_min) * u_i.dot(k0 * u_i);
        });

        return g;
    }

    // Compute the element stiffness matrix assuming that all the elements
    // are assigned the same fabrication material and the density is 1.0.
    const PerElementStiffness &fullDensityElementStiffnessMatrix() const { return m_K0; }

    PerElementStiffness elementStiffnessMatrix(size_t ei) const {
        if (hasCachedElementStiffness()) return m_KeCache[ei];
        return elementYoungModulus(ei) * fullDensityElementStiffnessMatrix();
    }

    const PerElementStiffness &cachedElementStiffnessMatrix(const size_t e) const {
        return m_KeCache.at(e);
    }

    bool hasCachedElementStiffness() const {
        return m_KeCache.size() == numElements();
    }

    // Override the default element stiffness matrices (computed by scaling the
    // fullDensityElementStiffnessMatrix) with the stiffness matrices in `Kes`.
    // This is helpful, e.g., for the coarsened simulators used in Multigrid.
    void cacheCustomElementStiffnessMatrices(aligned_std_vector<PerElementStiffness> &&Kes) {
        m_KeCache = std::move(Kes);
        m_numericFactorizationUpToDate = false;
    }

    void cacheElementStiffnessMatrices() {
        aligned_std_vector<PerElementStiffness> Ke;
        const size_t ne = numElements();
        Ke.reserve(ne);
        for (size_t i = 0; i < ne; ++i)
            Ke.push_back(elementStiffnessMatrix(i));
        cacheCustomElementStiffnessMatrices(Ke);
    }

    void clearCachedElementStiffness() { m_KeCache.clear(); m_numericFactorizationUpToDate = false; }
    
    // Compute global load under unit strain cstrain
    // F: global load
    // cstrain: constant unit strain
    VField constantStrainLoad(const typename Element::SMatrix &cstrain) const {
        typedef typename Element::ElementLoad ElementLoad;

        auto accumToGlobalVField = [&](size_t ei, const ElementLoad &l, VField &_F) {
            constexpr size_t nNodes = Element::nNodes;

            for (size_t j = 0; j < nNodes; ++j) {
                size_t dj = DoF(elemNodeGlobalIndex(ei, j));
                _F.row(dj) += l.col(j).transpose();
            }

        };

        VField F(numDoFs(), int(N));
        F.setZero();

        // Build all element loads in parallel, then collect
        std::vector<ElementLoad> elemLoads(m_numElements);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m_numElements),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t ei = r.begin(); ei < r.end(); ++ei)
                    m_element.constantStrainLoad(elemLoads[ei], m_E_tensor, m_densities[ei], cstrain);
            }
        );

        for (size_t i = 0; i < m_numElements; ++i)
            accumToGlobalVField(i, elemLoads[i], F);

        return F;
    }

    void elementStrain(size_t ei, const VField &u, Strain &e) const {
        auto nodeIndexGetter = [=](size_t n) { return elemNodeGlobalIndex(ei, n); };
        m_element.strain(nodeIndexGetter, u, e);
    }

    void elementAverageStrain(size_t ei, const VField &u, SMatrix &e) const {
        auto nodeIndexGetter = [=](size_t n) { return elemNodeGlobalIndex(ei, n); };
        m_element.averageStrain(nodeIndexGetter, u, e);
    }

    // Solve for equilibrium under DoF load f
    VField solve(const VField &f) const {
        std::vector<size_t> fixedVars;
        std::vector<Real>   fixedVarValues;
        getDirichletVarsAndValues(fixedVars, fixedVarValues);
        std::vector<bool> isFixed(N * numDoFs());
        for (size_t i : fixedVars) isFixed.at(i) = true;
        for (Real v : fixedVarValues) { if (v != 0) throw std::runtime_error("Nonzero Dirichlet constraints currently unsupported"); }

        Eigen::VectorXd rhs = Eigen::Map<const Eigen::VectorXd>(f.data(), f.size());
        removeFixedEntriesInPlace(rhs, isFixed);

        if (!m_numericFactorizationUpToDate || !m_solver) {
            SuiteSparseMatrix K = getK();
            m_numericFactorizationUpToDate = true;

            K.rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });

            if (!m_solver) {
                auto Hsp = m_hessianSparsityPattern;
                Hsp.rowColRemoval([&](SuiteSparse_long i) { return isFixed[i]; });
                m_solver = std::make_unique<CholmodFactorizer>(Hsp);
                m_solver->factorizeSymbolic();
            }
            m_solver->updateFactorization(K);
        }

        BENCHMARK_START_TIMER_SECTION("Elasticity Cholesky Solve");
        Eigen::VectorXd x = m_solver->solve(rhs);

        BENCHMARK_STOP_TIMER_SECTION("Elasticity Cholesky Solve");
        return dofToNodeField(extractFullSolution(x, isFixed));
    }

    // Construct a vector of reduced components by removing the entries of "x" corresponding
    // to fixed variables. This is a (partial) inverse of extractFullSolution.
    void removeFixedEntriesInPlace(Eigen::VectorXd &x, const std::vector<bool> &isFixed) const {
        int back = 0;
        for (int i = 0; i < x.size(); ++i)
            if (!isFixed[i]) x[back++] = x[i];
        x.conservativeResize(back);
    }

    // Extract the full linear system solution vector "x" from the reduced linear
    // system solution "xReduced" (which was solved by removing the rows/columns for fixed variables).
    Eigen::VectorXd extractFullSolution(const Eigen::VectorXd &xReduced, const std::vector<bool> &isFixed) const {
        Eigen::VectorXd x(N * numDoFs());
        int back = 0;
        for (int i = 0; i < x.size(); ++i) {
            if (!isFixed[i]) x[i] = xReduced[back++];
            else             x[i] = 0.0;
        }
        assert(back == xReduced.size());
        return x;
    }

    // Solve for equilibrium under loads and displacements imposed from .bc file
    VField solveWithImposedLoads() const { return solve(buildLoadVector()); }

    // Compute load on the DoFs from Neumann-like constraints
    VField buildLoadVector() const {
        VField load(numDoFs(), int(N));
        load.setZero();
        for (size_t ni = 0; ni < m_nodes.Size(); ni++) {
            if (m_nodes[ni].hasForce())
                load.row(DoF(ni)) += m_nodes[ni].force();
        }
        return load;
    }

    // Apply the stiffness matrix `K` to a displacement field `u`, computing
    // the elastic restoring forces.
    void applyK(Eigen::Ref<const VField> u, VField &result) const {
        if (periodic_BC) throw std::runtime_error("Periodic case unimplemented");
        BENCHMARK_SCOPED_TIMER_SECTION timer("applyK " + description());
        using LocalDisplacements = Eigen::Matrix<Real, KeSize, 1>;

        result.setZero(numNodes(), N);

        const auto &enodes = m_referenceElementNodes;

        if (hasCachedElementStiffness()) {
            auto accumulatePerElementContrib = [&u, &enodes, this](size_t ei, VField &Ku_out) {
                const size_t enode_offset = flattenedFirstNodeOfElement1D(ei);
                const auto &Ke = m_KeCache[ei];
                LocalDisplacements Ke_u_local(Ke.template block<KeSize, N>(0, 0) * u.row(enodes[0] + enode_offset).transpose());
                // Loop over nodal displacements
                for (size_t m = 1; m < enodes.size(); ++m)
                    Ke_u_local += Ke.template block<KeSize, N>(0, N * m) * u.row(enodes[m] + enode_offset).transpose();
                // Loop over nodal matvec contributions
                for (size_t m = 0; m < enodes.size(); ++m)
                    Ku_out.row(enodes[m] + enode_offset) += Ke_u_local.template segment<N>(N * m).transpose();
            };
#if 1
            if (m_applyKThreadWorkspace)   assemble_parallel(accumulatePerElementContrib, result, numElements(), *m_applyKThreadWorkspace);
            else m_applyKThreadWorkspace = assemble_parallel(accumulatePerElementContrib, result, numElements());
#else
            assemble_parallel(accumulatePerElementContrib, result, numElements());
#endif
        }
        else {
            auto accumulatePerElementContrib = [&u, &enodes, this](size_t ei, VField &Ku_out) {
                const size_t enode_offset = flattenedFirstNodeOfElement1D(ei);
                LocalDisplacements Ke_u_local(m_K0.template block<KeSize, N>(0, 0) * u.row(enodes[0] + enode_offset).transpose());
                // Loop over nodal displacements
                for (size_t m = 1; m < enodes.size(); ++m)
                    Ke_u_local += m_K0.template block<KeSize, N>(0, N * m) * u.row(enodes[m] + enode_offset).transpose();
                Ke_u_local *= elementYoungModulus(ei);
                // Loop over nodal matvec contributions
                for (size_t m = 0; m < enodes.size(); ++m)
                    Ku_out.row(enodes[m] + enode_offset) += Ke_u_local.template segment<N>(N * m).transpose();
            };
#if 1
            if (m_applyKThreadWorkspace)   assemble_parallel(accumulatePerElementContrib, result, numElements(), *m_applyKThreadWorkspace);
            else m_applyKThreadWorkspace = assemble_parallel(accumulatePerElementContrib, result, numElements());
#else
            assemble_parallel(accumulatePerElementContrib, result, numElements());
#endif
        }
    }

    VField applyK(Eigen::Ref<const VField> u) const {
        VField result(VField::Zero(numNodes(), N));
        applyK(u, result);
        return result;
    }

    void applyBlockK(Eigen::Ref<const VField> u, VField &result) const {
        if (!hasBlockK()) throw std::logic_error("Attempting to apply a nonexisting block matrix");
        BENCHMARK_SCOPED_TIMER_SECTION timer("applyBlockK " + description());
        m_blockK.applyTransposeParallel(u, result);
    }

    SuiteSparseMatrix getK() const {
        if (m_hessianSparsityPattern.nnz() == 0)
            m_cacheSparsityPattern();
        SuiteSparseMatrix K = m_hessianSparsityPattern;
        K.setZero();
        m_assembleStiffnessMatrix(K);
        return K;
    }

    // Returns a VField of the size of the mesh, given the values of the VField at the dof nodes
    template<class Vec>
    VField dofToNodeField(const Vec &dofValues) const {
        if (dofValues.size() != N * numDoFs()) {
            throw std::runtime_error("dofToNodeField : invalid number of values in input : "
                                     + std::to_string(dofValues.size()) + " -- should be "
                                     + std::to_string(numDoFs()));
        }

        const size_t nn = m_nodes.Size();
        VField F(nn, int(N));

        for (size_t i = 0; i < nn; ++i)
            F.row(i) = Eigen::Map<const VectorND<N>>(dofValues.data() + N * DoF(i), int(N));
        return F;
    }

    EigenNDIndex firstNodeOfElement(EigenNDIndex eiND) const {
        return eigenNDIndexWrapper(eiND) * (m_NbNodesPerDimensionPerElement - 1);
    }
    EigenNDIndex firstNodeOfElement(size_t ei) const {
        return firstNodeOfElement(m_densities.template unflattenIndex<N>(ei));
    }
    size_t flattenedFirstNodeOfElement1D(size_t ei) const {
        size_t result = 0;
        for (int d = N - 1; d >= 0; --d) { // right-to-left (least significant index to most significant)
            // Get d^th entry of unflattened index of `ei`
            size_t ei_d = ei % m_NbElementsPerDimension[d];
            ei /= m_NbElementsPerDimension[d];

            // Convert to d^th entry of unflattened index of element's first node
            size_t ni_d = ei_d * (m_NbNodesPerDimensionPerElement[d] - 1);

            // Accumulate to flattened index of first element node.
            result += ni_d * m_NodeGlobalIndexIncrement[d];
        }
        return result;
    }

    // Get the 1D index of the local node `nd_local_idx` of element `ei`.
    template<class NdLocalIndex, typename enable = decltype(std::declval<NdLocalIndex>()[0])>
    size_t elemNodeGlobalIndex(size_t ei, const NdLocalIndex &nd_local_idx) const {
        // 1D index ei => ND element index => minimum corner node index,
        // sum => ND node index => 1D index

        // Global node index
        EigenNDIndex global_nNd = firstNodeOfElement(ei) + eigenNDIndexWrapper(nd_local_idx);

        // return flat global linear index of the node
        return m_nodes.flatIndex(global_nNd);
    }

    // Returns the linear index (in the m_nodes vector) of the node "n" in element "ei"
    // where "n" is the linear linear index of the nodes belonging to the element "ei"
    // and "ei" is the linear index of the element in the vector m_densities
    size_t elemNodeGlobalIndex(size_t ei, size_t n) const {
        // 1D local node index n => ND node index offset from element min corner,
        return elemNodeGlobalIndex(ei, ElementNodeIndexer::unflattenIndex(n));
    }

    // Similar as elemNode, except "v" is the flat index of a vertex in the element "ei"
    // Vertices are the nodes at the corner of the elements, so there are 2 vertices per dimension
    size_t elemVertexGlobalIndex(size_t ei, size_t v) const {
        // Global ND element index
        EigenNDIndex ElementIndices = m_densities.template unflattenIndex<N>(ei);  // equal to index of vertex n = 0
        // Global ND node index
        EigenNDIndex LocalVertexIndex;
        NDVector<Node<N>>::unflattenIndex(v, EigenNDIndex::Constant(2), LocalVertexIndex);

        EigenNDIndex GlobalVertexIndex;
        // Go from vertex to node index.
        GlobalVertexIndex = (ElementIndices + LocalVertexIndex) * (m_NbNodesPerDimensionPerElement-1);

        // return flat global linear index of the node
        return m_nodes.flatIndex(GlobalVertexIndex);
    }

    template<typename Visitor>
    void visitLocalNodeNdIndices(Visitor &&visit) const {
        EigenNDIndex local_nNd = EigenNDIndex::Zero();
        while (true) {
            visit(local_nNd);

            // Increment N-digit counter
            ++local_nNd[N - 1];
            for (int d = N - 1; local_nNd[d] == m_NbNodesPerDimensionPerElement[d]; --d) {
                if (d == 0) return; // "most significant" digit has overflowed; we are done
                local_nNd[d] = 0;
                ++local_nNd[d - 1];
            }
        }
    }

    template<typename Visitor>
    void visitElementNodes(size_t ei, Visitor &&visit) const {
        size_t offset = flattenedFirstNodeOfElement1D(ei);
        for (size_t i = 0; i < numNodesPerElem; ++i)
            visit(m_referenceElementNodes[i] + offset);
    }

    template<typename Derived, typename Visitor>
    void visitElementNodes(const Eigen::ArrayBase<Derived> &ei, Visitor &&visit) const {
        visitElementNodes(m_densities.flatIndex(ei), visit);
    }

    using ENodesArray = Eigen::Array<size_t, numNodesPerElem, 1>;
    ENodesArray elementNodes(size_t ei) const {
        return m_referenceElementNodes + flattenedFirstNodeOfElement1D(ei);
    }

    size_t elementIndexForGridCell(const EigenNDIndex &cellIdxs) const {
        return m_densities.flatIndex(cellIdxs);
    }

    EigenNDIndex ndIndexForElement(size_t ei) const { return m_densities.template unflattenIndex<N>(ei); }
    EigenNDIndex ndIndexForNode   (size_t ni) const { return m_nodes    .template unflattenIndex<N>(ni); }

    // Get periodic variable corresponding to node ni
    // (nodes on the "max face" of the period cell
    // get the index of the corresponding node on the
    // "min face")
    // (Use a linear indexing of the grid of nodes with the
    // maximum "layers" removed)
    size_t DoF(size_t ni) const {
        if (!periodic_BC) // without periodic conditions, DoF always coincide with node index in m_nodes
            return ni;

        // 1D node index => ND node index => ND dof node index => 1D dof node index on smaller grid

        // 1D node index => ND node index
        EigenNDIndex NDnodeIndices = m_nodes.template unflattenIndex<N>(ni);

        // ND node index => ND dof node index
        for (size_t dim = 0; dim < N; ++dim) {
            if (NDnodeIndices[dim] == m_NbNodesPerDimension[dim]-1)
                NDnodeIndices[dim] = 0;
        }

        // ND dof node index => 1D dof node index on smaller grid
        EigenNDIndex NbNodesPerDimentionsOnSmallerGrid = m_NbNodesPerDimension;
        for (size_t dim = 0; dim < NbNodesPerDimentionsOnSmallerGrid.size(); dim++)
            --NbNodesPerDimentionsOnSmallerGrid[dim];
        return NDVector<Node<N>>::flatIndex(NDnodeIndices, NbNodesPerDimentionsOnSmallerGrid);
    }

    // Number of Degree of freedom is 1 less than number of nodes in each dimension because of periodic boundary condition
    size_t numDoFs() const {
        if (!periodic_BC) // without periodic conditions, DoF always coincide with node index in m_nodes
            return numNodes();

        size_t numDofs = 1;

        for (size_t dim = 0; dim < N; ++dim) {
            numDofs *= m_NbNodesPerDimension[dim]-1;
        }

        return numDofs;
    }

    size_t numNodes() const { return m_nodes.Size(); }
    const NDVector<Node<N>> &nodes() const { return m_nodes; }

    size_t numElements() const { return m_numElements; }

    Real volume() const {
        Real volume = 1;
        for (size_t dim = 0; dim < N; ++dim)
            volume *= m_NbElementsPerDimension[dim] * m_element_stretch[dim];
        return volume;
    }

    const ETensor &elementElasticityTensor(size_t ei) const { return m_E_tensor; }
    Real elementDensity(size_t ei) const { return m_densities[ei]; }
    Real elementVolume(size_t ei) const { return m_element.Volume(); }

    // Setting an element density changes the FEM system, which has to be cleared.
    void setElementDensity(size_t ei, Real value) {
        m_densities[ei] = value;
        m_numericFactorizationUpToDate = false;
    }

    const NDVector<Real> &elementDensities() const { return m_densities; }
    void setElementDensities(const NDVector<Real> &d) {
        if (m_densities.Size() != d.Size())
            throw std::runtime_error("size mismatch"); m_densities = d;
        m_numericFactorizationUpToDate = false;
    }

    const EigenNDIndex &NbElementsPerDimension()        const { return m_NbElementsPerDimension;               }
    const EigenNDIndex &NbNodesPerDimensionPerElement() const { return m_NbNodesPerDimensionPerElement;        }
    const EigenNDIndex &NbNodesPerDimension()           const { return m_NbNodesPerDimension;                  }
    const size_t       NbNodesPerElement()              const { return m_NbNodesPerDimensionPerElement.prod(); }

    const Point &nodePosition(size_t ni) const { return m_nodes[ni].coordinates(); }

    void setE_min(Real val) { m_E_min = val; }
    void setE_0(Real val) { m_E_0 = val; }
    void setSIMPExponent(Real val) { m_gamma = val; }
    Real E_min() const { return m_E_min; }
    Real E_0() const { return m_E_0; }
    Real SIMPExponent() const { return m_gamma; }

    // NDindex of the element in which the point belongs
    EigenNDIndex getElementNDIndex(const Point &point) const {
        EigenNDIndex result;
        for (size_t dimension = 0; dimension < N; ++dimension) {
            Point relativePoint = point - m_domain.minCorner;    // shift origin to domain minimum corner
            Real floatIndex = relativePoint[dimension] / m_element_stretch[dimension];
            // Clamp samples samples on the grid boundary back into the grid.
            if (std::abs(floatIndex - m_NbElementsPerDimension[dimension]) < 1e-10) {
                result[dimension] = m_NbElementsPerDimension[dimension] - 1;
                continue;
            }
            result[dimension] = size_t(floatIndex);
            if (result[dimension] >= m_NbElementsPerDimension[dimension]) {
                std::cout << "dimension = " << dimension << std::endl;
                std::cout << "N = " << N << std::endl;
                std::cout << point << std::endl;
                std::cout << m_element_stretch[dimension] << std::endl;
                std::cout << point[dimension] << std::endl;
                throw std::runtime_error("Point out of bounds: " + std::to_string(result[dimension]) + " vs " + std::to_string(m_NbElementsPerDimension[dimension]));
            }
        }
        return result;
    }

    size_t getElementIndex(const Point &point) const {
        return m_densities.flatIndex(getElementNDIndex(point));
    }

    // Get the index of the element containing point `p_in` and the reference/canonical
    // coordinates of the point within this element.
    std::pair<EigenNDIndex, Point> getElementAndReferenceCoordinates(Eigen::Ref<const Point> p_in) const {
        std::pair<EigenNDIndex, Point> result;
        Point &p_out = result.second;
        result.first = getElementNDIndex(p_in);
        // rescale p into [0,1]^d
        p_out = p_in - nodePosition(m_nodes.flatIndex(firstNodeOfElement(result.first)));
        for (size_t d = 0; d < N; ++d)
            p_out[d] /= getStretchings()[d];
        return result;
    }

    // Set whether the no rigid translation constraint should be implemented
    // using a node pinning constraint.
    void setUsePinNoRigidTranslationConstraint(bool use) {
        m_useNRTPinConstraint = use;
    }

    const VectorND<N> &stretchings() const { return m_element_stretch; }

    const VectorND<N> &getStretchings() const {
        return m_element_stretch;
    }

    // Call visitor(ei, local_n, elementNodes) for each incident element `ei`
    // where `local_n` is the index of `n` in element `ei` and
    // `elementNodes` is an array of global node indices for element `ei`.
    template<class F>
    void visitIncidentElements(EigenNDIndex globalNode, F &&visitor) const {
        // std::cout << "visitIncidentElements " << globalNode.transpose() << std::endl;
        // const size_t n = m_nodes.flatIndex(globalNode);
        EigenNDIndex elem;
        EigenNDIndex &localNode = globalNode;
        // Get the "primary element" and local node index within this element of `globalNode`
        // For nodes appearing in multiple elements, the "primary" element is the one for which
        // the local indices are *lowest*. (For example, in 2D, it is the upper-rightmost element.)
        for (size_t d = 0; d < N; ++d) {
            elem[d]      = globalNode[d] / (m_NbNodesPerDimensionPerElement[d] - 1);
            if (elem[d] == m_NbElementsPerDimension[d]) { // Handle upper grid boundary...
                elem      [d] = m_NbElementsPerDimension[d] - 1;
                localNode [d] = m_NbNodesPerDimensionPerElement[d] - 1;
            }
            else {
                localNode[d] = globalNode[d] % (m_NbNodesPerDimensionPerElement[d] - 1);
            }
        }

        static constexpr size_t maxNeighbors = 1 << N; // std::pow(2, N); -- Unfortunately std::pow is not constexpr on all compilers...
        for (size_t i = 0; i < maxNeighbors; ++i) {
            EigenNDIndex e_idx = elem, localNodeInNeighbor = localNode;
            for (size_t d = 0; d < N; ++d) {
                if ((1 << d) & i) continue;
                if ((localNode[d] != 0) || (e_idx[d] == 0)) goto invalid;
                else {
                    --e_idx[d];
                    localNodeInNeighbor[d] = m_NbNodesPerDimensionPerElement[d] - 1;
                }
            }

            {
                size_t ei = m_densities.flatIndex(e_idx);
                auto enodes = elementNodes(ei);
                // size_t localIndex = std::distance(enodes.begin(), std::find(enodes.begin(), enodes.end(), n));
                // for (size_t enode : enodes)
                //     std::cout << enode << "\t";
                // std::cout << std::endl << n << std::endl;
                // std::cout << localNode.transpose() << std::endl;
                // std::cout << localNodeInNeighbor.transpose() << std::endl;
                // std::cout << localIndex << "\t"
                //           << ElementNodeIndexer::flatIndex(localNodeInNeighbor) << std::endl << std::endl;
                // assert(localIndex == ElementNodeIndexer::flatIndex(localNodeInNeighbor));
                visitor(ei, ElementNodeIndexer::flatIndex(localNodeInNeighbor), enodes);
            }
            invalid: ;
        }
    }

    // Append to dirichletVars and dirichletValues the conditions stored in m_nodes
    // Note: method adapted from LinearElasticity::Simulator (changes due to different structure of nodes and elements)
    void getDirichletVarsAndValues(std::vector<size_t> &dirichletVars,
                                   std::vector<Real> &dirichletValues) const {
        // Validate and convert to per-periodic DoF constraints.
        // constraintDisplacements[i] holds the displacement to which
        // components constraintComponents[i] of DoF constraintDoFs[i] are
        // constrained.
        std::vector<Point>         constraintDisplacements;
        std::vector<int>           constraintDoFs;
        std::vector<ComponentMask> constraintComponents;
        // Index into the above arrays a DoF's constraint, or -1 for none.
        // I.e. if constraintDoFs[i] > -1, the following holds:
        //  constraintDoFs[constraintIndex[i]] = i
        std::vector<int> constraintIndex(numDoFs(), -1);
        for (size_t i = 0; i < m_nodes.Size(); ++i) {
            if (m_nodes[i].hasDirichlet()) {
                int dof = DoF(i); // DoF is same as index i whenever no periodic BCs applied
                if (constraintIndex[dof] < 0) { // No constraint? Then add new displacement constraint
                    constraintIndex[dof] = constraintDoFs.size();
                    constraintDoFs.push_back(dof);
                    constraintDisplacements.push_back(m_nodes[i].dirichletDisplacement());
                    constraintComponents.push_back(m_nodes[i].dirichletComponents());
                }
                else { // Error if there was already a constraint on dof
                    std::cerr << "WARNING: Dirichlet condition on periodic "
                              << "boundary applies to all identified nodes."
                              << std::endl;
                    auto diff = m_nodes[i].dirichletDisplacement() - constraintDisplacements[constraintIndex[dof]];
                    bool cdiffer = (m_nodes[i].dirichletComponents() != constraintComponents[constraintIndex[dof]]);
                    if ((diff.norm() > 1e-10) || cdiffer) {
                        throw std::runtime_error("Mismatched Dirichlet "
                                                         "constraint on periodic DoF");
                    }
                    // Ignore redundant but compatible Dirichlet conditions.
                }
            }
        }

        for (size_t i = 0; i < constraintDoFs.size(); ++i) {
            for (size_t c = 0; c < N; ++c) {
                if (!constraintComponents[i].has(c)) continue;
                dirichletVars.push_back(N * constraintDoFs[i] + c);
                dirichletValues.push_back(constraintDisplacements[i][c]);
            }
        }
    }

    // Get a description of this simulator (currently just the element size)
    const std::string &description() const {
        if (m_description.size() == 0) {
            m_description = std::to_string(m_NbElementsPerDimension[0]);
            for (size_t i = 1; i < m_NbElementsPerDimension.size(); ++i)
                m_description += "x" + std::to_string(m_NbElementsPerDimension[i]);
        }
        return m_description;
    }


private:
    void m_getPeriodicConditionFixedVariables(std::vector<size_t> &fixedVars, std::vector<Real> &fixedVarValues) const {
        if (!periodic_BC)
            throw std::runtime_error("Trying to set fixed variables for periodic conditions when periodic "
                                     + std::string("conditions are not set."));
        fixedVars.clear();
        fixedVarValues.clear();

        for (size_t dim = 0; dim < N; ++dim) {
            // fix the translation by setting the first node to the origin
            // rotations are fixed by periodic conditions on dofs.
            fixedVars.push_back(dim);
            fixedVarValues.push_back(0);
        }
    }

    void m_updateK0() {
        m_element.Stiffness(m_K0, m_E_tensor, 1);
        // Copy upper triangle into the lower triangle
        m_K0.template triangularView<Eigen::StrictlyLower>() =
                m_K0.template triangularView<Eigen::StrictlyUpper>().transpose();
        // std::cout << std::endl;
        // std::cout << m_K0 << std::endl;
        m_numericFactorizationUpToDate = false;
    }

    PerElementStiffness m_K0;
    aligned_std_vector<PerElementStiffness> m_KeCache;

	Element m_element; // Representative grid element (to save memory, we assume a regular grid where all elements are related by a rigid transformation)
	size_t m_numElements; 

    BBox<VectorND<N>> m_domain;                 // The domain
    NDVector<Node<N>> m_nodes;                  // The mesh nodes
    // TODO: allow storing only a single element instance (since all instances should
    // generally be identical).

    ETensor m_E_tensor = ETensor(1, 0);         // The material tensor is the same for all elements
    VectorND<N> m_element_stretch;              // Stretching is also the same for all elements
    NDVector<Real> m_densities;                 // But the density changes

    bool periodic_BC = false;           // Whether periodic condition have been applied
    bool m_useNRTPinConstraint = false;

    EigenNDIndex m_NbNodesPerDimensionPerElement; // Number of nodes belonging to one element, per dimension
    EigenNDIndex m_NbNodesPerDimension;           // Number of nodes per dimension
    EigenNDIndex m_NbElementsPerDimension;        // Number of elements per dimension
    EigenNDIndex m_NodeGlobalIndexIncrement;      // Increment in flattened node index induced by changing a ND index
    ENodesArray  m_referenceElementNodes; // Node indices for the 0th element (from which the others can be determined).

    Real m_E_min = 1e-9;       // minimum Young modulus multiplier (to be used in SIMP formula)
    Real m_E_0 = 1;            // Young modulus multiplier of solid voxels (to be used in SIMP formula)
    Real m_gamma = 3;          // SIMP exponent

    mutable std::string m_description; // simulator description, cached for re-use

    BlockSuiteSparseMatrix m_blockK;

    // Cached solver and sparsity pattern are mutable because they do not affect
    // user-visible state.
    // The sparsity pattern is for the full stiffness matrix (before boundary
    // conditions have been applied) and therefore should remain fixed throughout
    // the simulator's lifetime unless the periodicity conditions change.
    // However, the constructor should not build it since it is not needed for
    // the finer levels of the multigrid solver.
    // The factorizer must be rebuilt from scratch whenever the Dirichlet/
    // periodic conditions change. When just the densities or forces change we can
    // reuse the symbolic factorization, but must update the numeric factorization;
    // this is indicated by `m_numericFactorizationUpToDate == false`.
    mutable SuiteSparseMatrix m_hessianSparsityPattern;
    mutable bool m_numericFactorizationUpToDate = false;
    mutable std::unique_ptr<CholmodFactorizer> m_solver;

    mutable std::unique_ptr<DALocalData<VField>> m_applyKThreadWorkspace;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Work around alignment issues when using SIMD
};

#endif //MESHFEM_TENSORPRODUCTSIMULATOR_HH
