#include <iostream>

#include "../TensorProductSimulator.hh"
#include "../TPPeriodicHomogenization.hh"

template<size_t Dim>
struct Simulator;

// Use bi- and tri-linear simulators
template<> struct Simulator<2> { using type = TensorProductSimulator<1, 1>; };
template<> struct Simulator<3> { using type = TensorProductSimulator<1, 1, 1>; };

template<size_t Dim>
void execute(size_t gridSize, const std::string &materialPath, const std::string &densityPath, const std::string &constraintsPath, const std::string &outPath) {
    // Test for change from multi-index to linear index in ND vector
    /*NDVector<size_t> myNDVector;
    std::vector<size_t> sizeVector(2,65);
    myNDVector.Resize(sizeVector);

    for (size_t i = 0; i < myNDVector.Size(); ++i)
        if (myNDVector.flatIndex(myNDVector.flatToIndices(i)) != i) {
            std::cout << "Error at index " << i << std::endl;
            std::vector<size_t> indices = myNDVector.flatToIndices(i);
            std::cout << "indices of (";
            for (size_t i = 0; i < indices.size()-1; ++i)
                std::cout << indices[i] << ", ";
            std::cout << indices.back() << ")";
            std::cout << "Flat index of indices : " << myNDVector.flatIndex(myNDVector.flatToIndices(i)) << std::endl;
        }*/

    // Create a simulator reading mesh and densities from file

    // MSHFieldParser<Dim> fieldParser(densityPath);
    // const auto &elems = fieldParser.elements();
    // const auto &verts = fieldParser.vertices();
    // BBox<VectorND<Dim>> domainBox(verts);

    // // Create the simulator
    // std::vector<size_t> sizeVector(Dim, gridSize);
    // using Sim = typename Simulator<Dim>::type;
    // // Sim mySimulator(sizeVector); // ../../density_examples/0007_64_density.gmsh

    // // BBox initialization of TPS
    // Sim mySimulator(domainBox, sizeVector);

    // // Read material and densities from external files
    // mySimulator.readMaterial(materialPath);
    // mySimulator.readDensities(densityPath);


    // Create a simulator hardcoding mesh dimensions and densities

    BBox<VectorND<Dim>> domainBox;
    domainBox.minCorner = {0, 0};
    domainBox.maxCorner = {60, 10};

    // Create the simulator
    std::vector<size_t> sizeVector = {60, 10};
    using Sim = typename Simulator<Dim>::type;

    // BBox initialization of TPS
    Sim mySimulator(domainBox, sizeVector);

    // Read material and densities from external files
    mySimulator.readMaterial(materialPath);
    mySimulator.setUniformDensities(1.0);

    std::cout << "Mesh : " << densityPath << std::endl;

    // Siffness matrix
    // TripletMatrix<Triplet<Real>> K;
    // mySimulator.m_assembleStiffnessMatrix(K);
    // mySimulator.writeFEMMatrix("../../matlab_scripts/FEMMatrix", K);

    // Test stiffness and strain on first elements of simulator
    /*Eigen::Matrix<Real, 8, 8> Ke;

    Materials::Constant<2> mat("../../MeshFEM/examples/materials/B9Creator.material");
    auto E = mat.getTensor();

    Element<1,1> firstElem = mySimulator.getFirstElement();

    firstElem.Stiffness(Ke,E, 1);

    std::cout << "First element volume : " << std::endl;
    std::cout << firstElem.Volume() << std::endl;

    std::cout << "Elasticity tensor :" << std::endl;
    std::cout << E << std::endl;

    std::cout << "Stiffness matrix of first element : " << std::endl;
    std::cout << Ke << std::endl;

    auto FirstElemStrains = Strains<1,1>::getStrains(std::array<Real, 2>({{1,1}}));

    for (auto const strain : FirstElemStrains) {
        std::cout << strain[0] << std::endl;
    }

    auto FirstElemGradient = Gradients<1,1>::getGradients(std::array<Real, 2>({{1,1}}));

    for (auto const grad : FirstElemGradient) {
        std::cout << grad(0,0).transpose() << std::endl;
    }
    */

    // bool noRigidMotion;
    // BBox<VectorND<Dim>> domain = mySimulator.domain();
    // auto bconds = readBoundaryConditions<Dim>(constraintsPath, domain, noRigidMotion);
    // mySimulator.applyDisplacementsAndLoads(bconds);

    mySimulator.applyDisplacementsAndLoadsFromFile(constraintsPath);

    // Write results to msh
    std::vector<MeshIO::IOVertex> vertices;
    std::vector<MeshIO::IOElement> elements;
    mySimulator.getMesh(vertices, elements);

    using Vector = VectorND<Dim>;
    BBox<Vector> bb(vertices);

    MSHFieldWriter writer(outPath, vertices, elements, (Dim == 2) ? MeshIO::MESH_QUAD : MeshIO::MESH_HEX); // MESH_HEX or MESH_QUAD

    writer.addField("density", ScalarField<double>(mySimulator.getDensities()), DomainType::PER_ELEMENT);

    try {

        // Define external forces other than Neumann loads
        // auto myRHS = mySimulator.constantStrainLoad(-SymmetricMatrixValue<Real, Dim>::CanonicalBasis(0));
        // myRHS.setConstant(0);  // Try with no loads...

        // Solve
        auto solution = mySimulator.solveWithImposedLoads();

        auto toVF = [](const typename Sim::VField &v) { return VectorField<Real, Dim>(Eigen::Map<const Eigen::VectorXd>(v.data(), v.size())); };

        // Write to output
        // writer.addField("RHS", toVF(mySimulator.dofToNodeField(myRHS)), DomainType::PER_NODE); 
        writer.addField("RHS", toVF(mySimulator.buildLoadVector()), DomainType::PER_NODE);
        writer.addField("Sol", toVF(solution), DomainType::PER_NODE);

//         // Get Homogenized elasticity tensor
//         auto w_ijs = TPPeriodicHomogenization::solveCellProblems(mySimulator);
//         auto Eh = TPPeriodicHomogenization::homogenizedElasticityTensor(w_ijs, mySimulator);
//
//
//         std::cout << "Homogenized tensor:" << std::endl << Eh << std::endl;
//         std::cout << "Orthotropic: " << std::endl;
//         Eh.printOrthotropic(std::cout);
//
//         const size_t nn = mySimulator.numNodes();
//
//         // Center each fluctuation displacement so that the average normal
//         // displacement on each periodic boundary is zero.
//         for (auto &w : w_ijs) {
//             VectorND<Dim> translation;
//             translation.setZero();
//             std::vector<size_t> numAveraged(Dim);
//             for (size_t i = 0; i < nn; ++i) {
//                 auto pt = mySimulator.nodePosition(i);
//                 for (size_t d = 0; d < Dim; ++d) {
//                     if (std::abs(pt[d] - bb.minCorner[d]) < 1e-10) {
//                         translation += w(i);
//                         ++numAveraged[d];
//                     }
//                 }
//             }
//             for (size_t d = 0; d < Dim; ++d)
//                 translation[d] /= numAveraged[d];
//             for (size_t i = 0; i < w.domainSize(); ++i)
//                 w(i) -= translation;
//         }
//
//         // Output the actual microstructure displacement under a probing stress tensor.
//         using SMatrix = typename Sim::SMatrix;
//         using VField  = typename Sim::VField;
//
//         auto probeStress = SMatrix::CanonicalBasis(0);
//         auto probeStrain = Eh.inverse().doubleContract(probeStress);
//         probeStrain *= 1.0 / std::sqrt(probeStrain.frobeniusNormSq());
//
//         VField microDisp(nn);
//         for (size_t i = 0; i < nn; ++i)
//             microDisp(i) = probeStrain.contract(mySimulator.nodePosition(i) - bb.center());
//
//         for (size_t i = 0; i < w_ijs.size(); ++i) {
//             VField tmp(w_ijs[i]);
//             tmp *= (((i < Dim) ? 1.0 : 2.0) * probeStrain[i]);
//             microDisp += tmp;
//         }
//
//         writer.addField("micro displacement", microDisp, DomainType::PER_NODE);
//
//         // Objective elasticity tensor
//         typename Sim::ETensor objective(1.5, 0.578957);
//
// #if 0
//         // Gradient descent
//         std::cout << " ------- Gradient descent:" << std::endl;
//         TPPeriodicHomogenization::gradientDescent(mySimulator, objective, w_ijs, 1e-3, 1e-10, 20, writer, true);
// #endif
    }
    catch (const std::exception &e) {
        std::cout << "Exception during homogenization:" << e.what() << std::endl;
    }
}

int main(int argc, const char *argv[]) {
    if (argc != 5) {
        std::cerr << "usage: test_optimization material.mat initial_densities.msh boundary_conditions.bc out.msh" << std::endl;
        exit(-1);
    }

    const std::string materialPath(argv[1]), densityPath(argv[2]), constraintsPath(argv[3]), outPath(argv[4]);

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(densityPath, vertices, elements);

    if (type == MeshIO::MESH_QUAD) {
        size_t gridSize = std::round(std::sqrt(elements.size()));
        if (gridSize * gridSize != elements.size())
            throw std::runtime_error("Non-square grid");
        execute<2>(gridSize, materialPath, densityPath, constraintsPath, outPath);
    }
    else if (type == MeshIO::MESH_HEX) {
        size_t gridSize = std::round(std::cbrt(elements.size()));
        if (gridSize * gridSize * gridSize != elements.size())
            throw std::runtime_error("Non-cube grid");
        execute<3>(gridSize, materialPath, densityPath, constraintsPath, outPath);

    }
    else throw std::runtime_error("Unsupported input density mesh type");

    BENCHMARK_REPORT();

    return 0;
}
