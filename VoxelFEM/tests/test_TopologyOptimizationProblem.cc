#include <iostream>
#include <memory>
#include "../TopologyOptimizationProblem.hh"
#include "../TopologyOptimizationFilter.hh"

template<size_t Dim>
struct Simulator;

template<> struct Simulator<2> { using type = TensorProductSimulator<1, 1>; };
template<> struct Simulator<3> { using type = TensorProductSimulator<1, 1, 1>; };

int main(int argc, const char *argv[]) {

    using Sim2 = typename Simulator<2>::type;
    using Sim3 = typename Simulator<3>::type;

    // --------------------------------------
    // -                2D                  -
    // --------------------------------------

    // Problem parameters
    BBox<VectorND<2>> domainBox;
    domainBox.minCorner = {0, 0};
    domainBox.maxCorner = {2, 1};
    std::vector<size_t> sizeVector = {4, 2};
    double totalVolumePercentage = 0.6;
    double uniformDensity = 1.0;
    size_t filterRadius = 1;  // 0 indicates no filtering

    // Initialize TPS
    Sim2 simulator(domainBox, sizeVector);
    simulator.readMaterial(argv[1]);
    for(size_t i = 0; i < simulator.numElements(); i++)
        simulator.setElementDensity(i, uniformDensity); 
    simulator.applyDisplacementsAndLoadsFromFile(argv[2]);
    auto displacementsIndicators = simulator.getDirichletMask();
    auto forcesIndicators = simulator.getForceMask();

    // Dump stiffness matrix for Matlab processing
    TripletMatrix<Triplet<Real>> K;
    simulator.m_assembleStiffnessMatrix(K);
    simulator.writeFEMMatrix("./FEMMatrix", K);
    // simulator.writeFEMMatrixFixedDOFs("./FEMMatrixFixedDOFs");

    // Solve the elastic problem
    auto solution = simulator.solveWithImposedLoads();

    // Write solution of the elastic problem to file
    std::vector<MeshIO::IOVertex> vertices;
    std::vector<MeshIO::IOElement> elements;
    simulator.getMesh(vertices, elements);
    MSHFieldWriter writer("output.msh", vertices, elements, MeshIO::MESH_QUAD);
    writer.addField("Density", ScalarField<double>(simulator.getDensities()), DomainType::PER_ELEMENT);
    auto toVF = [](const typename Sim2::VField &v) { return VectorField<Real, 2>(Eigen::Map<const Eigen::VectorXd>(v.data(), v.size())); };
    writer.addField("RHS", toVF(simulator.buildLoadVector()), DomainType::PER_NODE);
    writer.addField("Solution", toVF(solution), DomainType::PER_NODE);

    // Initialize objective
    std::shared_ptr<Objective<Sim2>> objective = std::make_shared<ComplianceObjective<Sim2>>(simulator);

    // Initialize filters
    std::shared_ptr<Filter> projFilt = std::make_shared<ProjectionFilter>();
    std::shared_ptr<Filter> smoothFilt = std::make_shared<SmoothingFilter>();
    std::shared_ptr<Filter> langFilt = std::make_shared<LangelaarFilter>();
    std::vector<std::shared_ptr<Filter>> filters({smoothFilt, projFilt, langFilt});

    // Initialize constraints
    std::shared_ptr<Constraint> totalVolumeConstr = std::make_shared<TotalVolumeConstraint>(totalVolumePercentage);
    std::vector<std::shared_ptr<Constraint>> constraints({totalVolumeConstr});

    // Initialize TopologyOptimizationProblem and test function evaluations
    TopologyOptimizationProblem<Sim2> TOproblem(simulator, objective, constraints, filters);
    TopologyOptimizationProblem<Sim2> TOproblemNoFilters(simulator, objective, constraints);
    TOproblem.setVars(simulator.getDensities());
    TOproblemNoFilters.setVars(simulator.getDensities());
    Real TOobjective = TOproblem.evaluateObjective();
    Eigen::VectorXd TOconstraints = TOproblem.evaluateConstraints();
    Eigen::VectorXd TOgradient = TOproblem.evaluateObjectiveGradient();
    Eigen::MatrixXd TOjacobian = TOproblem.evaluateConstraintsJacobian();

    // --------------------------------------
    // -                3D                  -
    // --------------------------------------

    // Domain
    BBox<VectorND<3>> domainBox3D;
    domainBox3D.minCorner = {0, 0, 0};
    domainBox3D.maxCorner = {2, 1, 1};
    std::vector<size_t> sizeVector3D = {4, 2, 2};

    // Additional parameters are the same as in the 2D problem

    // Simulator
    Sim3 simulator3D(domainBox3D, sizeVector3D);
    simulator3D.readMaterial(argv[1]);
    for(size_t i = 0; i < simulator3D.numElements(); i++)
        simulator3D.setElementDensity(i, uniformDensity); 
    simulator3D.applyDisplacementsAndLoadsFromFile(argv[2]);

    // Objective
    std::shared_ptr<Objective<Sim3>> objective3D = std::make_shared<ComplianceObjective<Sim3>>(simulator3D);

    // Filters and constraints are the same as in the 2D problem

    // TopologyOptimizationProblem
    TopologyOptimizationProblem<Sim3> TOproblem3D(simulator3D, objective3D, constraints, filters);
    TOproblem3D.setVars(simulator3D.getDensities());
    Real TOobjective3D = TOproblem3D.evaluateObjective();
    Eigen::VectorXd TOconstraints3D = TOproblem3D.evaluateConstraints();
    Eigen::VectorXd TOgradient3D = TOproblem3D.evaluateObjectiveGradient();
    Eigen::MatrixXd TOjacobian3D = TOproblem3D.evaluateConstraintsJacobian();

    BENCHMARK_REPORT();

    return 0;
}
