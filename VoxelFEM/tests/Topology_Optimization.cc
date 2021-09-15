#include <iostream>

#include "../TensorProductSimulator.hh"
#include "../knitro_optimization.hh"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace po = boost::program_options;

void usage(int exitVal, const po::options_description &visible_opts) {
    std::cout << "Usage: Simulate_cli [options] mesh" << std::endl;
    std::cout << visible_opts << std::endl;
    exit(exitVal);
}

po::variables_map parseCmdLine(int argc, const char *argv[])
{

    po::options_description hidden_opts("Hidden Arguments");
    hidden_opts.add_options()
            ("initialDensities",                 po::value<std::string>(),        "mesh of initial densities")
            ;
    po::positional_options_description p;
    p.add("initialDensities",                    1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
            ("material,m",           po::value<std::string>()->default_value(""), "simulation material path")
            ("youngModulus,E",       po::value<double>(),                         "target Young modulus")
            ("PoissonRatio,r",      po::value<double>(),                         "target Poisson's ratio")
            ("SIMPexponent,p",   po::value<int>()->default_value(5),          "SIMP exponent")
            ("smoothType,t",           po::value<std::string>(),                    "Smoothness regularization term type")
            ("SmoothnessWeight,s", po::value<double>()->default_value(1.0),     "Smoothness Regularization weight")
            ("IntegerWeight,i",    po::value<double>()->default_value(1.0),     "Integer Regularization weight")
            ("volumeWeight,v",     po::value<double>()->default_value(0.0),     "Volume Regularization weight")
            ("outputMSH,o",          po::value<std::string>(),                    "output mesh")
            ;

    po::options_description cli_opts;
    cli_opts.add(visible_opts).add(hidden_opts);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).
                options(cli_opts).positional(p).run(), vm);
        po::notify(vm);
    }
    catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl << std::endl;
        usage(1, visible_opts);
    }

    bool fail = false;
    if (vm.count("initialDensities") == 0) {
        std::cout << "Error: must specify mesh of initial densities" << std::endl;
        fail = true;
    }
    if (vm.count("youngModulus") == 0) {
        std::cout << "Error: must specify a target Young modulus" << std::endl;
        fail = true;
    }
    if (vm.count("PoissonRatio") == 0) {
        std::cout << "Error: must specify a target Poisson's ratio" << std::endl;
        fail = true;
    }
    if (vm.count("outputMSH") == 0) {
        std::cout << "Error: must specify output msh file" << std::endl;
        fail = true;
    }

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}


template<size_t Dim>
struct Simulator;

// Use bi- and tri-linear simulators
template<> struct Simulator<2> { using type = TensorProductSimulator<1, 1>; };
template<> struct Simulator<3> { using type = TensorProductSimulator<1, 1, 1>; };

////////////////////////////////////////////////////////////////////////////////
/*! Solve the microstructure optimization problem using the knitro solver
//  @param[in] gridSize         Number of element per dimension in the initial mesh
//  @param[in] densityPath      Path to the .msh containing the initial microstructure densities
//  @param[in] materialPath     Path to the file describing the base material
        The target material is isotropic, thus only youg modulus and Poisson ratio are required to define it
//  @param[in] youngTarget      Young modulus of the target Elasticity tensor
//  @param[in] poissonTarget    Poisson's ratio of the target Elasticity tensor
//  @param[in] simpPower        SIMP exponent
//  @param[in] smoothType       Designs the formulation to use for the smoothness regularization term
//  @param[in] smoothnessWeight
*///////////////////////////////////////////////////////////////////////////////
template<size_t Dim>
void execute(size_t gridSize, const std::string &densityPath, const std::string &materialPath,
             Real youngTarget, Real poissonTarget,
             Real simpPower,SmoothingType smoothType,
             Real integerWeight, Real smoothnessWeight, Real volumeWeight,
             const std::string &outPath) {
    // lower bound for the densities, to ensure the finite element matrix is positive definite
    Real density_lower_bound = 1e-4;

    std::vector<size_t> sizeVector(Dim, gridSize);
    using Sim = typename Simulator<Dim>::type;
    Sim mySimulator(sizeVector);
    // Set the simulator stretching to 1
    std::array<Real, Dim> stretchings;
    stretchings.fill(1);
    mySimulator.setStretchings(stretchings);

    // applyPeriodicConditions
    mySimulator.readMaterial(materialPath);
    mySimulator.readDensities(densityPath);
    mySimulator.applyPeriodicConditions();

    // Constant strain load
    auto myRHS = mySimulator.constantStrainLoad(-SymmetricMatrixValue<Real, Dim>::CanonicalBasis(0));

    // Create writer to output to mesh
    std::vector<MeshIO::IOVertex> vertices;
    std::vector<MeshIO::IOElement> elements;
    mySimulator.getMesh(vertices, elements);
    MSHFieldWriter writer(outPath, vertices, elements, (Dim == 2) ? MeshIO::MESH_QUAD : MeshIO::MESH_HEX);

    // Objective elasticity tensor
    typename Sim::ETensor objective(youngTarget, poissonTarget);

    // Optimization using knitro solver
    optimize_knitro(mySimulator, objective, writer, 10000,
                    smoothnessWeight, integerWeight,volumeWeight,
                    smoothType, simpPower, density_lower_bound, "");
}

////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[]) {

    po::variables_map args = parseCmdLine(argc, argv);

    const std::string densityPath(     args["initialDensities"].as<std::string>() ),
                      materialPath(    args["material"].as<std::string>()         ),
                      outPath(         args["outputMSH"].as<std::string>()        ),
                      smoothTypeString(args["smoothType"].as<std::string>()       );
    Real youngTarget      = args["youngModulus"].as<double>(),
         poissonTarget    = args["PoissonRatio"].as<double>(),
         simpPower        = args["SIMPexponent"].as<int>(),
         smoothnessWeight = args["SmoothnessWeight"].as<double>(),
         integerWeight    = args["IntegerWeight"].as<double>(),
         volumeWeight     = args["volumeWeight"].as<double>();

    SmoothingType stype;
    if      (smoothTypeString ==   "laplacian") { stype = SmoothingType::Laplacian; }
    else if (smoothTypeString == "bilaplacian") { stype = SmoothingType::Bilaplacian; }
    else throw std::runtime_error("Invalid smoothing type; must be laplacian or bilaplacian");

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(densityPath, vertices, elements);

    if (type == MeshIO::MESH_QUAD) {
        size_t gridSize = std::round(std::sqrt(elements.size()));
        if (gridSize * gridSize != elements.size())
            throw std::runtime_error("Non-square grid");
        execute<2>(gridSize, densityPath, materialPath, youngTarget, poissonTarget, simpPower, stype, smoothnessWeight, integerWeight, volumeWeight, outPath);
    }
    else if (type == MeshIO::MESH_HEX) {
        size_t gridSize = std::round(std::cbrt(elements.size()));
        if (gridSize * gridSize * gridSize != elements.size())
            throw std::runtime_error("Non-cube grid");
        execute<3>(gridSize, densityPath, materialPath, youngTarget, poissonTarget, simpPower, stype, smoothnessWeight, integerWeight, volumeWeight, outPath);
    }
    else throw std::runtime_error("Unsupported input density mesh type");

    BENCHMARK_REPORT();

    return 0;
}
