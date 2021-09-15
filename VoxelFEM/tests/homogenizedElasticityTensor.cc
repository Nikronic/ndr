#include <iostream>

#include "../TensorProductSimulator.hh"
#include "../TPPeriodicHomogenization.hh"

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
            ("initialDensities",                 po::value<std::string>(),        "mesh of initial densities");
    po::positional_options_description p;
    p.add("initialDensities",                    1);

    po::options_description visible_opts;
    visible_opts.add_options()("help", "Produce this help message")
            ("material,m",           po::value<std::string>()->default_value(""), "simulation material path");

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

    if (fail || vm.count("help"))
        usage(fail, visible_opts);

    return vm;
}



template<size_t Dim>
struct Simulator;

// Use bi- and tri-linear simulators
template<> struct Simulator<2> { using type = TensorProductSimulator<1, 1>; };
template<> struct Simulator<3> { using type = TensorProductSimulator<1, 1, 1>; };


template<size_t Dim>
void execute(size_t gridSize, const std::string &densityPath, const std::string &materiaPath) {

    // Create the simulator
    std::vector<size_t> sizeVector(Dim,gridSize);
    using Sim = typename Simulator<Dim>::type;
    Sim mySimulator(sizeVector);
    // Set the simulator stretching to 1
    VectorND<Dim> stretchings(VectorND<Dim>::Ones());
    mySimulator.setStretchings(stretchings);

    // applyPeriodicConditions
    mySimulator.readMaterial(materiaPath);
    mySimulator.readDensities(densityPath);
    mySimulator.applyPeriodicConditions();

    // Constant strain load
    auto myRHS = mySimulator.constantStrainLoad(-SymmetricMatrixValue<Real, Dim>::CanonicalBasis(0));

    try {
        // Compute fluctuation field w_ijs
        auto w_ijs = TPPeriodicHomogenization::solveCellProblems(mySimulator);
        // Compute Homogenized elasticity tensor
        auto Eh = TPPeriodicHomogenization::homogenizedElasticityTensor(w_ijs, mySimulator);

        std::cout << "Homogenized tensor:" << std::endl << Eh << std::endl;
        std::cout << "Orthotropic: " << std::endl;
        Eh.printOrthotropic(std::cout);
    }
    catch (const std::exception &e) {
        std::cout << "Exception during homogenization:" << e.what() << std::endl;
    }
}


////////////////////////////////////////////////////////////////////////////////
/*! Program entry point
//  @param[in]  argc    Number of arguments
//  @param[in]  argv    Argument strings
//  @return     status  (0 on success)
*///////////////////////////////////////////////////////////////////////////////
int main(int argc, const char *argv[]) {

    po::variables_map args = parseCmdLine(argc, argv);

    const std::string densityPath( args["initialDensities"].as<std::string>() ),
                      materialPath(args["material"].as<std::string>());

    std::vector<MeshIO::IOVertex > vertices;
    std::vector<MeshIO::IOElement> elements;
    auto type = MeshIO::load(densityPath, vertices, elements);

    if (type == MeshIO::MESH_QUAD) {
        size_t gridSize = std::round(std::sqrt(elements.size()));
        if (gridSize * gridSize != elements.size())
            throw std::runtime_error("Non-square grid");
        execute<2>(gridSize, densityPath, materialPath);
    }
    else if (type == MeshIO::MESH_HEX) {
        size_t gridSize = std::round(std::cbrt(elements.size()));
        if (gridSize * gridSize * gridSize != elements.size())
            throw std::runtime_error("Non-cube grid");
        execute<3>(gridSize, densityPath, materialPath);
    }
    else throw std::runtime_error("Unsupported input density mesh type");

    BENCHMARK_REPORT();

    return 0;
}
