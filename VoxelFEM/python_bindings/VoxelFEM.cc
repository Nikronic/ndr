#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include <MeshFEM/Utilities/MeshConversion.hh>
#include "../TensorProductSimulator.hh"
#include "../TopologyOptimizationProblem.hh"
#include "../TopologyOptimizationFilter.hh"
#include "../NDVector.hh"
#include "../OptimalityCriterion.hh"

#include <map>
#include <functional>
#include <utility>

template<size_t Degree>
std::string degreeString() { return std::to_string(Degree); }

template<size_t Degree1, size_t Degree2, size_t... Degrees>
std::string degreeString() { return std::to_string(Degree1) + "_" + degreeString<Degree2, Degrees...>(); }

// We need to create a distinct python type per template instantiation of
// TensorProductSimulator; we use a "name mangling" scheme to give each of
// these types distinct names. These types will be hidden in
// `pyVoxelFEM.detail` to make the module's interface a bit cleaner.
template<size_t... Degrees>
std::string nameMangler(const std::string &name) {
    return name + degreeString<Degrees...>();
}

// "Factory" for creating an instantiation of the appropriate tensor product
// simulator type.
using FactoryType = std::function<py::object(const std::array<Eigen::VectorXd, 2> &, const std::vector<size_t> &)>;
static std::map<std::vector<size_t>, FactoryType> g_factories;

template<size_t... Degrees>
void addBindings(py::module &m, py::module &detail_module) {
    using TPS          = TensorProductSimulator<Degrees...>;
    using MG           = MultigridSolver<Degrees...>;
    using VField       = typename TPS::VField;
    using EigenNDIndex = typename TPS::EigenNDIndex;
    constexpr size_t N = sizeof...(Degrees);

    py::class_<TPS, std::shared_ptr<TPS>>(detail_module, (nameMangler<Degrees...>("TensorProductSimulator")).c_str())
        .def(py::init<const BBox<VectorND<N>> &, const std::vector<size_t> &>(), py::arg("domainBoundingBox"), py::arg("numElemg"))

        .def("numNodes",                           &TPS::numNodes)
        .def("numElements",                        &TPS::numElements)
        .def("getDensities",                       &TPS::getDensities)
        .def("readDensities",                      &TPS::readDensities,       py::arg("materialPath"), py::arg("fieldName") = "density")
        .def("readMaterial",                       &TPS::readMaterial,        py::arg("materialPath"))
        .def("setElementDensity",                  &TPS::setElementDensity,   py::arg("ei"),           py::arg("value"))
        .def("setUniformDensities",                &TPS::setUniformDensities, py::arg("density"))
        .def("elementDensity",                     &TPS::elementDensity,      py::arg("ei"))
        .def("solve",                              &TPS::solve,               py::arg("f"))
        .def("solveWithImposedLoads",              &TPS::solveWithImposedLoads)
        .def("multigridSolver",                    [](std::shared_ptr<TPS> tps, size_t numCoarseningLevels) { return std::make_shared<MG>(tps, numCoarseningLevels); })
        .def("getK",                               &TPS::getK)
        .def("applyK",                             [](const TPS &tps, const VField &u) { return tps.applyK(u); }, py::arg("u"))
        .def("getDirichletVarsAndValues",          [](const TPS &tps) { std::vector<size_t> vars; std::vector<Real> vals; tps.getDirichletVarsAndValues(vars, vals); return std::make_tuple(vars, vals); })
        .def("buildLoadVector",                    &TPS::buildLoadVector)
        .def("constantStrainLoad",                 &TPS::constantStrainLoad,  py::arg("eps"))
        .def("nodePosition",                       &TPS::nodePosition,        py::arg("ni"))
        .def("applyDisplacementsAndLoadsFromFile", &TPS::applyDisplacementsAndLoadsFromFile, py::arg("bcPath"))
        .def("elementIndexForGridCell",            &TPS::elementIndexForGridCell, py::arg("cellIdxs"))
        .def("getForceMask",                       &TPS::getForceMask)
        .def("elementStiffnessMatrix",             &TPS::elementStiffnessMatrix, py::arg("ei"))
        .def("elementNodes",                       [](const TPS &s, size_t ei) { return s.elementNodes(ei); }, py::arg("ei"))
        .def("elemNodeGlobalIndex",                [](const TPS &s, size_t ei, size_t n) { return s.elemNodeGlobalIndex(ei, n); }, py::arg("ei"), py::arg("n"))
        .def_property("dirichletMask",             &TPS::getDirichletMask,   &TPS::setDirichletMask)
        .def_property("dirichletValues",           &TPS::getDirichletValues, &TPS::setDirichletValues)
        .def_property("ETensor",                   &TPS::getETensor, &TPS::setETensor)
        .def_property("E_0",                       &TPS::E_0, &TPS::setE_0)
        .def_property("E_min",                     &TPS::E_min, &TPS::setE_min)
        .def_property("gamma",                     &TPS::SIMPExponent, &TPS::setSIMPExponent)
        .def("fullDensityElementStiffnessMatrix", &TPS::fullDensityElementStiffnessMatrix)
        .def("clearCachedElementStiffness",       &TPS::clearCachedElementStiffness)

        // `getMesh` takes ouput arrays as references, and we cannot easily bind this
        // type of method. But we can make custom bindings
        // function directly.
        .def("getMesh", [](const TPS &tps) {
                std::vector<MeshIO::IOVertex > vertices;
                std::vector<MeshIO::IOElement> elements;
                tps.getMesh(vertices, elements);
                return std::make_tuple(getV(vertices), getF(elements));
            })
    ;

    py::class_<MG, std::shared_ptr<MG>>(detail_module, (nameMangler<Degrees...>("MultigridSolver")).c_str())
        // .def(py::init<std::shared_ptr<TPS>, size_t>(), py::arg("numElemPerDimMin"), py::arg("numGrid"))
        .def("getSimulator",    py::overload_cast<const size_t>(&MG::getSimulator), py::arg("l"), py::return_value_policy::reference)
        .def("computeResidual", &MG::computeResidual,                               py::arg("l"), py::arg("u"), py::arg("b"))
        .def("applyK", py::overload_cast<const size_t, const VField &>(&MG::applyK), py::arg("l"), py::arg("u"))
        .def("smoothing", [](MG &mg, const size_t l, const VField &u, const VField &b) {
            VField result(u);
            mg.updateElementStiffnessMatrices();
            mg.smoothing(l, result, b);
            return result;
        }, py::arg("l"), py::arg("u"), py::arg("b"))
        .def("zeroOutDirichletComponents", [](MG &mg, const size_t l, VField u) {
                mg.zeroOutDirichletComponents(l, u);
                return u;
            }, py::arg("l"), py::arg("u"))
        .def("updateElementStiffnessMatrices", &MG::updateElementStiffnessMatrices)
        .def("setSymmetricGaussSeidel",        &MG::setSymmetricGaussSeidel, py::arg("symmetric"))
        .def("solve", [](MG &mg, const VField &u, const VField &f, size_t numSteps, size_t numSmoothingSteps, bool stiffnessUpdated,
                        bool zeroDirichlet, std::function<void(size_t, const VField &)> it_callback, bool fmg) {
                    return mg.solve(u, f, numSteps, numSmoothingSteps, stiffnessUpdated, zeroDirichlet, it_callback, fmg);
                },
                py::arg("u"), py::arg("f"), py::arg("numSteps"),
                py::arg("numSmoothingSteps"), py::arg("stiffnessUpdated") = false,
                py::arg("zeroDirichlet") = false, py::arg("it_callback") = nullptr,
                py::arg("fullMultigrid") = false)
        .def("preconditionedConjugateGradient", &MG::preconditionedConjugateGradient,
                py::arg("u"), py::arg("b"), py::arg("maxIter"), py::arg("tol"),
                py::arg("it_callback") = nullptr,
                py::arg("mgIterations") = 1,
                py::arg("mgSmoothingIterations") = 1,
                py::arg("fullMultigrid") = false)
        .def("debug_get_x", &MG::debug_get_x, py::arg("l"))
        .def("debug_get_b", &MG::debug_get_b, py::arg("l"))
        .def("updateBlockKs", &MG::updateBlockKs)
        .def("debugMulticolorVisit", &MG::debugMulticolorVisit)
        .def_readwrite("buildBlockStiffnessMatrices",     &MG::buildBlockStiffnessMatrices)
        .def_readwrite("buildFinestBlockStiffnessMatrix", &MG::buildFinestBlockStiffnessMatrix)
    ;

    // Register the factory for this TensorProductSimulator instantiation in g_factories
    std::vector<size_t> degs{Degrees...};

    g_factories.emplace(std::make_pair(degs, [](const std::array<Eigen::VectorXd, 2> &extremeNodes, const std::vector<size_t> &elementsPerDimension) {
            // Note: while py::cast is not yet documented in the official documentation,
            // it accepts the return_value_policy as discussed in:
            //      https://github.com/pybind/pybind11/issues/1201
            // by setting the return value policy to take_ownership, we can avoid
            // memory leaks and double frees regardless of the holder type for FEMMesh.

            // Initialize domain bounding box and call appropriate constructor
            BBox<VectorND<N>> domain;
            domain.minCorner = extremeNodes[0];
            domain.maxCorner = extremeNodes[1];
            return py::cast(new TPS(domain, elementsPerDimension), py::return_value_policy::take_ownership);
        }));

    using TOProblem = TopologyOptimizationProblem<TPS>;
    using FiltersList = typename TOProblem::FiltersList;
    using ConstraintsList = typename TOProblem::ConstraintsList;
    using ObjectivePtr = typename TOProblem::ObjectivePtr;

    // "Constructor" functions overloads that will create an instance of the appropriate instantiation.
    m.def("TopologyOptimizationProblem", [](TPS &tps, ObjectivePtr objective, ConstraintsList constraints, FiltersList filters) { 
        return std::make_unique<TOProblem>(tps, objective, constraints, filters); 
        }, py::arg("simulator"), py::arg("objective"), py::arg("constraints"), py::arg("filters"))
    ;
    m.def("ComplianceObjective", [](TPS &tps) {
            return std::make_shared<ComplianceObjective<TPS>>(tps); 
        }, py::arg("simulator"))
    ;
    m.def("MultigridComplianceObjective", [](std::shared_ptr<MG> mg) {
            return std::make_shared<MultigridComplianceObjective<TPS>>(mg); 
        }, py::arg("mg_solver"))
    ;

    // Topology Optimization problem
    py::class_<TOProblem>(detail_module, (nameMangler<Degrees...>("TopologyOptimizationProblem")).c_str())
        .def(py::init<TPS&, ObjectivePtr, ConstraintsList, FiltersList>())
        .def("evaluateObjective",           &TOProblem::evaluateObjective)
        .def("evaluateObjectiveGradient",   &TOProblem::evaluateObjectiveGradient)
        .def("evaluateConstraints",         &TOProblem::evaluateConstraints)
        .def("evaluateConstraintsJacobian", &TOProblem::evaluateConstraintsJacobian)
        .def("numVars",                     &TOProblem::numVars)
        .def("getVars",                     [](const TOProblem &top) -> Eigen::VectorXd { return top.getVars(); }) // force a copy to be returned.
        .def("setVars",                     &TOProblem::setVars, py::arg("x"), py::arg("forceUpdate") = false)
        .def("getDensities",                &TOProblem::getDensities)
        .def_property("objective",          &TOProblem::getObjective, &TOProblem::setObjective)
        .def_property("filters",            &TOProblem::getFilters, &TOProblem::setFilters)
        .def_property("constraints",        &TOProblem::getConstraints, &TOProblem::setConstraints)
    ;

    // Objective functions
    py::class_<Objective<TPS>, std::shared_ptr<Objective<TPS>>>(detail_module, (nameMangler<Degrees...>("Objective")).c_str())
        ;
    using CO = ComplianceObjective<TPS>;
    py::class_<CO, Objective<TPS>, std::shared_ptr<CO>>(
        detail_module, (nameMangler<Degrees...>("ComplianceObjective")).c_str())
        .def("gradient",   &CO::gradient)
        .def("compliance", &CO::compliance)
        .def("u",          &CO::u)
        .def("f",          &CO::f)
        ;
    using MGCO = MultigridComplianceObjective<TPS>;
    py::class_<MGCO, ComplianceObjective<TPS>, std::shared_ptr<MGCO>>(
        detail_module, (nameMangler<Degrees...>("MultigridComplianceObjective")).c_str())
        .def_readonly ("mg",                    &MGCO::mg)
        .def_readwrite("cgIter",                &MGCO::cgIter)
        .def_readwrite("tol",                   &MGCO::tol)
        .def_readwrite("mgIterations",          &MGCO::mgIterations)
        .def_readwrite("mgSmoothingIterations", &MGCO::mgSmoothingIterations)
        .def_readwrite("fullMultigrid",         &MGCO::fullMultigrid)
        .def_readwrite("zeroInit",              &MGCO::zeroInit)
        .def_readwrite("residual_cb",           &MGCO::residual_cb)
        ;

    // Optimization algorithms
    using OCO = OCOptimizer<TOProblem>;
    py::class_<OCO>(detail_module, (nameMangler<Degrees...>("OCOptimizer")).c_str())
        .def(py::init<TOProblem &>(), py::arg("problem"))
        .def("step", &OCO::step, py::arg("m") = 0.2, py::arg("ctol") = 1e-6)
        ;
    m.def("OCOptimizer", [](TOProblem &p) { return std::make_unique<OCO>(p); });
}

PYBIND11_MODULE(pyVoxelFEM, m)
{
    m.doc() = "Voxel-based finite element codebase";

    py::module::import("MeshFEM");

    py::module detail_module = m.def_submodule("detail");

    addBindings<1, 1>   (m, detail_module);
    // addBindings<2, 2>   (m, detail_module);
    addBindings<1, 1, 1>(m, detail_module);
    // addBindings<2, 2, 2>(m, detail_module);

    // Factory function masquerading as a Python class; based on the list of
    // degrees-per-dimension passed, we instantiate the correct type of
    // TensorProductSimulator.
    m.def("TensorProductSimulator", [](const std::vector<size_t> &degreesPerDimension,
                                       const std::array<Eigen::VectorXd, 2> &extremeNodes,
                                       const std::vector<size_t> &elementsPerDimension) {
                auto it = g_factories.find(degreesPerDimension);
                if (it == g_factories.end()) throw std::runtime_error("No template instantiation matching degreesPerDimension!");
                return it->second(extremeNodes, elementsPerDimension); // call the appropriate factory function, passing elementsPerDimension
            }, py::arg("degreesPerDimension"), py::arg("domainBBox"), py::arg("elementsPerDimension"));

    ////////////////////////////////////////////////////////////////////////////////
    // Benchmarking
    ////////////////////////////////////////////////////////////////////////////////
    m.def("benchmark_reset", &BENCHMARK_RESET);
    m.def("benchmark_start_timer_section", &BENCHMARK_START_TIMER_SECTION, py::arg("name"));
    m.def("benchmark_stop_timer_section",  &BENCHMARK_STOP_TIMER_SECTION,  py::arg("name"));
    m.def("benchmark_start_timer",         &BENCHMARK_START_TIMER,         py::arg("name"));
    m.def("benchmark_stop_timer",          &BENCHMARK_STOP_TIMER,          py::arg("name"));
    m.def("benchmark_report", [](bool includeMessages) {
            py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
            if (includeMessages) BENCHMARK_REPORT(); else BENCHMARK_REPORT_NO_MESSAGES();
        },
        py::arg("include_messages") = false)
        ;

    // Filters
    py::class_<Filter, std::shared_ptr<Filter>>(detail_module, "Filter");
    py::class_<PythonFilter, Filter, std::shared_ptr<PythonFilter>>(m, "PythonFilter")
        .def(py::init<>())
        .def_readwrite("apply_cb", &PythonFilter::apply_cb)
        .def_readwrite("backprop_cb", &PythonFilter::backprop_cb)
        ;
    py::class_<ProjectionFilter, Filter, std::shared_ptr<ProjectionFilter>>(m, "ProjectionFilter")
        .def(py::init<>())
        .def_property("beta", &ProjectionFilter::getBeta, &ProjectionFilter::setBeta)
        ;
    py::class_<SmoothingFilter, Filter, std::shared_ptr<SmoothingFilter>>(m, "SmoothingFilter")
        .def(py::init<>())
        .def_property("radius", &SmoothingFilter::getRadius, &SmoothingFilter::setRadius)
        ;
    py::class_<LangelaarFilter, Filter, std::shared_ptr<LangelaarFilter>>(m, "LangelaarFilter")
        .def(py::init<>())
        ;
    m.def("applyFilter", [](Filter &filter, const Eigen::VectorXd &x) -> Eigen::VectorXd { // allow testing filters individually
        filter.checkGridDimensionsAreSet();
        NDVector<Real> in(filter.getGridDimensions()), out(filter.getGridDimensions());
        in.fill(x);
        filter.apply(in, out);
        return Eigen::Map<const Eigen::VectorXd>(out.data().data(), x.size());
    }, py::arg("filter"), py::arg("x"));

    // Constraints
    py::class_<Constraint, std::shared_ptr<Constraint>>(detail_module, "Constraint");
    py::class_<TotalVolumeConstraint, Constraint, std::shared_ptr<TotalVolumeConstraint>>(m, "TotalVolumeConstraint")
        .def(py::init<Real>(), py::arg("volumeFraction"))
        .def_readwrite("volumeFraction", &TotalVolumeConstraint::m_volumeFraction, py::return_value_policy::reference)
        ;
}
