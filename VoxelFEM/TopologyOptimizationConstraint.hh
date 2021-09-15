#ifndef MESHFEM_TOPOLOGYOPTIMIZATIONCONSTRAINT_HH
#define MESHFEM_TOPOLOGYOPTIMIZATIONCONSTRAINT_HH

#include "NDVector.hh"

struct Constraint {

    virtual ~Constraint() {};

    /// Evaluation of volume constraint.
    /// @param[in] vars physical variables
    /// @param[out] out constraint value
    virtual Real evaluate(const NDVector<Real> &vars) const = 0;

    /// Derivative of constraint w.r.t. physical variables.
    /// @param[in] vars physical variables
    /// @param[out] out constraint derivatives
    virtual void backprop(const NDVector<Real> &vars, NDVector<Real> &out) const = 0;
};

struct TotalVolumeConstraint : public Constraint {

    TotalVolumeConstraint(Real volume): m_volumeFraction(volume) { }

    Real evaluate(const NDVector<Real> &vars) const override {
        return 1.0 - (vars.flattened().mean()) / m_volumeFraction;
    }

    void backprop(const NDVector<Real> &vars, NDVector<Real> &out) const override {
        out.fill(-1.0 / (m_volumeFraction * vars.Size()));
    }

    // A value in [0, 1] indicating the fraction of solid voxels
    Real m_volumeFraction;
};

#endif // MESHFEM_TOPOLOGYOPTIMIZATIONCONSTRAINT_HH
