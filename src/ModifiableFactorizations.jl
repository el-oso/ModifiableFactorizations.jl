module ModifiableFactorizations

using LinearAlgebra
using LinearAlgebra: givensAlgorithm, Givens, PosDefException, ZeroPivotException, QRCompactWY
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det
using TypeContracts
# `@strict` (type-stability + no owned-scratch dictionary lookups + allocation-freedom) guards
# the QR, Cholesky and LU rank-1 kernels below. Its checks run once per call site and argument
# signature, so a guarded verb allocates nothing when it runs. The guard's branch is still part
# of the enclosing verb's compiled body, and a static allocation proof reads every branch, so
# the verbs are proven allocation-free with checks disabled and the kernels with them enabled.
using StrictMode

export ModifiableCholesky, ModifiableLU, ModifiableQR
export insert_column!, try_insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!
export GrowCapacity, FixedCapacity
export qr_householder
export cholesky_crout, lu_crout, qr_bcgs
export cholesky_crout!, lu_crout!, qr_bcgs!
public AbstractQRep, DenseQ, materialize, capacity, CapacityPolicy
public AbstractModifiableCholesky, AbstractModifiableLU, AbstractModifiableQR
public default_rankk!

include("capacity.jl")
include("cholesky_type.jl")
include("cholesky_update.jl")
include("cholesky_resize.jl")
include("lu_type.jl")
include("lu_update.jl")
include("qr_rep.jl")
include("qr_type.jl")
include("qr_update.jl")
include("construct.jl")
include("contracts.jl")

# Contract checks, run at precompile time. Top-level `@verify` is not reachable from any entry
# point, so a trimmed binary does not contain it. One concrete instantiation per element type
# suffices because every verb is generic over the type parameters.
@verify DenseQ{Float64, Matrix{Float64}} trim_compat = true
@verify DenseQ{ComplexF64, Matrix{ComplexF64}} trim_compat = true
@verify ModifiableCholesky{Float64, Float64, Matrix{Float64}} trim_compat = true
@verify ModifiableCholesky{ComplexF64, Float64, Matrix{ComplexF64}} trim_compat = true
@verify ModifiableLU{Float64, Matrix{Float64}} trim_compat = true
@verify ModifiableLU{ComplexF64, Matrix{ComplexF64}} trim_compat = true
@verify ModifiableQR{Float64, Matrix{Float64}, DenseQ{Float64, Matrix{Float64}}} trim_compat = true
@verify ModifiableQR{ComplexF64, Matrix{ComplexF64}, DenseQ{ComplexF64, Matrix{ComplexF64}}} trim_compat = true

end
