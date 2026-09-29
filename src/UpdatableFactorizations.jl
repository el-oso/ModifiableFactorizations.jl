module UpdatableFactorizations

using LinearAlgebra
using LinearAlgebra: givensAlgorithm, Givens, PosDefException, ZeroPivotException, QRCompactWY
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det
using TypeContracts
# `@strict` (type-stability + no owned-scratch dictionary lookups + allocation-freedom) guards
# the QR, Cholesky and LU rank-1 kernels below. A guarded call is itself type stable and
# allocation-free, so the verbs that host one keep both guarantees with checks enabled.
using StrictMode

export UpdatableCholesky, UpdatableLU, UpdatableQR
export insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!
export qr_householder
export cholesky_crout, lu_crout, qr_bcgs
export cholesky_crout!, lu_crout!, qr_bcgs!
public AbstractQRep, DenseQ, materialize, capacity
public AbstractUpdatableCholesky, AbstractUpdatableLU, AbstractUpdatableQR
public default_rankk!

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

end
