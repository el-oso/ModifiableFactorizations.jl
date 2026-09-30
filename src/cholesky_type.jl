abstract type AbstractModifiableCholesky{T} <: Factorization{T} end

"""
    ModifiableCholesky(C::Cholesky; capacity = 2size(C, 1) + 1)
    ModifiableCholesky(A::AbstractMatrix; uplo = :L, capacity = 2size(A, 1) + 1)

Cholesky factorization that supports rank-1 update and downdate and symmetric insertion,
deletion and shifting of indices. `capacity` is the largest size the factorization can reach
before its storage is reallocated. The default is one column past `2size(A, 1)`: a column
count that is itself a multiple of a large power of two makes every row of a rank-1 update or
downdate map to the same cache set, and the `+1` avoids that without changing the asymptotic
storage cost.

The lower factor `L` of `A = L*L'` is what is stored, whichever triangle the input holds.
`F.L` is that factor, `F.U` its adjoint, and `Matrix(F)` the reconstructed `A`.
"""
mutable struct ModifiableCholesky{T, R <: Real, S <: AbstractMatrix{T}} <: AbstractModifiableCholesky{T}
    factors::S
    n::Int
    work::Vector{T}      # scratch: the update vector, consumed in place
    cosines::Vector{R}   # scratch: downdate rotation cosines
    rot::Vector{T}       # scratch: downdate rotation sines, and the reordered insertion vector
    perm::Vector{Int}    # scratch: the index permutation, consumed in place
end

# Wrap a capacity-sized buffer whose leading n x n block already holds the lower factor and
# whose untouched region is a true zero, allocating only the scratch vectors. Shared by every
# path that produces such a buffer directly, so the factor is copied into place once rather than
# built separately and copied in afterward.
function _wrap_cholesky(f::Matrix{T}, n::Int) where {T}
    R = real(T)
    cap = size(f, 1)
    return ModifiableCholesky{T, R, Matrix{T}}(
        f, n, zeros(T, cap), zeros(R, cap), zeros(T, cap), zeros(Int, cap)
    )
end

function ModifiableCholesky(C::Cholesky{T}; capacity::Int = 2size(C, 1) + 1) where {T}
    n = size(C, 1)
    capacity >= n || throw(ArgumentError("capacity $capacity is below the size $n"))
    f = zeros(T, capacity, capacity)
    # Cholesky.factors only guarantees the stored triangle; LAPACK leaves the factored matrix in
    # the other one. Copying the stored triangle alone keeps the unstored half a true zero,
    # which the resizing kernels rely on. An upper factor U is transposed on the way in, so the
    # storage always holds the lower factor L = U'.
    if C.uplo == 'L'
        for j in 1:n, i in j:n
            f[i, j] = C.factors[i, j]
        end
    else
        for j in 1:n, i in j:n
            f[i, j] = conj(C.factors[j, i])
        end
    end
    return _wrap_cholesky(f, n)
end

ModifiableCholesky(A::AbstractMatrix; uplo::Symbol = :L, capacity::Int = 2size(A, 1) + 1) =
    ModifiableCholesky(cholesky(Hermitian(A, uplo)); capacity)

# The active block of the stored lower factor.
_lower(F::ModifiableCholesky) = view(F.factors, 1:F.n, 1:F.n)

Base.size(F::ModifiableCholesky) = (F.n, F.n)
function Base.size(F::ModifiableCholesky, dim::Integer)
    dim < 1 && throw(ArgumentError("dimension must be positive, got $dim"))
    return dim <= 2 ? F.n : 1
end

function Base.getproperty(F::ModifiableCholesky, s::Symbol)
    s === :L && return LowerTriangular(_lower(F))
    s === :U && return UpperTriangular(adjoint(_lower(F)))
    return getfield(F, s)
end

Base.propertynames(::ModifiableCholesky, private::Bool = false) =
    private ? (:L, :U, fieldnames(ModifiableCholesky)...) : (:L, :U)

Base.AbstractMatrix(F::ModifiableCholesky) = (L = F.L; L * L')
Base.Matrix(F::ModifiableCholesky) = Matrix(AbstractMatrix(F))

"""
    capacity(F::ModifiableCholesky) -> Int

The largest size `F` can reach before its storage is reallocated.
"""
capacity(F::ModifiableCholesky) = size(F.factors, 1)

# Every operation either completes or throws with the factorization left as it was.
LinearAlgebra.issuccess(::ModifiableCholesky) = true

function LinearAlgebra.ldiv!(F::ModifiableCholesky, b::AbstractVecOrMat)
    L = LowerTriangular(_lower(F))
    ldiv!(L, b)
    ldiv!(L', b)
    return b
end

LinearAlgebra.logdet(F::ModifiableCholesky) =
    2 * sum(i -> log(real(F.factors[i, i])), 1:F.n; init = zero(real(eltype(F.factors))))
LinearAlgebra.det(F::ModifiableCholesky) = exp(logdet(F))
