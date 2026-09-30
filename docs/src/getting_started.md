# Getting started

```julia
using Pkg
Pkg.add("ModifiableFactorizations")
```

## Cholesky

`ModifiableCholesky` factors a Hermitian positive definite matrix. It can be built from a plain
matrix or from an existing `LinearAlgebra.Cholesky`; both input orientations, `uplo = :L` and
`uplo = :U`, give identical results because the package always stores the lower factor.

```@example cholesky
using LinearAlgebra, ModifiableFactorizations

A = [4.0 2.0; 2.0 3.0]
F = ModifiableCholesky(A)
F.L
```

`F.U` is the adjoint of `F.L`, and `Matrix(F)` reconstructs `A`:

```@example cholesky
F.U
```

```@example cholesky
Matrix(F) ≈ A
```

`F` supports the standard `Factorization` interface — `\`, `ldiv!`, `size`, `det`, `logdet`:

```@example cholesky
b = [1.0, 2.0]
F \ b
```

## LU

`ModifiableLU` factors a general square matrix, with partial pivoting by default. It is stored as
`P*A = L*Diagonal(d)*U`, but `F.L` and `F.U` return the factors in the same form
`LinearAlgebra.lu` gives them, so the two are interchangeable for reading off the factorization.

```@example lu
using LinearAlgebra, ModifiableFactorizations

B = [4.0 3.0; 6.0 3.0]
G = ModifiableLU(B)
G.L
```

```@example lu
G.U
```

```@example lu
Matrix(G) ≈ B
```

`G.info` and `issuccess(G)` report whether the factorization is still valid; see
[Updating and downdating](@ref) for what invalidates it.

## QR

`ModifiableQR` factors an `m` by `n` matrix with `m >= n` as `Q*R`, holding the thin `Q` (`m` by
`n`, orthonormal columns) and the upper triangular `R`.

```@example qr
using LinearAlgebra, ModifiableFactorizations

C = [1.0 1.0; 1.0 2.0; 1.0 3.0]
H = ModifiableQR(C)
H.R
```

`H.Q` has orthonormal columns and `Matrix(H)` reconstructs `C`:

```@example qr
H.Q' * H.Q ≈ I
```

```@example qr
Matrix(H) ≈ C
```

Because the factorization is thin and `m` may exceed `n`, `\` solves the least-squares problem
rather than a square system:

```@example qr
y = [1.0, 2.0, 2.0]
H \ y
```

`capacity` is a tuple here, since a QR factorization grows in both directions —
`insert_row!` adds an observation and `insert_column!` adds a variable:

```@example qr
H2 = ModifiableQR(C; capacity = (10, 4))
ModifiableFactorizations.capacity(H2)
```

`capacity` is public but not exported, so it is reached through the module name.

## Capacity

`ModifiableCholesky` and `ModifiableQR` can grow: `insert_column!` extends the factored matrix by
one index. The `capacity` keyword sets how large the factorization can grow before its storage is
reallocated — an integer for `ModifiableCholesky`, a `(rows, columns)` tuple for `ModifiableQR`:

```@example cholesky
F2 = ModifiableCholesky(A; capacity = 10)
size(F2)
```

`ModifiableLU` has no resizing operation and so takes no `capacity` argument.
