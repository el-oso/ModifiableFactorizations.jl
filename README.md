# ModifiableFactorizations.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/ModifiableFactorizations.jl/dev/)
[![CI](https://github.com/el-oso/ModifiableFactorizations.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/el-oso/ModifiableFactorizations.jl/actions/workflows/CI.yml)
[![Coverage Status](https://coveralls.io/repos/github/el-oso/ModifiableFactorizations.jl/badge.svg?branch=master)](https://coveralls.io/github/el-oso/ModifiableFactorizations.jl?branch=master)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Maintains Cholesky, LU, and QR factorizations under rank-1 update and
downdate, and under row/column insertion and deletion, without recomputing
the factorization from scratch. The name follows Gill, Golub, Murray and
Saunders, *Methods for modifying matrix factorizations*, Mathematics of
Computation 28 (1974), 505-535, the source of this package's Cholesky rank-1
update.

| Type | Operations |
|---|---|
| `ModifiableCholesky` | `lowrankupdate!`, `lowrankdowndate!`, `insert_column!`, `delete_column!`, `shift_columns!` |
| `ModifiableLU` | `lowrankupdate!` |
| `ModifiableQR` | `lowrankupdate!`, `insert_column!`, `try_insert_column!`, `delete_column!`, `shift_columns!`, `insert_row!`, `delete_row!` |

The package extends `LinearAlgebra.lowrankupdate!` and
`LinearAlgebra.lowrankdowndate!` with methods for its own factorization
types, rather than replacing the existing `Cholesky` methods those functions
already provide.

## Installation

```julia
using Pkg
Pkg.add("ModifiableFactorizations")
```

## Example

```julia
using LinearAlgebra, ModifiableFactorizations

A = randn(6, 3)
F = ModifiableQR(A)

u, v = randn(6), randn(3)
lowrankupdate!(F, u, v)        # F now factors A + u*v'

x = randn(6)
insert_column!(F, 2, x)        # insert x as column 2
delete_column!(F, 2)           # and remove it again

Matrix(F) ≈ A + u * v'         # true
```

See the [documentation](https://el-oso.github.io/ModifiableFactorizations.jl/dev/)
for the other factorizations, the numerical tolerances each operation takes,
and what makes an update fail.

## Relation to similarly named packages

[UpdatableQRFactorizations.jl](https://github.com/SebastianAment/UpdatableQRFactorizations.jl)
and
[UpdatableCholeskyFactorizations.jl](https://github.com/SebastianAment/UpdatableCholeskyFactorizations.jl)
are separate packages with overlapping scope. Their exported names do not
overlap with this package's, so either can be loaded alongside it.

- UpdatableQRFactorizations.jl adds and removes columns of a QR factorization,
  storing `Q` as a sequence of Givens rotations, which keeps memory low when
  the matrix has many rows. This package stores `Q` explicitly, and also
  provides rank-1 updates and row insertion and deletion.
- UpdatableCholeskyFactorizations.jl appends and removes rows and columns of a
  Cholesky factorization. This package also provides rank-1 update and
  downdate of it, and a modifiable LU factorization.

Measured comparisons against both, against `LinearAlgebra`, `QRupdate.jl`,
the `qrupdate-ng` wrapper `QRupdatesFast.jl`, and against recomputing the
factorization from scratch, are on the
[benchmarks page](https://el-oso.github.io/ModifiableFactorizations.jl/dev/benchmarks).
Building a factorization from scratch goes through `LinearAlgebra`; this
package does not offer a faster way to do that.

Every routine's provenance, the paper it derives from and the reference
implementations consulted during development, is recorded on the
[provenance page](https://el-oso.github.io/ModifiableFactorizations.jl/dev/provenance).

## Use of AI assistance

This package was developed with substantial assistance from Claude Code, using
Anthropic's Claude Opus 5, Claude Opus 5.5 and Claude Sonnet 5 models. Most of
the implementation, tests, documentation and benchmarks were drafted with it,
and 122 of the first 123 commits name the model in an `Assisted-by` or
`Co-Authored-By` trailer. I have read all of the code and I am responsible
for it.

## License

MIT. See [LICENSE](LICENSE).
