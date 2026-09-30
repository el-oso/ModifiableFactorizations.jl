@testitem "Cholesky symmetric deletion, every index" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        n = 7
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        for j in 1:n
            F = ModifiableCholesky(cholesky(Hermitian(A, uplo)))
            delete_column!(F, j)
            keep = [k for k in 1:n if k != j]
            L = F.L
            @test size(F) == (n - 1, n - 1)
            @test norm(L * L' - A[keep, keep]) / norm(A) < 1.0e-13
        end
    end
end

@testitem "Cholesky deletion rejects an out-of-range index" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    B = randn(5, 5)
    F = ModifiableCholesky(cholesky(Symmetric(Matrix(Symmetric(B * B' + 5I)), :L)))
    @test_throws BoundsError delete_column!(F, 6)
    @test_throws BoundsError delete_column!(F, 0)
end

@testitem "Cholesky append at the end" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        n = 6
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        F = ModifiableCholesky(cholesky(Hermitian(A[1:n, 1:n], uplo)))
        ModifiableFactorizations._append!(F, A[:, n + 1])
        L = F.L
        @test size(F) == (n + 1, n + 1)
        @test norm(L * L' - A) / norm(A) < 1.0e-13
    end
end

@testitem "Cholesky append grows past capacity" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 4
    B = randn(n + 1, n + 1)
    A = Matrix(Symmetric(B * B' + (n + 1) * I))
    F = ModifiableCholesky(cholesky(Symmetric(A[1:n, 1:n], :L)); capacity = n)
    ModifiableFactorizations._append!(F, A[:, n + 1])
    L = F.L
    @test norm(L * L' - A) / norm(A) < 1.0e-13
end

@testitem "Cholesky index shift, both directions" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 7
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        for i in 1:n, j in 1:n
            F = ModifiableCholesky(cholesky(Hermitian(A, uplo)))
            shift_columns!(F, i, j)
            p = ModifiableFactorizations._cyclicperm!(zeros(Int, n), i, j)
            L = F.L
            @test norm(L * L' - A[p, p]) / norm(A) < 1.0e-12
            @test all(x -> abs(imag(x)) < 1.0e-12 && real(x) > 0, diag(L))
            @test isfinite(logdet(F)) && isreal(logdet(F))
        end
    end
end

@testitem "Cholesky symmetric insertion, every index" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 6
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        for j in 1:(n + 1)
            keep = [k for k in 1:(n + 1) if k != j]
            F = ModifiableCholesky(cholesky(Hermitian(A[keep, keep], uplo)))
            insert_column!(F, j, A[:, j])
            L = F.L
            @test size(F) == (n + 1, n + 1)
            @test norm(L * L' - A) / norm(A) < 1.0e-12
            @test all(x -> abs(imag(x)) < 1.0e-12 && real(x) > 0, diag(L))
            @test isfinite(logdet(F)) && isreal(logdet(F))
        end
    end
end

@testitem "Cholesky insertion under FixedCapacity throws past capacity and leaves F unchanged" begin
    using LinearAlgebra, Random
    Random.seed!(20260930)
    n = 5
    for T in (Float64, ComplexF64)
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        col = A[:, 3]
        keep = [1, 2, 4, 5, 6]

        F = ModifiableCholesky(cholesky(Hermitian(A[keep, keep], :L)); capacity = n)
        L0 = copy(F.L)
        factors0 = copy(F.factors)
        @test_throws "capacity policy is FixedCapacity()" insert_column!(
            F, 3, col; capacity_policy = FixedCapacity()
        )
        @test size(F) == (n, n)
        @test ModifiableFactorizations.capacity(F) == n
        @test F.L == L0
        @test F.factors == factors0

        # Within capacity, FixedCapacity gives exactly what GrowCapacity does.
        G = ModifiableCholesky(cholesky(Hermitian(A[keep, keep], :L)); capacity = n + 1)
        H = ModifiableCholesky(cholesky(Hermitian(A[keep, keep], :L)); capacity = n + 1)
        insert_column!(G, 3, col; capacity_policy = FixedCapacity())
        insert_column!(H, 3, col)
        @test G.L == H.L
        @test norm(G.L * G.L' - A) / norm(A) < 1.0e-13
        @test ModifiableFactorizations.capacity(G) == n + 1
    end
end
