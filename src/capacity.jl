"""
    CapacityPolicy

What an inserting verb does when the insertion needs more storage than the factorization holds.
[`GrowCapacity`](@ref) reallocates and [`FixedCapacity`](@ref) throws. The verbs that insert take
one as their `capacity_policy` keyword.
"""
abstract type CapacityPolicy end

"""
    GrowCapacity()

Reallocate the factorization's storage when an insertion exceeds its capacity, at least doubling
the dimension that ran out. This is the default `capacity_policy`.
"""
struct GrowCapacity <: CapacityPolicy end

"""
    FixedCapacity()

Throw `ArgumentError` when an insertion would exceed the factorization's capacity. The check comes
before anything is modified, so the factorization is unchanged when it throws.

An inserting verb called with this policy contains no path that reallocates, so its freedom from
allocation holds for its argument types rather than only for the values a caller happens to pass:
a static proof such as `StrictModeTest.@test_noalloc` can establish it.
"""
struct FixedCapacity <: CapacityPolicy end
