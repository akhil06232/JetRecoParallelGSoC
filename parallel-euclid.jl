"""
Parallel Euclidean Distance Computation - Benchmark Suite

This script benchmarks serial and parallel implementations of pairwise
Euclidean distance calculation, sweeping over thread counts and producing
a performance plot.

## How to run:

    # Serial benchmark only:
    julia benchmark-serial.jl

    # Full parallel benchmark with plot (launches sub-processes with varying thread counts):
    julia run-benchmarks.jl

    # Or run the parallel version directly with a specific thread count:
    julia -t 4 parallel-euclid.jl
"""

using Base.Threads

"""
    pairwise_distances_serial!(distances, points)

Serial pairwise distance computation. Writes results into pre-allocated `distances`.
"""
function pairwise_distances_serial!(distances::AbstractMatrix{T}, points::AbstractMatrix{T}) where T
    n = size(points, 1)
    @inbounds for i in 1:n
        xi = points[i, 1]
        yi = points[i, 2]
        zi = points[i, 3]
        for j in 1:n
            dx = xi - points[j, 1]
            dy = yi - points[j, 2]
            dz = zi - points[j, 3]
            distances[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
        end
    end
    return distances
end

"""
    pairwise_distances_parallel!(distances, points)

Threaded pairwise distance computation. Parallelises over the outer loop (rows).
Each thread writes to disjoint rows of `distances`, so no synchronisation is needed.
"""
function pairwise_distances_parallel!(distances::AbstractMatrix{T}, points::AbstractMatrix{T}) where T
    n = size(points, 1)
    @threads for i in 1:n
        @inbounds begin
            xi = points[i, 1]
            yi = points[i, 2]
            zi = points[i, 3]
            for j in 1:n
                dx = xi - points[j, 1]
                dy = yi - points[j, 2]
                dz = zi - points[j, 3]
                distances[i, j] = sqrt(dx*dx + dy*dy + dz*dz)
            end
        end
    end
    return distances
end
