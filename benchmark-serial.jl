using BenchmarkTools

"""
    pairwise_distances(points::AbstractArray)

Calculate the pairwise distances between 3D points in an Nx3 array
"""
function pairwise_distances(points::AbstractArray{T}) where T
    @assert size(points)[2] == 3
    n = size(points)[1]
    distances = zeros(T, (n, n))
    for i in 1:n
        for j in 1:n
            dx = points[i, 1] - points[j, 1]
            dy = points[i, 2] - points[j, 2]
            dz = points[i, 3] - points[j, 3]
            distances[i, j] = sqrt(dx^2 + dy^2 + dz^2)
        end
    end
    return distances
end

function main()
    points = rand(Float32, (10_000, 3))

    # Warm-up: trigger JIT compilation on a small input
    println("Warm-up run (small problem)...")
    small_points = rand(Float32, (100, 3))
    pairwise_distances(small_points)

    # Benchmark the full problem
    println("\nBenchmarking pairwise_distances with N=10,000 points...")
    b = @benchmark pairwise_distances($points) samples=5 evals=1

    display(b)
    println()

    n = size(points, 1)
    total_pairs = n * n
    median_time_s = median(b).time / 1e9
    println("Total distance calculations: $(total_pairs)")
    println("Median time: $(round(median_time_s, digits=4)) s")
    println("Distance measures per second: $(round(total_pairs / median_time_s, sigdigits=4))")
end

main()
