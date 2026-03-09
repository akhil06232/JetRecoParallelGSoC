#=
    benchmark-worker.jl

    Called by run-benchmarks.jl as:
        julia -t <nthreads> benchmark-worker.jl <nthreads>

    Prints a single CSV line: nthreads,median_time_ns,n_points
=#

using BenchmarkTools

include("parallel-euclid.jl")

function main()
    N = 10_000
    nthreads = Threads.nthreads()

    points = rand(Float32, (N, 3))
    distances = zeros(Float32, (N, N))

    # Warm-up: run once to trigger JIT compilation
    if nthreads == 1
        pairwise_distances_serial!(distances, points)
    else
        pairwise_distances_parallel!(distances, points)
    end

    # Benchmark
    if nthreads == 1
        b = @benchmark pairwise_distances_serial!($distances, $points) samples=5 evals=1
    else
        b = @benchmark pairwise_distances_parallel!($distances, $points) samples=5 evals=1
    end

    med_ns = median(b).time  # in nanoseconds
    println("$(nthreads),$(med_ns),$(N)")
end

main()
