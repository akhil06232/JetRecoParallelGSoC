#=
    run-benchmarks.jl

    Main entry point: launches Julia sub-processes with different thread counts,
    collects results, and produces a plot of distance-measures-per-second vs.
    thread count.

    Usage:
        julia run-benchmarks.jl

    This will test thread counts 1,2,4,6,8 (capped at your CPU thread count)
    and save a PNG plot to "benchmark_results.png".
=#

using Plots

function main()
    max_threads = Sys.CPU_THREADS
    thread_counts = sort(unique(filter(t -> t <= max_threads, [1, 2, 4, 6, 8, max_threads])))

    julia_cmd = Base.julia_cmd()
    script = joinpath(@__DIR__, "benchmark-worker.jl")

    results = Dict{Int, Float64}()  # nthreads => median_time_ns

    println("=" ^ 60)
    println("Pairwise Euclidean Distance Benchmark")
    println("CPU threads available: $(max_threads)")
    println("Thread counts to test: $(thread_counts)")
    println("=" ^ 60)

    for nt in thread_counts
        println("\nRunning benchmark with $(nt) thread(s)...")
        output = read(`$julia_cmd -t $nt $script`, String)
        parts = split(strip(output), ",")
        nthreads = parse(Int, parts[1])
        med_ns = parse(Float64, parts[2])
        n_points = parse(Int, parts[3])

        total_pairs = n_points * n_points
        med_s = med_ns / 1e9
        dps = total_pairs / med_s

        results[nthreads] = dps
        println("  Threads: $(nthreads), Median time: $(round(med_s, digits=4)) s, " *
                "Distance measures/s: $(round(dps, sigdigits=4))")
    end

    # Sort and plot
    sorted_threads = sort(collect(keys(results)))
    dps_values = [results[t] for t in sorted_threads]

    serial_dps = results[1]
    speedups = dps_values ./ serial_dps

    println("\n" * "=" ^ 60)
    println("Summary")
    println("=" ^ 60)
    println("Threads | Dist. measures/s  | Speedup")
    println("-" ^ 50)
    for (t, d, s) in zip(sorted_threads, dps_values, speedups)
        println("  $(lpad(t, 3))   | $(lpad(round(d, sigdigits=4), 17)) | $(round(s, digits=2))x")
    end

    # Plot
    p = plot(sorted_threads, dps_values ./ 1e6,
        xlabel="Thread count",
        ylabel="Distance measures per second (millions)",
        title="Pairwise Euclidean Distance: Throughput vs. Threads",
        marker=:circle,
        markersize=6,
        linewidth=2,
        legend=false,
        grid=true,
        xticks=sorted_threads,
        size=(800, 500),
        dpi=150
    )

    outfile = joinpath(@__DIR__, "benchmark_results.png")
    savefig(p, outfile)
    println("\nPlot saved to: $(outfile)")

    # Also make a speedup plot
    p2 = plot(sorted_threads, speedups,
        xlabel="Thread count",
        ylabel="Speedup (relative to 1 thread)",
        title="Pairwise Euclidean Distance: Speedup vs. Threads",
        marker=:square,
        markersize=6,
        linewidth=2,
        label="Measured speedup",
        grid=true,
        xticks=sorted_threads,
        size=(800, 500),
        dpi=150
    )
    plot!(p2, sorted_threads, sorted_threads,
        linestyle=:dash, label="Ideal (linear) speedup", color=:gray)

    outfile2 = joinpath(@__DIR__, "speedup_results.png")
    savefig(p2, outfile2)
    println("Speedup plot saved to: $(outfile2)")
end

main()
