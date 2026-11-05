@info "Starting PDB variances script."

using Revise
using BioStructures
using LinearAlgebra
using Random
using PointCloudRegistration
using GLMakie
using Statistics
using Distances

# We start with a collection of IDs from the Protein Data Base that identify
# certain DNA polymerase conformers, found by a 90 % sequence similarity search
# (base sequence has PDB ID 1ih7).
#
# (See PDB query in file query-for-1ih7-90percent-sim.json.)

@info "Loading PDB IDs..."

all_ids = split(read("1ih7-90percent-sim.txt", String), ",") .|> chomp

# We download the PDB files via `retrievepdb` from BioStructures.jl and take
# the longest chain of each of them.

@info "Loading PDB data..."

all_chains = map(all_ids) do id
    entry_id = chopsuffix(id, r"_.*")
    pdb_entry = retrievepdb(entry_id; dir = "data")
    the_chains = pdb_entry |> chains |> values
    argmax(countresidues, the_chains)
end

# To make our life simpler and avoid having to perform sequence alignments,
# we restrict ourselves to structures with exactly `residuecount` residues, or
# in other words, `residuecount` Cα-atoms.

@info "Extract chains..."

residuecount = 903
exact_seq_chains = filter(all_chains) do chain
    countatoms(chain, calphaselector) == residuecount
end

# For further analysis, we produce a point cloud of the Cα coordinates for each
# structure.

@info "Make point clouds..."

pointclouds = [
    PointCloud(coordarray(chain, calphaselector)) for chain in exact_seq_chains
]

# Next, we (arbitrarily) pick the first point cloud as the target for rigid
# registration.

target = pointclouds[1]

# We define the algorithms we want to compare...

algorithms = (
    kabsch = Kabsch(),
    gmc = GemanMcClureMM(),
)

# ... and then run the registration.

@info "Perform rigid registration..."

pointclouds_registered = map(algorithms) do algorithm
    [rigidly_registered(source, target, algorithm) for source in pointclouds]
end;

# We quantify the result of the registration by measuring the variance of each
# point across the whole registered ensemble.
# To highlight differences better, we compute the logarithm of the variance.

function varperpos(pcs)
    len = only(unique(length.(pcs)))
    map(1:len) do i
        m = mean(pc -> points(pc)[i], pcs)
        v = mean(pc -> sqeuclidean(points(pc)[i], m), pcs)
        # return log(v)
        return v
    end
end

# Finally, we produce the plot with the algorithms compared side by side.

@info "Creating figure..."

let
    fig = Figure(size = (600, 300))
    Label(fig[0, 1], "Kabsch"; tellwidth = false, font = :bold)
    Label(fig[0, 2], "Geman-McClure"; tellwidth = false, font = :bold)
    axs = (
        kabsch = Axis3(fig[1, 1]),
        gmc = Axis3(fig[1, 2]),
    )
    for ax in axs
        ax.aspect = :data
        ax.viewmode = :fit
        ax.protrusions = (0, 0, 0, 10)
        ax.alignmode = Outside(-50)
        ax.elevation = deg2rad(-15)
        ax.azimuth = deg2rad(-10)
        hidedecorations!(ax)
        hidespines!(ax)
    end
    vs = map(varperpos, pointclouds_registered)
    minv = minimum(minimum, vs)
    maxv = maximum(maximum, vs)
    colorrange = (minv, maxv)
    colormap = :coolwarm
    for k in keys(axs)
        plot!(
            axs[k],
            target;
            sizefactor = 2.0,
            color = vs[k],
            colorrange,
            colormap,
            colorscale = log10,
        )
    end
    Colorbar(
        fig[0:1, 3];
        colorrange,
        colormap,
        scale = log10,
        label = "variance"
    )
    fig
    save("pdb-variances.png", fig)
end

try
    @info "Trying to minimize figure..."
    run(`oxipng pdb-variances.png`)
catch e
    if e isa IOError
        @warn "`oxipng` probably not available, figure will not be minimized."
    end
end

@info "Done."
