@info "Starting Kabsch benchmark script."

using GLMakie
using StaticArrays
import PointCloudRegistration as PCReg
import CoordinateTransformations as CT
import BioStructures as BioS
import Kabsch as K
import BiochemicalAlgorithms as BA

dim = 3
len = 1000
num = 10_000

@info "Setting up mock data..." dim len num

mat_target = rand(dim, len);
mat_sources = [rand(dim, len) for _ in 1:num];

@info "Converting data to package specific types..."

# This is friendly to BiochemicalAlgorithms.jl in two ways:
# First, we have an individual system per molecule.
# Otherwise, the whole system of all molecules would have to be traversed for
# every registration of two molecules.
# Second, we already represent a point cloud as an `AtomTable` which seems to be
# more of an implementation detail than a user facing interface but its
# construction takes a significant time otherwise.
function mat2mol(mat)
    sys = BA.System{Float64}()
    mol = BA.Molecule(sys)
    for (i, point) in enumerate(eachcol(mat))
        BA.Atom(mol, i, BA.Elements.C; r = SVector{3}(point))
    end
    return BA.atoms(mol)
end

ba_target = mat2mol(mat_target)
ba_sources = [mat2mol(mat_source) for mat_source in mat_sources]

pc_target = PCReg.PointCloud(mat_target)
pc_sources = [PCReg.PointCloud(mat_source) for mat_source in mat_sources]

@info "Defining measurement functions..."

function time_ct(target, sources)
    return @timed for source in sources
        CT.kabsch(source => target)
    end
end
function time_k(target, sources)
    return @timed for source in sources
        K.kabsch(source, target)
    end
end
function time_bios(target, sources)
    return @timed for source in sources
        BioS.Transformation(source, target)
    end
end
function time_ba(target, sources)
    return @timed for source in sources
        BA.compute_rmsd_minimizer(source, target; minimizer = BA.RMSDMinimizerKabsch)
    end
end
function time_pcreg(target, sources)
    return @timed for source in sources
        PCReg.rigid_registration(source, target, PCReg.Kabsch())
    end
end

run_measurements() = (
    ct = time_ct(mat_target, mat_sources),
    k = time_k(mat_target, mat_sources),
    bios = time_bios(mat_target, mat_sources),
    ba = time_ba(ba_target, ba_sources),
    pcreg_mat = time_pcreg(mat_target, mat_sources),
    pcreg_pc = time_pcreg(pc_target, pc_sources),
)

@info "Performing warm up..."

run_measurements() # warm up

@info "Running benchmarks..."

timings = run_measurements()

@info "Creating plot..."

labels = (
    pcreg_mat = rich("PointCloudRegistration.jl"; font = :bold),
    pcreg_pc = rich(
        rich("PointCloudRegistration.jl"; font = :bold),
        " with PointCloud type",
    ),
    ct = "CoordinateTransformations.jl",
    bios = "BioStructures.jl",
    k = "Kabsch.jl",
    ba = "BiochemicalAlgorithms.jl",
)

let
    fig = Figure(; size = (600, 300))
    kys = collect(keys(timings))
    # kys = filter(!=(:pcreg_pc), collect(keys(timings)))
    sort!(kys; by = key -> timings[key].time)
    ax = Axis(
        fig[1, 1];
        yticks = (eachindex(kys), [labels[k] for k in kys]),
        xtickformat = "{:.1f} s",
        xtickcolor = RGBAf(0, 0, 0, 0.12),
        xgridcolor = RGBAf(0, 0, 0, 0.12),
        # title = "Kabsch registration of $num point clouds in $dim dimensions with $len points each",
    )
    hideydecorations!(ax)
    hidespines!(ax)
    all_times = Float64[]
    all_categories = Int[]
    all_groups = Int[]
    for (i, key) in enumerate(kys)
        (; time, gctime) = timings[key]
        append!(all_times, [gctime, time - gctime])
        append!(all_categories, [i, i])
        append!(all_groups, [1, 2])
    end
    gc_color = colorant"#39cccc"
    rest_color = colorant"#7fdbff"
    barplot!(
        ax,
        all_categories,
        all_times;
        direction = :x,
        stack = all_groups,
        color = all_groups,
        colormap = [gc_color, rest_color],
    )
    for (i, key) in enumerate(kys)
        text!(
            ax,
            Point(0, i);
            text = rich(
                labels[key],
                " ",
                rich(
                    "(",
                    string(round(Int, 1000 * timings[key].time)),
                    " ms)";
                    color = colorant"#444",
                ),
            ),
            align = (:left, :center),
            offset = (10, 0),
        )
    end
    legend_gc_prop = 0.3
    Legend(
        fig[1, 1],
        [
            [
                PolyElement(
                    color = gc_color,
                    strokewidth = 0,
                    points = Point2f[(0, 0), (legend_gc_prop, 0), (legend_gc_prop, 1), (0, 1)]
                ),
                PolyElement(
                    color = rest_color,
                    strokewidth = 0,
                    points = Point2f[(legend_gc_prop, 0), (1, 0), (1, 1), (legend_gc_prop, 1)]
                ),
            ],
            PolyElement(
                color = gc_color,
                strokewidth = 0,
                points = Point2f[(0, 0), (legend_gc_prop, 0), (legend_gc_prop, 1), (0, 1)]
            ),
        ],
        ["total time", "GC time"];
        patchsize = (45, 15),
        halign = :right,
        valign = :bottom,
        tellwidth = false,
        tellheight = false,
    )
    fig
    save("kabsch-benchmark.png", fig)
end

try
    @info "Trying to minimize figure..."
    run(`oxipng kabsch-benchmark.png`)
catch e
    if e isa IOError
        @warn "`oxipng` probably not available, figure will not be minimized."
    end
end

@info "Done."
