@info "Starting cryo-EM densities script."

using PointCloudRegistration
using GLMakie
using MRCFile
using DimensionalData
using Rotations
using StaticArrays
using CoordinateTransformations
using Random
using Downloads

Random.seed!(3)

function load_density(id)
    filename = joinpath("data", "emd_$id.map.gz")
    if !isdir(dirname(filename))
        mkdir(dirname(filename))
    end
    if !isfile(filename)
        url =
            "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/" *
            "EMD-$id/map/$(basename(filename))"
        @info "downloading from EMDB..." url filename
        Downloads.download(url, filename)
    end
    mrc = read(filename, MRCData)
    return DimArray(mrc)
end

emdb_ids = (open = "74466", closed = "74465")

@info "Loading densities..."

densities = map(load_density, emdb_ids)

@info "Conversion to point clouds..."

pcs_full = map(density2pointcloud, densities)

@info "Thresholding..."

thresholds = (open = 0.16f0, closed = 0.22f0)
pcs_thresholded = map(drop_threshold, pcs_full, thresholds)

@info "Thinning..."

pcs_thinned = map(pc -> thin_to_distance(pc, 10.0), pcs_thresholded)

@info "Randomizing pose..."

open_rand_pose = let
    ctr, _ = PointCloudRegistration.mean_cov(pcs_thinned.open)
    rot = rand(RotMatrix3{Float32})
    compose(
        Translation(SA[0f0, 150f0, 0f0]),
        Translation(ctr),
        LinearMap(rot),
        inv(Translation(ctr))
    )
end

pcs = (; open = open_rand_pose(pcs_thinned.open), pcs_thinned.closed)

@info "Performing rigid registration..."

rr_open = rigidly_registered(pcs.open, pcs.closed)

rr_pcs = (open = rr_open, closed = pcs.closed)

@info "Performing non-rigid registration..."

dp_prep = prepare_source_distancepreserving(rr_pcs.open; max_edge_length = 25);

# nsph = []
displacement = nonrigid_registration(
    rr_pcs.open,
    rr_pcs.closed,
    DistancePreserving(
        sensitivity = 2.0,
        rel_deviation = 0.01,
        # report_iteration = function(; iteration, new_source_points, sqsigma, kwargs...)
        #     iteration % 1000 == 1 || return
        #     @info "iteration" iteration sqrt(sqsigma)
        #     push!(nsph, copy(new_source_points))
        # end,
    );
    source_preparation = dp_prep,
)

# let
#     fig = Figure()
#     ax = Axis3(fig[1, 1]; aspect = :data)
#     sl = Slider(fig[2, 1]; range = eachindex(nsph))
#     nsrc = map(sl.value) do idx
#         PointCloud(nsph[idx], weights(rr_pcs.open))
#     end
#     plot!(ax, rr_pcs.closed; alpha = 0.1)
#     plot!(ax, dp_prep, nsrc; radius = 0.5, colormap = :coolwarm, colorrange = (0, log1p(0.5)))
#     fig
# end

@info "Creating figure..."

let
    clrs = (open = colorant"#0074D9", closed = colorant"#FF4136")
    p = open_rand_pose
    q = QuatRotation(p.linear)
    fig = Figure()
    ax_dns = Axis3(fig[1, 1])
    ax_thr = Axis3(fig[1, 2])
    ax_rr = Axis3(fig[2, 1])
    ax_nrr = Axis3(fig[2, 2])
    for ax in [ax_dns, ax_thr, ax_rr, ax_nrr]
        ax.aspect = :data
        ax.azimuth = deg2rad(170)
        ax.elevation = deg2rad(15)
        ax.protrusions = 0
        ax.viewmode = :fit
        ax.alignmode = Outside(-70)
        hidedecorations!(ax)
        hidespines!(ax)
    end
    wm(lo, hi) = (d = hi - lo; (lo - 0.1d, hi + 0.1d))
    (xlo, ylo, zlo), (xhi, yhi, zhi) = PointCloudRegistration.bbox(
        PointCloud(vcat(pcs.open, pcs.closed))
    )
    for ax in [ax_dns, ax_thr]
        ax.aspect = :data
        xlims!(ax, wm(xlo, xhi))
        ylims!(ax, wm(ylo, yhi))
        zlims!(ax, wm(zlo, zhi))
    end
    (xlo, ylo, zlo), (xhi, yhi, zhi) = PointCloudRegistration.bbox(
        PointCloud(vcat(rr_pcs.open, rr_pcs.closed))
    )
    for ax in [ax_rr, ax_nrr]
        ax.aspect = :data
        xlims!(ax, wm(xlo, xhi))
        ylims!(ax, wm(ylo, yhi))
        zlims!(ax, wm(zlo, zhi))
    end
    for k in (:open, :closed)
        thr = thresholds[k]
        plt = Makie.volume!(ax_dns, densities[k]; algorithm = :iso, isovalue = 1.5thr, isorange = 0.5thr, colormap = [clrs[k]])
        if k == :open
            Makie.rotate!(Accum, plt, q.x, q.y, q.z, q.w)
            Makie.translate!(Accum, plt, p.translation)
        end
    end
    for k in (:open, :closed)
        plt = Makie.plot!(ax_thr, pcs[k]; color = clrs[k])
    end
    for k in (:open, :closed)
        plt = Makie.plot!(ax_rr, rr_pcs[k]; color = clrs[k])
    end
    arrows3d!(ax_nrr, displacement.origin, displacement.result .- displacement.origin)
    for (letter, ci) in zip('A':'D', [(1, 1), (1, 2), (2, 1), (2, 2)])
        Label(fig[ci...], "($letter)"; font = :bold, tellwidth = false, tellheight = false, halign = :left, valign = :top)
    end
    fig
    save("cryo-em-densities.png", fig)
end

try
    @info "Trying to minimize figure..."
    run(`oxipng cryo-em-densities.png`)
catch e
    if e isa IOError
        @warn "`oxipng` probably not available, figure will not be minimized."
    end
end

@info "Done."
