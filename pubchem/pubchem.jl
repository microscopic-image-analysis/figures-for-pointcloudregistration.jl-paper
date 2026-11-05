@info "Starting PubChem script."

using PointCloudRegistration
using GLMakie
using Format
using GeometryBasics
using PubChemCrawler
using JSON
using StaticArrays
using LinearAlgebra
using Distances
using Random
using Statistics

Random.seed!(3)

struct Bond
    from::Int
    to::Int
    order::Int
end

struct Molecule
    pointcloud::PointCloud
    elements::Vector
    bonds::Vector{Bond}
end


function cached_get_cid(name)
    cachefile = joinpath("data", "cache-cid-$name")
    if !isdir(dirname(cachefile))
        mkdir(dirname(cachefile))
    end
    if !isfile(cachefile)
        @info "missing cache file for $name, calling PubChem API"
        cid = get_cid(; name)
        write(cachefile, cid)
    end
    return read(cachefile, Int64)
end

function load_pubchem_data(cids; record_type = "3d")
    all_cids = values(cids)
    needs_download = true
    cachefile = joinpath("data", "cache-pubchem.json")
    if !isdir(dirname(cachefile))
        mkdir(dirname(cachefile))
    end
    if isfile(cachefile)
        json = JSON.parse(read(cachefile, String))
        if all(cid -> find_compound(json, cid) !== nothing, all_cids)
            needs_download = false
        end
    end
    if needs_download
        @info "Downloading PubChem data for $(length(all_cids)) compounds."
        pubchem_res = get_for_cids(all_cids; output = "JSON", record_type)
        write(cachefile, pubchem_res)
        json = JSON.parse(read(cachefile, String))
    end
    return map(cid -> find_compound(json, cid), cids)
end

function find_compound(json, cid)
    i = findfirst(c -> c.id.id.cid == cid, json.PC_Compounds)
    isnothing(i) && return nothing
    c = json.PC_Compounds[i]
    at = c.atoms
    elements = at.element
    b = c.bonds
    bonds = map(Bond, b.aid1, b.aid2, b.order)
    cnf = c.coords[1].conformers[1]
    z = :z in keys(cnf) ? cnf.z : zeros(length(elements))
    crds = hcat(cnf.x, cnf.y, z) |> transpose .|> float
    pointcloud = PointCloud(crds)
    return Molecule(pointcloud, elements, bonds)
end

function degrees(m)
    map(1:length(m.bonds)) do i
        count(m.bonds) do bond
            bond.from == i || bond.to == i
        end
    end
end

function neighbors(m, i)
    n = Int[]
    for bond in m.bonds
        if bond.from == i
            push!(n, bond.to)
        elseif bond.to == i
            push!(n, bond.from)
        end
    end
    return n
end


# [NOTE] The following function is based on code in MolecularGraph.jl.
# (MIT License, Seiji Matsuoka and contributors)
#
# The aesthetic presentation is inspired by PyMol.

function plot_molecule!(ax, mol::Molecule; color)
    dgr = degrees(mol)
    cylinders = Cylinder{Float64}[]
    spheres = Sphere{Float64}[]
    for bond in mol.bonds
        (; from, to, order) = bond
        pos1 = mol.pointcloud[from].coords
        pos2 = mol.pointcloud[to].coords
        bonddiameter = if order > 1 || any(==(1), mol.elements[[from, to]])
            0.1
        else
            0.2
        end
        normaldir = SA[0.0, 0.0, 1.0]
        if order > 1
            d1, d2 = dgr[from], dgr[to]
            # determine the plane for double bonds
            ns = if d1 == 3
                filter(!=(to), neighbors(mol, from))
            elseif d2 == 3
                filter(!=(from), neighbors(mol, to))
            end
            @assert length(ns) == 2
            npos1 = mol.pointcloud[ns[1]].coords
            npos2 = mol.pointcloud[ns[2]].coords
            normaldir = cross(npos1, npos2)
        end
        sepdir = normalize(cross(normaldir, pos2 - pos1))
        dists = (bonddiameter * 3) .* collect((-0.5 * (order - 1)):(0.5 * (order - 1)))
        for dist in dists
            dvec = dist * sepdir
            p1, p2 = pos1 + dvec, pos2 + dvec
            cyl = Cylinder(Point3(p1), Point3(p2), bonddiameter)
            sph1 = Sphere(Point3(p1), bonddiameter)
            sph2 = Sphere(Point3(p2), bonddiameter)
            push!(cylinders, cyl)
            push!(spheres, sph1)
            push!(spheres, sph2)
        end
    end
    mesh!(ax, cylinders; color)
    mesh!(ax, spheres; color)
end

compound_names = (source = "adenine", target = "adenosine")

@info "Acquiring PubChem CIDs..."

cids = map(cached_get_cid, compound_names)

@info "Loading PubChem data..."

compounds = load_pubchem_data(cids);

@info "Computing correspondences..."

correspondences_full =
    matching_labels(compounds.source.elements, compounds.target.elements)
correspondences_filtered = compatible_triangles(
    correspondences_full,
    compounds.source.pointcloud,
    compounds.target.pointcloud;
    deviation = 0.01,
)
@info "Number of correspondences" length(correspondences_full.idcs) length(correspondences_filtered.idcs)

@info "Setting up algorithms..."

algorithms = [
    Kabsch(),
    GemanMcClureMM(),
    GemanMcClureMM(scale = 0.1, restarts = RandomRestarts(100)),
];
corrs = [correspondences_full, correspondences_filtered];

@info "Performing registration..."

registered_sources = [
    rigidly_registered(
        compounds.source.pointcloud,
        compounds.target.pointcloud,
        alg;
        correspondences = corr,
    )
    for corr in corrs, alg in algorithms
]

@info "Computing RMSDs..."

best_fit_source = registered_sources[end, end]
true_correspondences = map(enumerate(points(best_fit_source))) do (j, src)
    d, i = findmin(points(compounds.target.pointcloud)) do trg
        euclidean(src, trg)
    end
    # There is a H-atom in adenine that has no corresponding atom in adenosine,
    # so we skip it.
    d < 0.2 ? (i, j) : missing
end |> skipmissing |> collect

rmsds = map(registered_sources) do source
    msd = mean(true_correspondences) do (i, j)
        src = points(source)[j]
        trg = points(compounds.target.pointcloud)[i]
        sqeuclidean(src, trg)
    end
    sqrt(msd)
end

@info "Creating figure..."

let
    fig = Figure()
    axs = [
        Axis3(
            fig[Tuple(ci)...];
            aspect = :data,
            protrusions = 0,
            alignmode = Outside(-30),
            viewmode = :fit,
            clip = false,
            elevation = deg2rad(120),
            azimuth = deg2rad(70),
            titlegap = 0.0,
            titlefont = :regular,
        )
        for ci in CartesianIndices(registered_sources)
    ]
    hidedecorations!.(axs)
    hidespines!.(axs)
    collabel(c, txt) = Label(
        fig[0, c],
        txt;
        font = :bold,
        tellwidth = false,
        valign = :top
    )
    rowlabel(r, txt) = Label(
        fig[r, 0],
        txt;
        font = :bold,
        tellheight = false,
        rotation = deg2rad(90)
    )
    rowlabel(1, "matching\nelements")
    rowlabel(2, "compatible\ntriangles")
    collabel(1, "Kabsch")
    collabel(2, "Geman-McClure")
    collabel(3, "Geman-McClure\ncustomized")
    for ci in CartesianIndices(axs)
        ax = axs[ci]
        Label(
            fig[Tuple(ci)...],
            format("RMSD = {:.2f} Å", rmsds[ci]);
            tellwidth = false,
            tellheight = false,
            halign = 0.3,
            valign = 0.9,
        )
        registered_source = registered_sources[ci]
        plot_molecule!(
            ax,
            Molecule(
                registered_source,
                compounds.source.elements,
                compounds.source.bonds
            );
            color = colorant"#0074d9"
        )
        plot_molecule!(
            ax,
            compounds.target;
            color = colorant"#ff4136"
        )
    end
    colgap!(fig.layout, 0)
    rowgap!(fig.layout, 0)
    resize_to_layout!(fig)
    fig
    save("pubchem.png", fig; px_per_unit = 2)
end

try
    @info "Trying to minimize figure..."
    run(`oxipng pubchem.png`)
catch e
    if e isa IOError
        @warn "`oxipng` probably not available, figure will not be minimized."
    end
end

@info "Done."

