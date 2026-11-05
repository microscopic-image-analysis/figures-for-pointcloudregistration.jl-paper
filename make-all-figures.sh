#!/bin/bash

pushd kabsch-benchmark
julia --project=. kabsch-benchmark.jl
popd

pushd pubchem
julia --project=. pubchem.jl
popd

pushd pdb-variances
julia --project=. pdb-variances.jl
popd

pushd cryo-em-densities
julia --project=. cryo-em-densities.jl
popd

