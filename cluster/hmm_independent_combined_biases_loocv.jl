using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/hmm_independent.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_day(data, results; add_leaf=true, rewscaled, delay_turn_bias)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = hmm_independent_lik(view(data, data.daynum .== i, :), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

fns = [
    ("hmm_independent_leaf_depletion", run_hmm_independent_leaf_depletion),
    ("hmm_independent_leaf_γ2_depletion", run_hmm_independent_leaf_γ2_depletion),
    ("hmm_independent_leaf_retainbelief_depletion", run_hmm_independent_leaf_retainbelief_depletion),
    ("hmm_independent_leaf_stay_depletion", run_hmm_independent_leaf_stay_depletion),
    ("hmm_independent_leaf_turn_depletion", run_hmm_independent_leaf_turn_depletion),
    ("hmm_independent_leaf_spatial_depletion", run_hmm_independent_leaf_spatial_depletion),
    ("hmm_independent_leaf_leafspatial_depletion", run_hmm_independent_leaf_leafspatial_depletion),
    ("hmm_independent_leaf_leafturn_depletion", run_hmm_independent_leaf_leafturn_depletion),

    ("hmm_independent_leaf_stay_γ2_depletion", run_hmm_independent_leaf_stay_γ2_depletion),
    ("hmm_independent_leaf_stay_retainbelief_depletion", run_hmm_independent_leaf_stay_retainbelief_depletion),
    ("hmm_independent_leaf_stay_turn_depletion", run_hmm_independent_leaf_stay_turn_depletion),
    ("hmm_independent_leaf_stay_spatial_depletion", run_hmm_independent_leaf_stay_spatial_depletion),
    ("hmm_independent_leaf_stay_leafspatial_depletion", run_hmm_independent_leaf_stay_leafspatial_depletion),
    ("hmm_independent_leaf_stay_leafturn_depletion", run_hmm_independent_leaf_stay_leafturn_depletion),
    ("hmm_independent_leaf_stay_turn_leafspatial_depletion", run_hmm_independent_leaf_stay_turn_leafspatial_depletion),
    ("hmm_independent_leaf_stay_turn_leafturn_depletion", run_hmm_independent_leaf_stay_turn_leafturn_depletion),
    ("hmm_independent_leaf_stay_spatial_leafspatial_depletion", run_hmm_independent_leaf_stay_spatial_leafspatial_depletion),
    ("hmm_independent_leaf_stay_spatial_leafturn_depletion", run_hmm_independent_leaf_stay_spatial_leafturn_depletion),

    ("hmm_independent_leaf_turn_γ2_depletion", run_hmm_independent_leaf_turn_γ2_depletion),
    ("hmm_independent_leaf_spatial_γ2_depletion", run_hmm_independent_leaf_spatial_γ2_depletion),
    ("hmm_independent_leaf_turn_leafspatial_γ2_depletion", run_hmm_independent_leaf_turn_leafspatial_γ2_depletion),
    ("hmm_independent_leaf_turn_leafturn_γ2_depletion", run_hmm_independent_leaf_turn_leafturn_γ2_depletion),
    ("hmm_independent_leaf_spatial_leafspatial_γ2_depletion", run_hmm_independent_leaf_spatial_leafspatial_γ2_depletion),
    ("hmm_independent_leaf_spatial_leafturn_γ2_depletion", run_hmm_independent_leaf_spatial_leafturn_γ2_depletion),

    ("hmm_independent_leaf_stay_turn_γ2_depletion", run_hmm_independent_leaf_stay_turn_γ2_depletion),
    ("hmm_independent_leaf_stay_spatial_γ2_depletion", run_hmm_independent_leaf_stay_spatial_γ2_depletion),
    ("hmm_independent_leaf_stay_leafturn_γ2_depletion", run_hmm_independent_leaf_stay_leafturn_γ2_depletion),
    ("hmm_independent_leaf_stay_leafspatial_γ2_depletion", run_hmm_independent_leaf_stay_leafspatial_γ2_depletion),
    ("hmm_independent_leaf_stay_turn_leafspatial_γ2_depletion", run_hmm_independent_leaf_stay_turn_leafspatial_γ2_depletion),
    ("hmm_independent_leaf_stay_turn_leafturn_γ2_depletion", run_hmm_independent_leaf_stay_turn_leafturn_γ2_depletion),
    ("hmm_independent_leaf_stay_spatial_leafspatial_γ2_depletion", run_hmm_independent_leaf_stay_spatial_leafspatial_γ2_depletion),
    ("hmm_independent_leaf_stay_spatial_leafturn_γ2_depletion", run_hmm_independent_leaf_stay_spatial_leafturn_γ2_depletion),
]
base_dir = "../results/hmm_independent_combined_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal_combined(animal, "..")
(fn_name, fn) = fns[fn_ind]

@info animal
@info fn_name

function run_fn(fn_name, fn, rewscaled, delay_turn_bias)
    # Create the base filename
    fname = fn_name
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname *= "_$(animal)_combined"
    @info fname
    
    fname_loocv = fname * "_loocv"
    
    results = load("$(base_dir)/$(fname).jld2", fname)
    results_loocv = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, loocv_data=results)
    save("$(base_dir)/$(fname_loocv).jld2", fname_loocv, results_loocv; compress=true)
end

# run_fn(fn_name, fn, false, false)
run_fn(fn_name, fn, true, false)
# run_fn(fn_name, fn, true, true)