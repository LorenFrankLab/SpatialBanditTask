using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/hmm.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_day(data, results; add_leaf=true, rewscaled, delay_turn_bias)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = hmm_lik(view(data, data.daynum .== i, :), get_contingencies(), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

fns = [
    # ("hmm_leaf_depletion", run_hmm_leaf_depletion),
    # ("hmm_leaf_γ2_depletion", run_hmm_leaf_γ2_depletion),
    # ("hmm_leaf_retainbelief_depletion", run_hmm_leaf_retainbelief_depletion),
    ("hmm_leaf_stay_depletion", run_hmm_leaf_stay_depletion),
    # ("hmm_leaf_turn_depletion", run_hmm_leaf_turn_depletion),
    # ("hmm_leaf_spatial_depletion", run_hmm_leaf_spatial_depletion),
    # ("hmm_leaf_leafspatial_depletion", run_hmm_leaf_leafspatial_depletion),
    # ("hmm_leaf_leafturn_depletion", run_hmm_leaf_leafturn_depletion),

    ("hmm_leaf_stay_γ2_depletion", run_hmm_leaf_stay_γ2_depletion),
    ("hmm_leaf_stay_retainbelief_depletion", run_hmm_leaf_stay_retainbelief_depletion),
    ("hmm_leaf_stay_turn_depletion", run_hmm_leaf_stay_turn_depletion),
    # ("hmm_leaf_stay_spatial_depletion", run_hmm_leaf_stay_spatial_depletion),
    ("hmm_leaf_stay_leafspatial_depletion", run_hmm_leaf_stay_leafspatial_depletion),
    ("hmm_leaf_stay_leafturn_depletion", run_hmm_leaf_stay_leafturn_depletion),
    ("hmm_leaf_stay_turn_leafspatial_depletion", run_hmm_leaf_stay_turn_leafspatial_depletion),
    ("hmm_leaf_stay_turn_leafturn_depletion", run_hmm_leaf_stay_turn_leafturn_depletion),
    # ("hmm_leaf_stay_spatial_leafspatial_depletion", run_hmm_leaf_stay_spatial_leafspatial_depletion),
    # ("hmm_leaf_stay_spatial_leafturn_depletion", run_hmm_leaf_stay_spatial_leafturn_depletion),

    ("hmm_leaf_turn_γ2_depletion", run_hmm_leaf_turn_γ2_depletion),
    # ("hmm_leaf_spatial_γ2_depletion", run_hmm_leaf_spatial_γ2_depletion),
    ("hmm_leaf_turn_leafspatial_γ2_depletion", run_hmm_leaf_turn_leafspatial_γ2_depletion),
    ("hmm_leaf_turn_leafturn_γ2_depletion", run_hmm_leaf_turn_leafturn_γ2_depletion),
    # ("hmm_leaf_spatial_leafspatial_γ2_depletion", run_hmm_leaf_spatial_leafspatial_γ2_depletion),
    # ("hmm_leaf_spatial_leafturn_γ2_depletion", run_hmm_leaf_spatial_leafturn_γ2_depletion),

    ("hmm_leaf_stay_turn_γ2_depletion", run_hmm_leaf_stay_turn_γ2_depletion),
    # ("hmm_leaf_stay_spatial_γ2_depletion", run_hmm_leaf_stay_spatial_γ2_depletion),
    ("hmm_leaf_stay_leafturn_γ2_depletion", run_hmm_leaf_stay_leafturn_γ2_depletion),
    ("hmm_leaf_stay_leafspatial_γ2_depletion", run_hmm_leaf_stay_leafspatial_γ2_depletion),
    ("hmm_leaf_stay_turn_leafspatial_γ2_depletion", run_hmm_leaf_stay_turn_leafspatial_γ2_depletion),
    ("hmm_leaf_stay_turn_leafturn_γ2_depletion", run_hmm_leaf_stay_turn_leafturn_γ2_depletion),
    # ("hmm_leaf_stay_spatial_leafspatial_γ2_depletion", run_hmm_leaf_stay_spatial_leafspatial_γ2_depletion),
    # ("hmm_leaf_stay_spatial_leafturn_γ2_depletion", run_hmm_leaf_stay_spatial_leafturn_γ2_depletion),

    # ("hmm_depletion", run_hmm_depletion),
    # ("hmm_γ2_depletion", run_hmm_γ2_depletion),
    # ("hmm_retainbelief_depletion", run_hmm_retainbelief_depletion),
    ("hmm_stay_depletion", run_hmm_stay_depletion),
    # ("hmm_turn_depletion", run_hmm_turn_depletion),
    # ("hmm_spatial_depletion", run_hmm_spatial_depletion),

    # ("hmm_stay_γ2_depletion", run_hmm_stay_γ2_depletion),
    # ("hmm_stay_retainbelief_depletion", run_hmm_stay_retainbelief_depletion),
    ("hmm_stay_turn_depletion", run_hmm_stay_turn_depletion),
    # ("hmm_stay_spatial_depletion", run_hmm_stay_spatial_depletion),

    # ("hmm_turn_γ2_depletion", run_hmm_turn_γ2_depletion),
    # ("hmm_spatial_γ2_depletion", run_hmm_spatial_γ2_depletion),
    ("hmm_stay_turn_γ2_depletion", run_hmm_stay_turn_γ2_depletion),
    # ("hmm_stay_spatial_γ2_depletion", run_hmm_stay_spatial_γ2_depletion),
]
base_dir = "../results/hmm_combined_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal_combined(animal, "..")
(fn_name, fn) = fns[fn_ind]
loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info loocv

function run_fn(fn_name, fn, rewscaled, delay_turn_bias, loocv)
    # Create the base filename
    fname = fn_name
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname *= "_$(animal)_combined"
    @info fname
    
    if loocv
        fname_loocv = fname * "_loocv"
    
        results = load("$(base_dir)/$(fname).jld2", "results")
        results_loocv = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, loocv_data=results)
        save("$(base_dir)_loocv/$(fname_loocv).jld2", "results_loocv", results_loocv; compress=true)
    else
        results = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias)
        save("$(base_dir)/$(fname).jld2", "results", results; compress=true)
        write_EM_to_mat(results, "$(base_dir)/$(fname).mat"; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias)
        Q = find_Q_vals_by_day(data, results; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias);
        CSV.write("$(base_dir)/Q_vals_$(fname).csv.gz", Q; compress=true)
    end
end

# run_fn(fn_name, fn, false, false, loocv)
run_fn(fn_name, fn, true, false, loocv)
# run_fn(fn_name, fn, true, true, loocv)