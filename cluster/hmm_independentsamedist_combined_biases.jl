using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/hmm_independentsamedist.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_day(data, results; add_leaf=true, rewscaled, delay_turn_bias)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = hmm_independentsamedist_lik(view(data, data.daynum .== i, :), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

fns = [
    # ("hmm_independentsamedist_leaf_depletion", run_hmm_independentsamedist_leaf_depletion),
    # ("hmm_independentsamedist_leaf_γ2_depletion", run_hmm_independentsamedist_leaf_γ2_depletion),
    # ("hmm_independentsamedist_leaf_retainbelief_depletion", run_hmm_independentsamedist_leaf_retainbelief_depletion),
    ("hmm_independentsamedist_leaf_stay_depletion", run_hmm_independentsamedist_leaf_stay_depletion),
    # ("hmm_independentsamedist_leaf_turn_depletion", run_hmm_independentsamedist_leaf_turn_depletion),
    # ("hmm_independentsamedist_leaf_spatial_depletion", run_hmm_independentsamedist_leaf_spatial_depletion),
    # ("hmm_independentsamedist_leaf_leafspatial_depletion", run_hmm_independentsamedist_leaf_leafspatial_depletion),
    # ("hmm_independentsamedist_leaf_leafturn_depletion", run_hmm_independentsamedist_leaf_leafturn_depletion),

    ("hmm_independentsamedist_leaf_stay_γ2_depletion", run_hmm_independentsamedist_leaf_stay_γ2_depletion),
    ("hmm_independentsamedist_leaf_stay_retainbelief_depletion", run_hmm_independentsamedist_leaf_stay_retainbelief_depletion),
    ("hmm_independentsamedist_leaf_stay_turn_depletion", run_hmm_independentsamedist_leaf_stay_turn_depletion),
    # ("hmm_independentsamedist_leaf_stay_spatial_depletion", run_hmm_independentsamedist_leaf_stay_spatial_depletion),
    ("hmm_independentsamedist_leaf_stay_leafspatial_depletion", run_hmm_independentsamedist_leaf_stay_leafspatial_depletion),
    ("hmm_independentsamedist_leaf_stay_leafturn_depletion", run_hmm_independentsamedist_leaf_stay_leafturn_depletion),
    ("hmm_independentsamedist_leaf_stay_turn_leafspatial_depletion", run_hmm_independentsamedist_leaf_stay_turn_leafspatial_depletion),
    ("hmm_independentsamedist_leaf_stay_turn_leafturn_depletion", run_hmm_independentsamedist_leaf_stay_turn_leafturn_depletion),
    # ("hmm_independentsamedist_leaf_stay_spatial_leafspatial_depletion", run_hmm_independentsamedist_leaf_stay_spatial_leafspatial_depletion),
    # ("hmm_independentsamedist_leaf_stay_spatial_leafturn_depletion", run_hmm_independentsamedist_leaf_stay_spatial_leafturn_depletion),

    ("hmm_independentsamedist_leaf_turn_γ2_depletion", run_hmm_independentsamedist_leaf_turn_γ2_depletion),
    # ("hmm_independentsamedist_leaf_spatial_γ2_depletion", run_hmm_independentsamedist_leaf_spatial_γ2_depletion),
    ("hmm_independentsamedist_leaf_turn_leafspatial_γ2_depletion", run_hmm_independentsamedist_leaf_turn_leafspatial_γ2_depletion),
    ("hmm_independentsamedist_leaf_turn_leafturn_γ2_depletion", run_hmm_independentsamedist_leaf_turn_leafturn_γ2_depletion),
    # ("hmm_independentsamedist_leaf_spatial_leafspatial_γ2_depletion", run_hmm_independentsamedist_leaf_spatial_leafspatial_γ2_depletion),
    # ("hmm_independentsamedist_leaf_spatial_leafturn_γ2_depletion", run_hmm_independentsamedist_leaf_spatial_leafturn_γ2_depletion),

    ("hmm_independentsamedist_leaf_stay_turn_γ2_depletion", run_hmm_independentsamedist_leaf_stay_turn_γ2_depletion),
    # ("hmm_independentsamedist_leaf_stay_spatial_γ2_depletion", run_hmm_independentsamedist_leaf_stay_spatial_γ2_depletion),
    ("hmm_independentsamedist_leaf_stay_leafturn_γ2_depletion", run_hmm_independentsamedist_leaf_stay_leafturn_γ2_depletion),
    ("hmm_independentsamedist_leaf_stay_leafspatial_γ2_depletion", run_hmm_independentsamedist_leaf_stay_leafspatial_γ2_depletion),
    ("hmm_independentsamedist_leaf_stay_turn_leafspatial_γ2_depletion", run_hmm_independentsamedist_leaf_stay_turn_leafspatial_γ2_depletion),
    ("hmm_independentsamedist_leaf_stay_turn_leafturn_γ2_depletion", run_hmm_independentsamedist_leaf_stay_turn_leafturn_γ2_depletion),
    # ("hmm_independentsamedist_leaf_stay_spatial_leafspatial_γ2_depletion", run_hmm_independentsamedist_leaf_stay_spatial_leafspatial_γ2_depletion),
    # ("hmm_independentsamedist_leaf_stay_spatial_leafturn_γ2_depletion", run_hmm_independentsamedist_leaf_stay_spatial_leafturn_γ2_depletion),

    # ("hmm_independentsamedist_depletion", run_hmm_independentsamedist_depletion),
    # ("hmm_independentsamedist_γ2_depletion", run_hmm_independentsamedist_γ2_depletion),
    # ("hmm_independentsamedist_retainbelief_depletion", run_hmm_independentsamedist_retainbelief_depletion),
    ("hmm_independentsamedist_stay_depletion", run_hmm_independentsamedist_stay_depletion),
    # ("hmm_independentsamedist_turn_depletion", run_hmm_independentsamedist_turn_depletion),
    # ("hmm_independentsamedist_spatial_depletion", run_hmm_independentsamedist_spatial_depletion),

    # ("hmm_independentsamedist_stay_γ2_depletion", run_hmm_independentsamedist_stay_γ2_depletion),
    # ("hmm_independentsamedist_stay_retainbelief_depletion", run_hmm_independentsamedist_stay_retainbelief_depletion),
    ("hmm_independentsamedist_stay_turn_depletion", run_hmm_independentsamedist_stay_turn_depletion),
    # ("hmm_independentsamedist_stay_spatial_depletion", run_hmm_independentsamedist_stay_spatial_depletion),

    # ("hmm_independentsamedist_turn_γ2_depletion", run_hmm_independentsamedist_turn_γ2_depletion),
    # ("hmm_independentsamedist_spatial_γ2_depletion", run_hmm_independentsamedist_spatial_γ2_depletion),
    ("hmm_independentsamedist_stay_turn_γ2_depletion", run_hmm_independentsamedist_stay_turn_γ2_depletion),
    # ("hmm_independentsamedist_stay_spatial_γ2_depletion", run_hmm_independentsamedist_stay_spatial_γ2_depletion),
]
base_dir = "../results/hmm_independentsamedist_combined_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal_combined(animal, "..")
(fn_name, fn) = fns[fn_ind]
flag_loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info flag_loocv

function run_fn(fn_name, fn, rewscaled, delay_turn_bias, flag_loocv)
    # Create the base filename
    fname = fn_name
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname *= "_$(animal)_combined"
    @info fname
    
    if flag_loocv
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

# run_fn(fn_name, fn, false, false, flag_loocv)
run_fn(fn_name, fn, true, false, flag_loocv)
# run_fn(fn_name, fn, true, true, flag_loocv)