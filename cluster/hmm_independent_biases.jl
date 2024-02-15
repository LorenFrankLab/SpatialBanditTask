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
("hmm_independent_leaf", run_hmm_independent_leaf, true),
("hmm_independent_leaf_γ2", run_hmm_independent_leaf_γ2, true),
("hmm_independent_leaf_retainbelief", run_hmm_independent_leaf_retainbelief, true),
("hmm_independent_leaf_stay", run_hmm_independent_leaf_stay, true),
("hmm_independent_leaf_turn", run_hmm_independent_leaf_turn, true),
("hmm_independent_leaf_spatial", run_hmm_independent_leaf_spatial, true),
("hmm_independent_leaf_leafspatial", run_hmm_independent_leaf_leafspatial, true),
("hmm_independent_leaf_leafturn", run_hmm_independent_leaf_leafturn, true),

("hmm_independent_leaf_stay_γ2", run_hmm_independent_leaf_stay_γ2, true),
("hmm_independent_leaf_stay_retainbelief", run_hmm_independent_leaf_stay_retainbelief, true),
("hmm_independent_leaf_stay_turn", run_hmm_independent_leaf_stay_turn, true),
("hmm_independent_leaf_stay_spatial", run_hmm_independent_leaf_stay_spatial, true),
("hmm_independent_leaf_stay_leafturn", run_hmm_independent_leaf_stay_leafturn, true),
("hmm_independent_leaf_stay_leafspatial", run_hmm_independent_leaf_stay_leafspatial, true),
("hmm_independent_leaf_stay_turn_leafspatial", run_hmm_independent_leaf_stay_turn_leafspatial, true),
("hmm_independent_leaf_stay_turn_leafturn", run_hmm_independent_leaf_stay_turn_leafturn, true),
("hmm_independent_leaf_stay_spatial_leafspatial", run_hmm_independent_leaf_stay_spatial_leafspatial, true),
("hmm_independent_leaf_stay_spatial_leafturn", run_hmm_independent_leaf_stay_spatial_leafturn, true),

("hmm_independent_leaf_turn_γ2", run_hmm_independent_leaf_turn_γ2, true),
("hmm_independent_leaf_spatial_γ2", run_hmm_independent_leaf_spatial_γ2, true),
("hmm_independent_leaf_turn_leafspatial_γ2", run_hmm_independent_leaf_turn_leafspatial_γ2, true),
("hmm_independent_leaf_turn_leafturn_γ2", run_hmm_independent_leaf_turn_leafturn_γ2, true),
("hmm_independent_leaf_spatial_leafspatial_γ2", run_hmm_independent_leaf_spatial_leafspatial_γ2, true),
("hmm_independent_leaf_spatial_leafturn_γ2", run_hmm_independent_leaf_spatial_leafturn_γ2, true),

("hmm_independent_leaf_stay_turn_γ2", run_hmm_independent_leaf_stay_turn_γ2, true),
("hmm_independent_leaf_stay_spatial_γ2", run_hmm_independent_leaf_stay_spatial_γ2, true),
("hmm_independent_leaf_stay_leafturn_γ2", run_hmm_independent_leaf_stay_leafturn_γ2, true),
("hmm_independent_leaf_stay_leafspatial_γ2", run_hmm_independent_leaf_stay_leafspatial_γ2, true),
("hmm_independent_leaf_stay_turn_leafspatial_γ2", run_hmm_independent_leaf_stay_turn_leafspatial_γ2, true),
("hmm_independent_leaf_stay_turn_leafturn_γ2", run_hmm_independent_leaf_stay_turn_leafturn_γ2, true),
("hmm_independent_leaf_stay_spatial_leafspatial_γ2", run_hmm_independent_leaf_stay_spatial_leafspatial_γ2, true),
("hmm_independent_leaf_stay_spatial_leafturn_γ2", run_hmm_independent_leaf_stay_spatial_leafturn_γ2, true),

# Test against depletion code
("hmm_independent_leaf_stay_turn_depletion", run_hmm_independent_leaf_stay_turn_depletion, true),
("hmm_independent_leaf_stay_spatial_depletion", run_hmm_independent_leaf_stay_spatial_depletion, true),
("hmm_independent_leaf_stay_turn_leafspatial_depletion", run_hmm_independent_leaf_stay_turn_leafspatial_depletion, true),
("hmm_independent_leaf_stay_turn_leafturn_depletion", run_hmm_independent_leaf_stay_turn_leafturn_depletion, true),
("hmm_independent_leaf_stay_spatial_leafspatial_depletion", run_hmm_independent_leaf_stay_spatial_leafspatial_depletion, true),
("hmm_independent_leaf_stay_spatial_leafturn_depletion", run_hmm_independent_leaf_stay_spatial_leafturn_depletion, true),
]
base_dir = "../results/hmm_independent_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, "..")
(fn_name, fn, fn_add_leaf) = fns[fn_ind]
flag_loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info flag_loocv

function run_fn(fn_name, fn, fn_add_leaf, rewscaled, delay_turn_bias, flag_loocv, full)
    # Create the base filename
    fname = fn_name
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname *= "_$(animal)"
    fname = full ? fname * "_full" : fname
    @info fname
    
    if flag_loocv
        fname_loocv = fname * "_loocv"
    
        results = load("$(base_dir)/$(fname).jld2", "results")
        results_loocv = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, loocv_data=results, full)
        save("$(base_dir)_loocv/$(fname_loocv).jld2", "results_loocv", results_loocv; compress=true)
    else
        results = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, full)
        save("$(base_dir)/$(fname).jld2", "results", results; compress=true)
        write_EM_to_mat(results, "$(base_dir)/$(fname).mat"; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias)
        Q = find_Q_vals_by_day(data, results; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, add_leaf=fn_add_leaf);
        CSV.write("$(base_dir)/Q_vals_$(fname).csv.gz", Q; compress=true)
    end
end

# run_fn(fn_name, fn, false, false, flag_loocv, false)
run_fn(fn_name, fn, fn_add_leaf, true, false, flag_loocv, false)
# run_fn(fn_name, fn, true, true, flag_loocv, false)