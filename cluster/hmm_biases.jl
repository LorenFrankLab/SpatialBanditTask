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
# ("hmm_leaf", run_hmm_leaf),
# ("hmm_leaf_γ2", run_hmm_leaf_γ2),
# ("hmm_leaf_retainbelief", run_hmm_leaf_retainbelief),
# ("hmm_leaf_stay", run_hmm_leaf_stay),
# ("hmm_leaf_turn", run_hmm_leaf_turn),
# ("hmm_leaf_spatial", run_hmm_leaf_spatial),
# ("hmm_leaf_leafspatial", run_hmm_leaf_leafspatial),
# ("hmm_leaf_leafturn", run_hmm_leaf_leafturn),

# ("hmm_leaf_stay_γ2", run_hmm_leaf_stay_γ2),
# ("hmm_leaf_stay_retainbelief", run_hmm_leaf_stay_retainbelief),
# ("hmm_leaf_stay_turn", run_hmm_leaf_stay_turn),
# ("hmm_leaf_stay_spatial", run_hmm_leaf_stay_spatial),
# ("hmm_leaf_stay_leafturn", run_hmm_leaf_stay_leafturn),
# ("hmm_leaf_stay_leafspatial", run_hmm_leaf_stay_leafspatial),
# ("hmm_leaf_stay_turn_leafspatial", run_hmm_leaf_stay_turn_leafspatial),
# ("hmm_leaf_stay_turn_leafturn", run_hmm_leaf_stay_turn_leafturn),
# ("hmm_leaf_stay_spatial_leafspatial", run_hmm_leaf_stay_spatial_leafspatial),
# ("hmm_leaf_stay_spatial_leafturn", run_hmm_leaf_stay_spatial_leafturn),

# ("hmm_leaf_turn_γ2", run_hmm_leaf_turn_γ2),
# ("hmm_leaf_spatial_γ2", run_hmm_leaf_spatial_γ2),
# ("hmm_leaf_turn_leafspatial_γ2", run_hmm_leaf_turn_leafspatial_γ2),
# ("hmm_leaf_turn_leafturn_γ2", run_hmm_leaf_turn_leafturn_γ2),
# ("hmm_leaf_spatial_leafspatial_γ2", run_hmm_leaf_spatial_leafspatial_γ2),
# ("hmm_leaf_spatial_leafturn_γ2", run_hmm_leaf_spatial_leafturn_γ2),

# ("hmm_leaf_stay_turn_γ2", run_hmm_leaf_stay_turn_γ2),
# ("hmm_leaf_stay_spatial_γ2", run_hmm_leaf_stay_spatial_γ2),
# ("hmm_leaf_stay_leafturn_γ2", run_hmm_leaf_stay_leafturn_γ2),
# ("hmm_leaf_stay_leafspatial_γ2", run_hmm_leaf_stay_leafspatial_γ2),
# ("hmm_leaf_stay_turn_leafspatial_γ2", run_hmm_leaf_stay_turn_leafspatial_γ2),
# ("hmm_leaf_stay_turn_leafturn_γ2", run_hmm_leaf_stay_turn_leafturn_γ2),
# ("hmm_leaf_stay_spatial_leafspatial_γ2", run_hmm_leaf_stay_spatial_leafspatial_γ2),
# ("hmm_leaf_stay_spatial_leafturn_γ2", run_hmm_leaf_stay_spatial_leafturn_γ2),

# Test against depletion code
# ("hmm_leaf_stay_turn_depletion", run_hmm_leaf_stay_turn_depletion),
# ("hmm_leaf_stay_spatial_depletion", run_hmm_leaf_stay_spatial_depletion),
# ("hmm_leaf_stay_turn_leafspatial_depletion", run_hmm_leaf_stay_turn_leafspatial_depletion),
# ("hmm_leaf_stay_turn_leafturn_depletion", run_hmm_leaf_stay_turn_leafturn_depletion),
# ("hmm_leaf_stay_spatial_leafspatial_depletion", run_hmm_leaf_stay_spatial_leafspatial_depletion),
# ("hmm_leaf_stay_spatial_leafturn_depletion", run_hmm_leaf_stay_spatial_leafturn_depletion),

("hmm_stay", run_hmm_stay),
("hmm_stay_turn", run_hmm_stay_turn),
("hmm_stay_spatial", run_hmm_stay_spatial),
]
base_dir = "../results/hmm_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, "..")
(fn_name, fn) = fns[fn_ind]
flag_loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info flag_loocv

function run_fn(fn_name, fn, rewscaled, delay_turn_bias, flag_loocv, full)
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
        Q = find_Q_vals_by_day(data, results; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias);
        CSV.write("$(base_dir)/Q_vals_$(fname).csv.gz", Q; compress=true)
    end
end

# run_fn(fn_name, fn, false, false, flag_loocv, false)
run_fn(fn_name, fn, true, false, flag_loocv, false)
# run_fn(fn_name, fn, true, true, flag_loocv, false)