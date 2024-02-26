using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/hmm_staygo.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_subject(data, results; add_leaf=true, rewscaled, delay_turn_bias, subjectlevel=:daynum)
    nsubjects = maximum(data[:, subjectlevel] )
    liks = zeros(nsubjects)
    dfs = []
    for i in 1:nsubjects
        (liks[i], df) = hmm_staygo_lik(view(data, data[:, subjectlevel] .== i, :), get_contingencies(), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

fns = [
("hmm_staygo_Bgo_Bstay_Bturn_leaf_stay_turn_leafspatial", run_hmm_staygo_Bgo_Bstay_Bturn_leaf_stay_turn_leafspatial, true),
("hmm_staygo_Bstay_Bturn_leaf_stay_turn_leafspatial", run_hmm_staygo_Bstay_Bturn_leaf_stay_turn_leafspatial, true),
("hmm_staygo_Bgo_Bstay_leaf_stay_turn_leafspatial", run_hmm_staygo_Bgo_Bstay_leaf_stay_turn_leafspatial, true),
]
base_dir = "../results/hmm_staygo_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, "..")
(fn_name, fn, fn_add_leaf) = fns[fn_ind]
flag_loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info flag_loocv

function run_fn(fn_name, fn, fn_add_leaf, rewscaled, delay_turn_bias, bysession, flag_loocv, full)
    # Create the base filename
    fname = fn_name
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname = bysession ? fname * "_bysession" : fname
    fname *= "_$(animal)"
    fname = full ? fname * "_full" : fname
    @info fname

    subjectlevel=:daynum
    if bysession
        subjectlevel=:daysessionnum
    end
    
    if flag_loocv
        fname_loocv = fname * "_loocv"
    
        results = load("$(base_dir)/$(fname).jld2", "results")
        results_loocv = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, subjectlevel, loocv_data=results, full)
        save("$(base_dir)_loocv/$(fname_loocv).jld2", "results_loocv", results_loocv; compress=true)
    else
        results = fn(data; extended=true, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, subjectlevel, full)
        save("$(base_dir)/$(fname).jld2", "results", results; compress=true)
        write_EM_to_mat(results, "$(base_dir)/$(fname).mat"; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, subjectlevel)
        Q = find_Q_vals_by_subject(data, results; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, add_leaf=fn_add_leaf, subjectlevel);
        CSV.write("$(base_dir)/Q_vals_$(fname).csv.gz", Q; compress=true)
    end
end

# run_fn(fn_name, fn, false, false, flag_loocv, false)
# rewscaled, delay_turn_bias, bysession
# run_fn(fn_name, fn, fn_add_leaf, false, false, false, flag_loocv, false)
run_fn(fn_name, fn, fn_add_leaf, true, false, false, flag_loocv, false)
# run_fn(fn_name, fn, true, true, flag_loocv, false)