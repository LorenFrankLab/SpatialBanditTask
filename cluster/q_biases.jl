using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/qlearner.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_day(data, results; add_leaf=true, rewscaled, delay_turn_bias)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = qlik(view(data, data.daynum .== i, :), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

fns = [
# ("q_leaf", run_q_leaf, true),
# ("q_leaf_γ2", run_q_leaf_γ2, true),
# ("q_leaf_retainbelief", run_q_leaf_retainbelief, true),
# ("q_leaf_stay", run_q_leaf_stay, true),
# ("q_leaf_stay_γ2", run_q_leaf_stay_γ2, true),
# ("q_leaf_stay_retainbelief", run_q_leaf_stay_retainbelief, true),

# ("q_leaf_stay_turn", run_q_leaf_stay_turn, true),
# ("q_leaf_stay_spatial", run_q_leaf_stay_spatial, true),
# ("q_leaf_stay_leafturn", run_q_leaf_stay_leafturn, true),
# ("q_leaf_stay_leafspatial", run_q_leaf_stay_leafspatial, true),
# ("q_leaf_stay_turn_leafspatial", run_q_leaf_stay_turn_leafspatial, true),
# ("q_leaf_stay_turn_leafturn", run_q_leaf_stay_turn_leafturn, true),
# ("q_leaf_stay_spatial_leafspatial", run_q_leaf_stay_spatial_leafspatial, true),
# ("q_leaf_stay_spatial_leafturn", run_q_leaf_stay_spatial_leafturn, true),

# ("q_leaf_stay_turn_γ2", run_q_leaf_stay_turn_γ2, true),
# ("q_leaf_stay_spatial_γ2", run_q_leaf_stay_spatial_γ2, true),
# ("q_leaf_stay_leafturn_γ2", run_q_leaf_stay_leafturn_γ2, true),
# ("q_leaf_stay_leafspatial_γ2", run_q_leaf_stay_leafspatial_γ2, true),
# ("q_leaf_stay_turn_leafspatial_γ2", run_q_leaf_stay_turn_leafspatial_γ2, true),
# ("q_leaf_stay_turn_leafturn_γ2", run_q_leaf_stay_turn_leafturn_γ2, true),
# ("q_leaf_stay_spatial_leafspatial_γ2", run_q_leaf_stay_spatial_leafspatial_γ2, true),
# ("q_leaf_stay_spatial_leafturn_γ2", run_q_leaf_stay_spatial_leafturn_γ2, true),

# ("q_leaf_initialQ", run_q_leaf_initialQ, true),
# ("q_leaf_initialQ_γ2", run_q_leaf_initialQ_γ2, true),
# ("q_leaf_initialQ_retainbelief", run_q_leaf_initialQ_retainbelief, true),
# ("q_leaf_initialQ_stay", run_q_leaf_initialQ_stay, true),
# ("q_leaf_initialQ_stay_γ2", run_q_leaf_initialQ_stay_γ2, true),
# ("q_leaf_initialQ_stay_retainbelief", run_q_leaf_initialQ_stay_retainbelief, true),

# ("q_leaf_initialQ_stay_turn", run_q_leaf_initialQ_stay_turn, true),
# ("q_leaf_initialQ_stay_spatial", run_q_leaf_initialQ_stay_spatial, true),
# ("q_leaf_initialQ_stay_leafturn", run_q_leaf_initialQ_stay_leafturn, true),
# ("q_leaf_initialQ_stay_leafspatial", run_q_leaf_initialQ_stay_leafspatial, true),
# ("q_leaf_initialQ_stay_turn_leafspatial", run_q_leaf_initialQ_stay_turn_leafspatial, true),
# ("q_leaf_initialQ_stay_turn_leafturn", run_q_leaf_initialQ_stay_turn_leafturn, true),
# ("q_leaf_initialQ_stay_spatial_leafspatial", run_q_leaf_initialQ_stay_spatial_leafspatial, true),
# ("q_leaf_initialQ_stay_spatial_leafturn", run_q_leaf_initialQ_stay_spatial_leafturn, true),

# ("q_leaf_initialQ_stay_turn_γ2", run_q_leaf_initialQ_stay_turn_γ2, true),
# ("q_leaf_initialQ_stay_spatial_γ2", run_q_leaf_initialQ_stay_spatial_γ2, true),
# ("q_leaf_initialQ_stay_leafturn_γ2", run_q_leaf_initialQ_stay_leafturn_γ2, true),
# ("q_leaf_initialQ_stay_leafspatial_γ2", run_q_leaf_initialQ_stay_leafspatial_γ2, true),
# ("q_leaf_initialQ_stay_turn_leafspatial_γ2", run_q_leaf_initialQ_stay_turn_leafspatial_γ2, true),
# ("q_leaf_initialQ_stay_turn_leafturn_γ2", run_q_leaf_initialQ_stay_turn_leafturn_γ2, true),
# ("q_leaf_initialQ_stay_spatial_leafspatial_γ2", run_q_leaf_initialQ_stay_spatial_leafspatial_γ2, true),
# ("q_leaf_initialQ_stay_spatial_leafturn_γ2", run_q_leaf_initialQ_stay_spatial_leafturn_γ2, true),

("q_leaf_decay", run_q_leaf_decay, true),
("q_leaf_γ2_decay", run_q_leaf_γ2_decay, true),
("q_leaf_retainbelief_decay", run_q_leaf_retainbelief_decay, true),
("q_leaf_stay_decay", run_q_leaf_stay_decay, true),
("q_leaf_stay_γ2_decay", run_q_leaf_stay_γ2_decay, true),
("q_leaf_stay_retainbelief_decay", run_q_leaf_stay_retainbelief_decay, true),

("q_leaf_stay_turn_decay", run_q_leaf_stay_turn_decay, true),
("q_leaf_stay_spatial_decay", run_q_leaf_stay_spatial_decay, true),
("q_leaf_stay_leafturn_decay", run_q_leaf_stay_leafturn_decay, true),
("q_leaf_stay_leafspatial_decay", run_q_leaf_stay_leafspatial_decay, true),
("q_leaf_stay_turn_leafspatial_decay", run_q_leaf_stay_turn_leafspatial_decay, true),
("q_leaf_stay_turn_leafturn_decay", run_q_leaf_stay_turn_leafturn_decay, true),
("q_leaf_stay_spatial_leafspatial_decay", run_q_leaf_stay_spatial_leafspatial_decay, true),
("q_leaf_stay_spatial_leafturn_decay", run_q_leaf_stay_spatial_leafturn_decay, true),

("q_leaf_stay_turn_γ2_decay", run_q_leaf_stay_turn_γ2_decay, true),
("q_leaf_stay_spatial_γ2_decay", run_q_leaf_stay_spatial_γ2_decay, true),
("q_leaf_stay_leafturn_γ2_decay", run_q_leaf_stay_leafturn_γ2_decay, true),
("q_leaf_stay_leafspatial_γ2_decay", run_q_leaf_stay_leafspatial_γ2_decay, true),
("q_leaf_stay_turn_leafspatial_γ2_decay", run_q_leaf_stay_turn_leafspatial_γ2_decay, true),
("q_leaf_stay_turn_leafturn_γ2_decay", run_q_leaf_stay_turn_leafturn_γ2_decay, true),
("q_leaf_stay_spatial_leafspatial_γ2_decay", run_q_leaf_stay_spatial_leafspatial_γ2_decay, true),
("q_leaf_stay_spatial_leafturn_γ2_decay", run_q_leaf_stay_spatial_leafturn_γ2_decay, true),

("q_leaf_initialQ_decay", run_q_leaf_initialQ_decay, true),
("q_leaf_initialQ_γ2_decay", run_q_leaf_initialQ_γ2_decay, true),
("q_leaf_initialQ_retainbelief_decay", run_q_leaf_initialQ_retainbelief_decay, true),
("q_leaf_initialQ_stay_decay", run_q_leaf_initialQ_stay_decay, true),
("q_leaf_initialQ_stay_γ2_decay", run_q_leaf_initialQ_stay_γ2_decay, true),
("q_leaf_initialQ_stay_retainbelief_decay", run_q_leaf_initialQ_stay_retainbelief_decay, true),

("q_leaf_initialQ_stay_turn_decay", run_q_leaf_initialQ_stay_turn_decay, true),
("q_leaf_initialQ_stay_spatial_decay", run_q_leaf_initialQ_stay_spatial_decay, true),
("q_leaf_initialQ_stay_leafturn_decay", run_q_leaf_initialQ_stay_leafturn_decay, true),
("q_leaf_initialQ_stay_leafspatial_decay", run_q_leaf_initialQ_stay_leafspatial_decay, true),
("q_leaf_initialQ_stay_turn_leafspatial_decay", run_q_leaf_initialQ_stay_turn_leafspatial_decay, true),
("q_leaf_initialQ_stay_turn_leafturn_decay", run_q_leaf_initialQ_stay_turn_leafturn_decay, true),
("q_leaf_initialQ_stay_spatial_leafspatial_decay", run_q_leaf_initialQ_stay_spatial_leafspatial_decay, true),
("q_leaf_initialQ_stay_spatial_leafturn_decay", run_q_leaf_initialQ_stay_spatial_leafturn_decay, true),

("q_leaf_initialQ_stay_turn_γ2_decay", run_q_leaf_initialQ_stay_turn_γ2_decay, true),
("q_leaf_initialQ_stay_spatial_γ2_decay", run_q_leaf_initialQ_stay_spatial_γ2_decay, true),
("q_leaf_initialQ_stay_leafturn_γ2_decay", run_q_leaf_initialQ_stay_leafturn_γ2_decay, true),
("q_leaf_initialQ_stay_leafspatial_γ2_decay", run_q_leaf_initialQ_stay_leafspatial_γ2_decay, true),
("q_leaf_initialQ_stay_turn_leafspatial_γ2_decay", run_q_leaf_initialQ_stay_turn_leafspatial_γ2_decay, true),
("q_leaf_initialQ_stay_turn_leafturn_γ2_decay", run_q_leaf_initialQ_stay_turn_leafturn_γ2_decay, true),
("q_leaf_initialQ_stay_spatial_leafspatial_γ2_decay", run_q_leaf_initialQ_stay_spatial_leafspatial_γ2_decay, true),
("q_leaf_initialQ_stay_spatial_leafturn_γ2_decay", run_q_leaf_initialQ_stay_spatial_leafturn_γ2_decay, true),

# No Leaf
# ("q_base", run_q_base, false),
# ("q_γ2", run_q_γ2, false),
# ("q_retainbelief", run_q_retainbelief, false),
# ("q_stay", run_q_stay, false),
# ("q_stay_γ2", run_q_stay_γ2, false),
# ("q_stay_retainbelief", run_q_stay_retainbelief, false),

# ("q_stay_turn", run_q_stay_turn, false),
# ("q_stay_spatial", run_q_stay_spatial, false),
# ("q_stay_turn_γ2", run_q_stay_turn_γ2, false),
# ("q_stay_spatial_γ2", run_q_stay_spatial_γ2, false),

# ("q_initialQ", run_q_initialQ, false),
# ("q_initialQ_γ2", run_q_initialQ_γ2, false),
# ("q_initialQ_retainbelief", run_q_initialQ_retainbelief, false),
# ("q_initialQ_stay", run_q_initialQ_stay, false),
# ("q_initialQ_stay_γ2", run_q_initialQ_stay_γ2, false),
# ("q_initialQ_stay_retainbelief", run_q_initialQ_stay_retainbelief, false),

# ("q_initialQ_stay_turn", run_q_initialQ_stay_turn, false),
# ("q_initialQ_stay_spatial", run_q_initialQ_stay_spatial, false),
# ("q_initialQ_stay_turn_γ2", run_q_initialQ_stay_turn_γ2, false),
# ("q_initialQ_stay_spatial_γ2", run_q_initialQ_stay_spatial_γ2, false),

("q_decay", run_q_decay, false),
("q_γ2_decay", run_q_γ2_decay, false),
("q_retainbelief_decay", run_q_retainbelief_decay, false),
("q_stay_decay", run_q_stay_decay, false),
("q_stay_γ2_decay", run_q_stay_γ2_decay, false),
("q_stay_retainbelief_decay", run_q_stay_retainbelief_decay, false),

("q_stay_turn_decay", run_q_stay_turn_decay, false),
("q_stay_spatial_decay", run_q_stay_spatial_decay, false),
("q_stay_turn_γ2_decay", run_q_stay_turn_γ2_decay, false),
("q_stay_spatial_γ2_decay", run_q_stay_spatial_γ2_decay, false),

("q_initialQ_decay", run_q_initialQ_decay, false),
("q_initialQ_γ2_decay", run_q_initialQ_γ2_decay, false),
("q_initialQ_retainbelief_decay", run_q_initialQ_retainbelief_decay, false),
("q_initialQ_stay_decay", run_q_initialQ_stay_decay, false),
("q_initialQ_stay_γ2_decay", run_q_initialQ_stay_γ2_decay, false),
("q_initialQ_stay_retainbelief_decay", run_q_initialQ_stay_retainbelief_decay, false),

("q_initialQ_stay_turn_decay", run_q_initialQ_stay_turn_decay, false),
("q_initialQ_stay_spatial_decay", run_q_initialQ_stay_spatial_decay, false),
("q_initialQ_stay_turn_γ2_decay", run_q_initialQ_stay_turn_γ2_decay, false),
("q_initialQ_stay_spatial_γ2_decay", run_q_initialQ_stay_spatial_γ2_decay, false),
]
base_dir = "../results/q_biases"
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

run_fn(fn_name, fn, fn_add_leaf, false, false, flag_loocv, false)
run_fn(fn_name, fn, fn_add_leaf, true, false, flag_loocv, false)
# run_fn(fn_name, fn, fn_add_leaf, true, true, flag_loocv, false)