using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/q2learner.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_day(data, results; add_leaf=true, rewscaled, delay_turn_bias, kwargs...)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = q2lik(view(data, data.daynum .== i, :), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true, kwargs...)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

# (fn_name, fn, fn_add_leaf, fn_fit_initial_Q) 
fns = [
# ("q2_leaf", run_q2_leaf, true, false),
# ("q2_leaf_γ2", run_q2_leaf_γ2, true, false),
# ("q2_leaf_retainbelief", run_q2_leaf_retainbelief, true, false),
# ("q2_leaf_stay", run_q2_leaf_stay, true, false),
# ("q2_leaf_stay_γ2", run_q2_leaf_stay_γ2, true, false),
# ("q2_leaf_stay_retainbelief", run_q2_leaf_stay_retainbelief, true, false),

# ("q2_leaf_stay_turn", run_q2_leaf_stay_turn, true, false),
# ("q2_leaf_stay_spatial", run_q2_leaf_stay_spatial, true, false),
# ("q2_leaf_stay_leafturn", run_q2_leaf_stay_leafturn, true, false),
# ("q2_leaf_stay_leafspatial", run_q2_leaf_stay_leafspatial, true, false),
# ("q2_leaf_stay_turn_leafspatial", run_q2_leaf_stay_turn_leafspatial, true, false),
# ("q2_leaf_stay_turn_leafturn", run_q2_leaf_stay_turn_leafturn, true, false),
# ("q2_leaf_stay_spatial_leafspatial", run_q2_leaf_stay_spatial_leafspatial, true, false),
# ("q2_leaf_stay_spatial_leafturn", run_q2_leaf_stay_spatial_leafturn, true, false),

# ("q2_leaf_stay_turn_γ2", run_q2_leaf_stay_turn_γ2, true, false),
# ("q2_leaf_stay_spatial_γ2", run_q2_leaf_stay_spatial_γ2, true, false),
# ("q2_leaf_stay_leafturn_γ2", run_q2_leaf_stay_leafturn_γ2, true, false),
# ("q2_leaf_stay_leafspatial_γ2", run_q2_leaf_stay_leafspatial_γ2, true, false),
# ("q2_leaf_stay_turn_leafspatial_γ2", run_q2_leaf_stay_turn_leafspatial_γ2, true, false),
# ("q2_leaf_stay_turn_leafturn_γ2", run_q2_leaf_stay_turn_leafturn_γ2, true, false),
# ("q2_leaf_stay_spatial_leafspatial_γ2", run_q2_leaf_stay_spatial_leafspatial_γ2, true, false),
# ("q2_leaf_stay_spatial_leafturn_γ2", run_q2_leaf_stay_spatial_leafturn_γ2, true, false),

# ("q2_leaf_initialQ", run_q2_leaf_initialQ, true, true),
# ("q2_leaf_initialQ_γ2", run_q2_leaf_initialQ_γ2, true, true),
# ("q2_leaf_initialQ_retainbelief", run_q2_leaf_initialQ_retainbelief, true, true),
# ("q2_leaf_initialQ_stay", run_q2_leaf_initialQ_stay, true, true),
# ("q2_leaf_initialQ_stay_γ2", run_q2_leaf_initialQ_stay_γ2, true, true),
# ("q2_leaf_initialQ_stay_retainbelief", run_q2_leaf_initialQ_stay_retainbelief, true, true),

# ("q2_leaf_initialQ_stay_turn", run_q2_leaf_initialQ_stay_turn, true, true),
# ("q2_leaf_initialQ_stay_spatial", run_q2_leaf_initialQ_stay_spatial, true, true),
# ("q2_leaf_initialQ_stay_leafturn", run_q2_leaf_initialQ_stay_leafturn, true, true),
# ("q2_leaf_initialQ_stay_leafspatial", run_q2_leaf_initialQ_stay_leafspatial, true, true),
# ("q2_leaf_initialQ_stay_turn_leafspatial", run_q2_leaf_initialQ_stay_turn_leafspatial, true, true),
# ("q2_leaf_initialQ_stay_turn_leafturn", run_q2_leaf_initialQ_stay_turn_leafturn, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial", run_q2_leaf_initialQ_stay_spatial_leafspatial, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn", run_q2_leaf_initialQ_stay_spatial_leafturn, true, true),

# ("q2_leaf_initialQ_stay_turn_γ2", run_q2_leaf_initialQ_stay_turn_γ2, true, true),
# ("q2_leaf_initialQ_stay_spatial_γ2", run_q2_leaf_initialQ_stay_spatial_γ2, true, true),
# ("q2_leaf_initialQ_stay_leafturn_γ2", run_q2_leaf_initialQ_stay_leafturn_γ2, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_γ2", run_q2_leaf_initialQ_stay_leafspatial_γ2, true, true),
# ("q2_leaf_initialQ_stay_turn_leafspatial_γ2", run_q2_leaf_initialQ_stay_turn_leafspatial_γ2, true, true),
# ("q2_leaf_initialQ_stay_turn_leafturn_γ2", run_q2_leaf_initialQ_stay_turn_leafturn_γ2, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_γ2", run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_γ2", run_q2_leaf_initialQ_stay_spatial_leafturn_γ2, true, true),

# ("q2_leaf_decay", run_q2_leaf_decay, true, false),
# ("q2_leaf_γ2_decay", run_q2_leaf_γ2_decay, true, false),
# ("q2_leaf_retainbelief_decay", run_q2_leaf_retainbelief_decay, true, false),
# ("q2_leaf_stay_decay", run_q2_leaf_stay_decay, true, false),
# ("q2_leaf_stay_γ2_decay", run_q2_leaf_stay_γ2_decay, true, false),
# ("q2_leaf_stay_retainbelief_decay", run_q2_leaf_stay_retainbelief_decay, true, false),

# ("q2_leaf_stay_turn_decay", run_q2_leaf_stay_turn_decay, true, false),
# ("q2_leaf_stay_spatial_decay", run_q2_leaf_stay_spatial_decay, true, false),
# ("q2_leaf_stay_leafturn_decay", run_q2_leaf_stay_leafturn_decay, true, false),
# ("q2_leaf_stay_leafspatial_decay", run_q2_leaf_stay_leafspatial_decay, true, false),
# ("q2_leaf_stay_turn_leafspatial_decay", run_q2_leaf_stay_turn_leafspatial_decay, true, false),
# ("q2_leaf_stay_turn_leafturn_decay", run_q2_leaf_stay_turn_leafturn_decay, true, false),
# ("q2_leaf_stay_spatial_leafspatial_decay", run_q2_leaf_stay_spatial_leafspatial_decay, true, false),
# ("q2_leaf_stay_spatial_leafturn_decay", run_q2_leaf_stay_spatial_leafturn_decay, true, false),

# ("q2_leaf_stay_turn_γ2_decay", run_q2_leaf_stay_turn_γ2_decay, true, false),
# ("q2_leaf_stay_spatial_γ2_decay", run_q2_leaf_stay_spatial_γ2_decay, true, false),
# ("q2_leaf_stay_leafturn_γ2_decay", run_q2_leaf_stay_leafturn_γ2_decay, true, false),
# ("q2_leaf_stay_leafspatial_γ2_decay", run_q2_leaf_stay_leafspatial_γ2_decay, true, false),
# ("q2_leaf_stay_turn_leafspatial_γ2_decay", run_q2_leaf_stay_turn_leafspatial_γ2_decay, true, false),
# ("q2_leaf_stay_turn_leafturn_γ2_decay", run_q2_leaf_stay_turn_leafturn_γ2_decay, true, false),
# ("q2_leaf_stay_spatial_leafspatial_γ2_decay", run_q2_leaf_stay_spatial_leafspatial_γ2_decay, true, false),
# ("q2_leaf_stay_spatial_leafturn_γ2_decay", run_q2_leaf_stay_spatial_leafturn_γ2_decay, true, false),

# ("q2_leaf_initialQ_decay", run_q2_leaf_initialQ_decay, true, true),
# ("q2_leaf_initialQ_γ2_decay", run_q2_leaf_initialQ_γ2_decay, true, true),
# ("q2_leaf_initialQ_retainbelief_decay", run_q2_leaf_initialQ_retainbelief_decay, true, true),
# ("q2_leaf_initialQ_stay_decay", run_q2_leaf_initialQ_stay_decay, true, true),
# ("q2_leaf_initialQ_stay_γ2_decay", run_q2_leaf_initialQ_stay_γ2_decay, true, true),
# ("q2_leaf_initialQ_stay_retainbelief_decay", run_q2_leaf_initialQ_stay_retainbelief_decay, true, true),

# ("q2_leaf_initialQ_stay_turn_decay", run_q2_leaf_initialQ_stay_turn_decay, true, true),
# ("q2_leaf_initialQ_stay_spatial_decay", run_q2_leaf_initialQ_stay_spatial_decay, true, true),
# ("q2_leaf_initialQ_stay_leafturn_decay", run_q2_leaf_initialQ_stay_leafturn_decay, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_decay", run_q2_leaf_initialQ_stay_leafspatial_decay, true, true),
("q2_leaf_initialQ_stay_turn_leafspatial_decay", run_q2_leaf_initialQ_stay_turn_leafspatial_decay, true, true),
("q2_leaf_initialQ_stay_turn_leafturn_decay", run_q2_leaf_initialQ_stay_turn_leafturn_decay, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_decay", run_q2_leaf_initialQ_stay_spatial_leafspatial_decay, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_decay", run_q2_leaf_initialQ_stay_spatial_leafturn_decay, true, true),

# ("q2_leaf_initialQ_stay_turn_γ2_decay", run_q2_leaf_initialQ_stay_turn_γ2_decay, true, true),
# ("q2_leaf_initialQ_stay_spatial_γ2_decay", run_q2_leaf_initialQ_stay_spatial_γ2_decay, true, true),
# ("q2_leaf_initialQ_stay_leafturn_γ2_decay", run_q2_leaf_initialQ_stay_leafturn_γ2_decay, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_γ2_decay", run_q2_leaf_initialQ_stay_leafspatial_γ2_decay, true, true),
("q2_leaf_initialQ_stay_turn_leafspatial_γ2_decay", run_q2_leaf_initialQ_stay_turn_leafspatial_γ2_decay, true, true),
("q2_leaf_initialQ_stay_turn_leafturn_γ2_decay", run_q2_leaf_initialQ_stay_turn_leafturn_γ2_decay, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_γ2_decay", run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2_decay, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_γ2_decay", run_q2_leaf_initialQ_stay_spatial_leafturn_γ2_decay, true, true),

# No Leaf
# ("q2_base", run_q2_base, false, false),
# ("q2_γ2", run_q2_γ2, false, false),
# ("q2_retainbelief", run_q2_retainbelief, false, false),
# ("q2_stay", run_q2_stay, false, false),
# ("q2_stay_γ2", run_q2_stay_γ2, false, false),
# ("q2_stay_retainbelief", run_q2_stay_retainbelief, false, false),

# ("q2_stay_turn", run_q2_stay_turn, false, false),
# ("q2_stay_spatial", run_q2_stay_spatial, false, false),
# ("q2_stay_turn_γ2", run_q2_stay_turn_γ2, false, false),
# ("q2_stay_spatial_γ2", run_q2_stay_spatial_γ2, false, false),

# ("q2_initialQ", run_q2_initialQ, false, true),
# ("q2_initialQ_γ2", run_q2_initialQ_γ2, false, true),
# ("q2_initialQ_retainbelief", run_q2_initialQ_retainbelief, false, true),
# ("q2_initialQ_stay", run_q2_initialQ_stay, false, true),
# ("q2_initialQ_stay_γ2", run_q2_initialQ_stay_γ2, false, true),
# ("q2_initialQ_stay_retainbelief", run_q2_initialQ_stay_retainbelief, false, true),

# ("q2_initialQ_stay_turn", run_q2_initialQ_stay_turn, false, true),
# ("q2_initialQ_stay_spatial", run_q2_initialQ_stay_spatial, false, true),
# ("q2_initialQ_stay_turn_γ2", run_q2_initialQ_stay_turn_γ2, false, true),
# ("q2_initialQ_stay_spatial_γ2", run_q2_initialQ_stay_spatial_γ2, false, true),

# ("q2_decay", run_q2_decay, false, false),
# ("q2_γ2_decay", run_q2_γ2_decay, false, false),
# ("q2_retainbelief_decay", run_q2_retainbelief_decay, false, false),
# ("q2_stay_decay", run_q2_stay_decay, false, false),
# ("q2_stay_γ2_decay", run_q2_stay_γ2_decay, false, false),
# ("q2_stay_retainbelief_decay", run_q2_stay_retainbelief_decay, false, false),

# ("q2_stay_turn_decay", run_q2_stay_turn_decay, false, false),
# ("q2_stay_spatial_decay", run_q2_stay_spatial_decay, false, false),
# ("q2_stay_turn_γ2_decay", run_q2_stay_turn_γ2_decay, false, false),
# ("q2_stay_spatial_γ2_decay", run_q2_stay_spatial_γ2_decay, false, false),

# ("q2_initialQ_decay", run_q2_initialQ_decay, false, true),
# ("q2_initialQ_γ2_decay", run_q2_initialQ_γ2_decay, false, true),
# ("q2_initialQ_retainbelief_decay", run_q2_initialQ_retainbelief_decay, false, true),
# ("q2_initialQ_stay_decay", run_q2_initialQ_stay_decay, false, true),
# ("q2_initialQ_stay_γ2_decay", run_q2_initialQ_stay_γ2_decay, false, true),
# ("q2_initialQ_stay_retainbelief_decay", run_q2_initialQ_stay_retainbelief_decay, false, true),

("q2_initialQ_stay_turn_decay", run_q2_initialQ_stay_turn_decay, false, true),
# ("q2_initialQ_stay_spatial_decay", run_q2_initialQ_stay_spatial_decay, false, true),
("q2_initialQ_stay_turn_γ2_decay", run_q2_initialQ_stay_turn_γ2_decay, false, true),
# ("q2_initialQ_stay_spatial_γ2_decay", run_q2_initialQ_stay_spatial_γ2_decay, false, true),

# DEPLETION

# ("q2_leaf_depletion", run_q2_leaf_depletion, true, false),
# ("q2_leaf_γ2_depletion", run_q2_leaf_γ2_depletion, true, false),
# ("q2_leaf_retainbelief_depletion", run_q2_leaf_retainbelief_depletion, true, false),
# ("q2_leaf_stay_depletion", run_q2_leaf_stay_depletion, true, false),
# ("q2_leaf_stay_γ2_depletion", run_q2_leaf_stay_γ2_depletion, true, false),
# ("q2_leaf_stay_retainbelief_depletion", run_q2_leaf_stay_retainbelief_depletion, true, false),

# ("q2_leaf_stay_turn_depletion", run_q2_leaf_stay_turn_depletion, true, false),
# ("q2_leaf_stay_spatial_depletion", run_q2_leaf_stay_spatial_depletion, true, false),
# ("q2_leaf_stay_leafturn_depletion", run_q2_leaf_stay_leafturn_depletion, true, false),
# ("q2_leaf_stay_leafspatial_depletion", run_q2_leaf_stay_leafspatial_depletion, true, false),
# ("q2_leaf_stay_turn_leafspatial_depletion", run_q2_leaf_stay_turn_leafspatial_depletion, true, false),
# ("q2_leaf_stay_turn_leafturn_depletion", run_q2_leaf_stay_turn_leafturn_depletion, true, false),
# ("q2_leaf_stay_spatial_leafspatial_depletion", run_q2_leaf_stay_spatial_leafspatial_depletion, true, false),
# ("q2_leaf_stay_spatial_leafturn_depletion", run_q2_leaf_stay_spatial_leafturn_depletion, true, false),

# ("q2_leaf_stay_turn_γ2_depletion", run_q2_leaf_stay_turn_γ2_depletion, true, false),
# ("q2_leaf_stay_spatial_γ2_depletion", run_q2_leaf_stay_spatial_γ2_depletion, true, false),
# ("q2_leaf_stay_leafturn_γ2_depletion", run_q2_leaf_stay_leafturn_γ2_depletion, true, false),
# ("q2_leaf_stay_leafspatial_γ2_depletion", run_q2_leaf_stay_leafspatial_γ2_depletion, true, false),
# ("q2_leaf_stay_turn_leafspatial_γ2_depletion", run_q2_leaf_stay_turn_leafspatial_γ2_depletion, true, false),
# ("q2_leaf_stay_turn_leafturn_γ2_depletion", run_q2_leaf_stay_turn_leafturn_γ2_depletion, true, false),
# ("q2_leaf_stay_spatial_leafspatial_γ2_depletion", run_q2_leaf_stay_spatial_leafspatial_γ2_depletion, true, false),
# ("q2_leaf_stay_spatial_leafturn_γ2_depletion", run_q2_leaf_stay_spatial_leafturn_γ2_depletion, true, false),

# ("q2_leaf_initialQ_depletion", run_q2_leaf_initialQ_depletion, true, true),
# ("q2_leaf_initialQ_γ2_depletion", run_q2_leaf_initialQ_γ2_depletion, true, true),
# ("q2_leaf_initialQ_retainbelief_depletion", run_q2_leaf_initialQ_retainbelief_depletion, true, true),
# ("q2_leaf_initialQ_stay_depletion", run_q2_leaf_initialQ_stay_depletion, true, true),
# ("q2_leaf_initialQ_stay_γ2_depletion", run_q2_leaf_initialQ_stay_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_retainbelief_depletion", run_q2_leaf_initialQ_stay_retainbelief_depletion, true, true),

# ("q2_leaf_initialQ_stay_turn_depletion", run_q2_leaf_initialQ_stay_turn_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_depletion", run_q2_leaf_initialQ_stay_spatial_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafturn_depletion", run_q2_leaf_initialQ_stay_leafturn_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_depletion", run_q2_leaf_initialQ_stay_leafspatial_depletion, true, true),
# ("q2_leaf_initialQ_stay_turn_leafspatial_depletion", run_q2_leaf_initialQ_stay_turn_leafspatial_depletion, true, true),
# ("q2_leaf_initialQ_stay_turn_leafturn_depletion", run_q2_leaf_initialQ_stay_turn_leafturn_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_depletion", run_q2_leaf_initialQ_stay_spatial_leafspatial_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_depletion", run_q2_leaf_initialQ_stay_spatial_leafturn_depletion, true, true),

# ("q2_leaf_initialQ_stay_turn_γ2_depletion", run_q2_leaf_initialQ_stay_turn_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_γ2_depletion", run_q2_leaf_initialQ_stay_spatial_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafturn_γ2_depletion", run_q2_leaf_initialQ_stay_leafturn_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_γ2_depletion", run_q2_leaf_initialQ_stay_leafspatial_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_turn_leafspatial_γ2_depletion", run_q2_leaf_initialQ_stay_turn_leafspatial_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_turn_leafturn_γ2_depletion", run_q2_leaf_initialQ_stay_turn_leafturn_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_γ2_depletion", run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_γ2_depletion", run_q2_leaf_initialQ_stay_spatial_leafturn_γ2_depletion, true, true),

# ("q2_leaf_decay_depletion", run_q2_leaf_decay_depletion, true, false),
# ("q2_leaf_γ2_decay_depletion", run_q2_leaf_γ2_decay_depletion, true, false),
# ("q2_leaf_retainbelief_decay_depletion", run_q2_leaf_retainbelief_decay_depletion, true, false),
# ("q2_leaf_stay_decay_depletion", run_q2_leaf_stay_decay_depletion, true, false),
# ("q2_leaf_stay_γ2_decay_depletion", run_q2_leaf_stay_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_retainbelief_decay_depletion", run_q2_leaf_stay_retainbelief_decay_depletion, true, false),

# ("q2_leaf_stay_turn_decay_depletion", run_q2_leaf_stay_turn_decay_depletion, true, false),
# ("q2_leaf_stay_spatial_decay_depletion", run_q2_leaf_stay_spatial_decay_depletion, true, false),
# ("q2_leaf_stay_leafturn_decay_depletion", run_q2_leaf_stay_leafturn_decay_depletion, true, false),
# ("q2_leaf_stay_leafspatial_decay_depletion", run_q2_leaf_stay_leafspatial_decay_depletion, true, false),
# ("q2_leaf_stay_turn_leafspatial_decay_depletion", run_q2_leaf_stay_turn_leafspatial_decay_depletion, true, false),
# ("q2_leaf_stay_turn_leafturn_decay_depletion", run_q2_leaf_stay_turn_leafturn_decay_depletion, true, false),
# ("q2_leaf_stay_spatial_leafspatial_decay_depletion", run_q2_leaf_stay_spatial_leafspatial_decay_depletion, true, false),
# ("q2_leaf_stay_spatial_leafturn_decay_depletion", run_q2_leaf_stay_spatial_leafturn_decay_depletion, true, false),

# ("q2_leaf_stay_turn_γ2_decay_depletion", run_q2_leaf_stay_turn_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_spatial_γ2_decay_depletion", run_q2_leaf_stay_spatial_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_leafturn_γ2_decay_depletion", run_q2_leaf_stay_leafturn_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_leafspatial_γ2_decay_depletion", run_q2_leaf_stay_leafspatial_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_turn_leafspatial_γ2_decay_depletion", run_q2_leaf_stay_turn_leafspatial_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_turn_leafturn_γ2_decay_depletion", run_q2_leaf_stay_turn_leafturn_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_spatial_leafspatial_γ2_decay_depletion", run_q2_leaf_stay_spatial_leafspatial_γ2_decay_depletion, true, false),
# ("q2_leaf_stay_spatial_leafturn_γ2_decay_depletion", run_q2_leaf_stay_spatial_leafturn_γ2_decay_depletion, true, false),

# ("q2_leaf_initialQ_decay_depletion", run_q2_leaf_initialQ_decay_depletion, true, true),
# ("q2_leaf_initialQ_γ2_decay_depletion", run_q2_leaf_initialQ_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_retainbelief_decay_depletion", run_q2_leaf_initialQ_retainbelief_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_decay_depletion", run_q2_leaf_initialQ_stay_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_γ2_decay_depletion", run_q2_leaf_initialQ_stay_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_retainbelief_decay_depletion", run_q2_leaf_initialQ_stay_retainbelief_decay_depletion, true, true),

# ("q2_leaf_initialQ_stay_turn_decay_depletion", run_q2_leaf_initialQ_stay_turn_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_decay_depletion", run_q2_leaf_initialQ_stay_spatial_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafturn_decay_depletion", run_q2_leaf_initialQ_stay_leafturn_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_decay_depletion", run_q2_leaf_initialQ_stay_leafspatial_decay_depletion, true, true),
("q2_leaf_initialQ_stay_turn_leafspatial_decay_depletion", run_q2_leaf_initialQ_stay_turn_leafspatial_decay_depletion, true, true),
("q2_leaf_initialQ_stay_turn_leafturn_decay_depletion", run_q2_leaf_initialQ_stay_turn_leafturn_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_decay_depletion", run_q2_leaf_initialQ_stay_spatial_leafspatial_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_decay_depletion", run_q2_leaf_initialQ_stay_spatial_leafturn_decay_depletion, true, true),

# ("q2_leaf_initialQ_stay_turn_γ2_decay_depletion", run_q2_leaf_initialQ_stay_turn_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_γ2_decay_depletion", run_q2_leaf_initialQ_stay_spatial_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafturn_γ2_decay_depletion", run_q2_leaf_initialQ_stay_leafturn_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_leafspatial_γ2_decay_depletion", run_q2_leaf_initialQ_stay_leafspatial_γ2_decay_depletion, true, true),
("q2_leaf_initialQ_stay_turn_leafspatial_γ2_decay_depletion", run_q2_leaf_initialQ_stay_turn_leafspatial_γ2_decay_depletion, true, true),
("q2_leaf_initialQ_stay_turn_leafturn_γ2_decay_depletion", run_q2_leaf_initialQ_stay_turn_leafturn_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafspatial_γ2_decay_depletion", run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2_decay_depletion, true, true),
# ("q2_leaf_initialQ_stay_spatial_leafturn_γ2_decay_depletion", run_q2_leaf_initialQ_stay_spatial_leafturn_γ2_decay_depletion, true, true),

# No Leaf
# ("q2_base_depletion", run_q2_base_depletion, false, false),
# ("q2_γ2_depletion", run_q2_γ2_depletion, false, false),
# ("q2_retainbelief_depletion", run_q2_retainbelief_depletion, false, false),
# ("q2_stay_depletion", run_q2_stay_depletion, false, false),
# ("q2_stay_γ2_depletion", run_q2_stay_γ2_depletion, false, false),
# ("q2_stay_retainbelief_depletion", run_q2_stay_retainbelief_depletion, false, false),

# ("q2_stay_turn_depletion", run_q2_stay_turn_depletion, false, false),
# ("q2_stay_spatial_depletion", run_q2_stay_spatial_depletion, false, false),
# ("q2_stay_turn_γ2_depletion", run_q2_stay_turn_γ2_depletion, false, false),
# ("q2_stay_spatial_γ2_depletion", run_q2_stay_spatial_γ2_depletion, false, false),

# ("q2_initialQ_depletion", run_q2_initialQ_depletion, false, true),
# ("q2_initialQ_γ2_depletion", run_q2_initialQ_γ2_depletion, false, true),
# ("q2_initialQ_retainbelief_depletion", run_q2_initialQ_retainbelief_depletion, false, true),
# ("q2_initialQ_stay_depletion", run_q2_initialQ_stay_depletion, false, true),
# ("q2_initialQ_stay_γ2_depletion", run_q2_initialQ_stay_γ2_depletion, false, true),
# ("q2_initialQ_stay_retainbelief_depletion", run_q2_initialQ_stay_retainbelief_depletion, false, true),

# ("q2_initialQ_stay_turn_depletion", run_q2_initialQ_stay_turn_depletion, false, true),
# ("q2_initialQ_stay_spatial_depletion", run_q2_initialQ_stay_spatial_depletion, false, true),
# ("q2_initialQ_stay_turn_γ2_depletion", run_q2_initialQ_stay_turn_γ2_depletion, false, true),
# ("q2_initialQ_stay_spatial_γ2_depletion", run_q2_initialQ_stay_spatial_γ2_depletion, false, true),

# ("q2_decay_depletion", run_q2_decay_depletion, false, false),
# ("q2_γ2_decay_depletion", run_q2_γ2_decay_depletion, false, false),
# ("q2_retainbelief_decay_depletion", run_q2_retainbelief_decay_depletion, false, false),
# ("q2_stay_decay_depletion", run_q2_stay_decay_depletion, false, false),
# ("q2_stay_γ2_decay_depletion", run_q2_stay_γ2_decay_depletion, false, false),
# ("q2_stay_retainbelief_decay_depletion", run_q2_stay_retainbelief_decay_depletion, false, false),

# ("q2_stay_turn_decay_depletion", run_q2_stay_turn_decay_depletion, false, false),
# ("q2_stay_spatial_decay_depletion", run_q2_stay_spatial_decay_depletion, false, false),
# ("q2_stay_turn_γ2_decay_depletion", run_q2_stay_turn_γ2_decay_depletion, false, false),
# ("q2_stay_spatial_γ2_decay_depletion", run_q2_stay_spatial_γ2_decay_depletion, false, false),

# ("q2_initialQ_decay_depletion", run_q2_initialQ_decay_depletion, false, true),
# ("q2_initialQ_γ2_decay_depletion", run_q2_initialQ_γ2_decay_depletion, false, true),
# ("q2_initialQ_retainbelief_decay_depletion", run_q2_initialQ_retainbelief_decay_depletion, false, true),
# ("q2_initialQ_stay_decay_depletion", run_q2_initialQ_stay_decay_depletion, false, true),
# ("q2_initialQ_stay_γ2_decay_depletion", run_q2_initialQ_stay_γ2_decay_depletion, false, true),
# ("q2_initialQ_stay_retainbelief_decay_depletion", run_q2_initialQ_stay_retainbelief_decay_depletion, false, true),

("q2_initialQ_stay_turn_decay_depletion", run_q2_initialQ_stay_turn_decay_depletion, false, true),
# ("q2_initialQ_stay_spatial_decay_depletion", run_q2_initialQ_stay_spatial_decay_depletion, false, true),
("q2_initialQ_stay_turn_γ2_decay_depletion", run_q2_initialQ_stay_turn_γ2_decay_depletion, false, true),
# ("q2_initialQ_stay_spatial_γ2_decay_depletion", run_q2_initialQ_stay_spatial_γ2_decay_depletion, false, true),
]
base_dir = "../results/q2_combined_biases"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal_combined(animal, "..")
(fn_name, fn, fn_add_leaf, fn_fit_initial_Q) = fns[fn_ind]
flag_loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info flag_loocv

function run_fn(fn_name, fn, fn_add_leaf, fn_fit_initial_Q, initial_Q, rewscaled, delay_turn_bias, flag_loocv, full)
    # Create the base filename
    fname = fn_name
    fname = fn_fit_initial_Q ? fname : fname * "_initialQ-$(initial_Q)"
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname *= "_$(animal)_combined"
    fname = full ? fname * "_full" : fname
    @info fname
    
    fn_initial_Q = nothing
    params = Dict{Symbol, Any}()
    if !fn_fit_initial_Q
        fn_initial_Q = initial_Q
        params[:initial_Q] = initial_Q
    end
    if flag_loocv
        fname_loocv = fname * "_loocv"
    
        results = load("$(base_dir)/$(fname).jld2", "results")
        results_loocv = fn(data; extended=true, initial_Q=fn_initial_Q, rewscaled, delay_turn_bias, loocv_data=results, full)
        save("$(base_dir)_loocv/$(fname_loocv).jld2", "results_loocv", results_loocv; compress=true)
    else
        results = fn(data; extended=true, initial_Q=fn_initial_Q, rewscaled, delay_turn_bias, full)
        save("$(base_dir)/$(fname).jld2", "results", results; compress=true)
        write_EM_to_mat(results, "$(base_dir)/$(fname).mat"; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias)
        Q = find_Q_vals_by_day(data, results; params, rewscaled, delay_turn_bias, add_leaf=fn_add_leaf);
        CSV.write("$(base_dir)/Q_vals_$(fname).csv.gz", Q; compress=true)
    end
end

# Function name, function, add leaf, fit initial Q, initial Q value, rewscaled, delay turn bias, flag_loocv, full
run_fn(fn_name, fn, fn_add_leaf, fn_fit_initial_Q, 0.5, false, false, flag_loocv, false)
run_fn(fn_name, fn, fn_add_leaf, fn_fit_initial_Q, 0.5, true, false, flag_loocv, false)
# run_fn(fn_name, fn, fn_add_leaf, fn_fit_initial_Q, 0.5, true, true, flag_loocv, false)