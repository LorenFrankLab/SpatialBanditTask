using Logging
using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
include("../code/hmm_partial_independence.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

function find_Q_vals_by_day(data, results; add_leaf=true, rewscaled, delay_turn_bias)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = hmm_partial_independence_lik(view(data, data.daynum .== i, :), get_contingencies(), results;
        subject=i, add_leaf=add_leaf, rewscaled=rewscaled, delay_turn_bias=delay_turn_bias, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

fns = [
("hmm_partial_independence_leaf_ρ0", run_hmm_partial_independence_leaf_ρ0, true),
("hmm_partial_independence_leaf_γ2_ρ0", run_hmm_partial_independence_leaf_γ2_ρ0, true),
("hmm_partial_independence_leaf_retainbelief_ρ0", run_hmm_partial_independence_leaf_retainbelief_ρ0, true),
("hmm_partial_independence_leaf_stay_ρ0", run_hmm_partial_independence_leaf_stay_ρ0, true),
("hmm_partial_independence_leaf_turn_ρ0", run_hmm_partial_independence_leaf_turn_ρ0, true),
("hmm_partial_independence_leaf_spatial_ρ0", run_hmm_partial_independence_leaf_spatial_ρ0, true),
("hmm_partial_independence_leaf_leafspatial_ρ0", run_hmm_partial_independence_leaf_leafspatial_ρ0, true),
("hmm_partial_independence_leaf_leafturn_ρ0", run_hmm_partial_independence_leaf_leafturn_ρ0, true),

("hmm_partial_independence_leaf_stay_γ2_ρ0", run_hmm_partial_independence_leaf_stay_γ2_ρ0, true),
("hmm_partial_independence_leaf_stay_retainbelief_ρ0", run_hmm_partial_independence_leaf_stay_retainbelief_ρ0, true),
("hmm_partial_independence_leaf_stay_turn_ρ0", run_hmm_partial_independence_leaf_stay_turn_ρ0, true),
("hmm_partial_independence_leaf_stay_spatial_ρ0", run_hmm_partial_independence_leaf_stay_spatial_ρ0, true),
("hmm_partial_independence_leaf_stay_leafturn_ρ0", run_hmm_partial_independence_leaf_stay_leafturn_ρ0, true),
("hmm_partial_independence_leaf_stay_leafspatial_ρ0", run_hmm_partial_independence_leaf_stay_leafspatial_ρ0, true),
("hmm_partial_independence_leaf_stay_turn_leafspatial_ρ0", run_hmm_partial_independence_leaf_stay_turn_leafspatial_ρ0, true),
("hmm_partial_independence_leaf_stay_turn_leafturn_ρ0", run_hmm_partial_independence_leaf_stay_turn_leafturn_ρ0, true),
("hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ0", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ0, true),
("hmm_partial_independence_leaf_stay_spatial_leafturn_ρ0", run_hmm_partial_independence_leaf_stay_spatial_leafturn_ρ0, true),

# ("hmm_partial_independence_leaf_turn_γ2_ρ0", run_hmm_partial_independence_leaf_turn_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_spatial_γ2_ρ0", run_hmm_partial_independence_leaf_spatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ0", run_hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_turn_leafturn_γ2_ρ0", run_hmm_partial_independence_leaf_turn_leafturn_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ0", run_hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ0", run_hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ0, true),

# ("hmm_partial_independence_leaf_stay_turn_γ2_ρ0", run_hmm_partial_independence_leaf_stay_turn_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_spatial_γ2_ρ0", run_hmm_partial_independence_leaf_stay_spatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_leafturn_γ2_ρ0", run_hmm_partial_independence_leaf_stay_leafturn_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ0", run_hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ0", run_hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ0", run_hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ0", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ0, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ0", run_hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ0, true),

("hmm_partial_independence_leaf_ρ1", run_hmm_partial_independence_leaf_ρ1, true),
("hmm_partial_independence_leaf_γ2_ρ1", run_hmm_partial_independence_leaf_γ2_ρ1, true),
("hmm_partial_independence_leaf_retainbelief_ρ1", run_hmm_partial_independence_leaf_retainbelief_ρ1, true),
("hmm_partial_independence_leaf_stay_ρ1", run_hmm_partial_independence_leaf_stay_ρ1, true),
("hmm_partial_independence_leaf_turn_ρ1", run_hmm_partial_independence_leaf_turn_ρ1, true),
("hmm_partial_independence_leaf_spatial_ρ1", run_hmm_partial_independence_leaf_spatial_ρ1, true),
("hmm_partial_independence_leaf_leafspatial_ρ1", run_hmm_partial_independence_leaf_leafspatial_ρ1, true),
("hmm_partial_independence_leaf_leafturn_ρ1", run_hmm_partial_independence_leaf_leafturn_ρ1, true),

("hmm_partial_independence_leaf_stay_γ2_ρ1", run_hmm_partial_independence_leaf_stay_γ2_ρ1, true),
("hmm_partial_independence_leaf_stay_retainbelief_ρ1", run_hmm_partial_independence_leaf_stay_retainbelief_ρ1, true),
("hmm_partial_independence_leaf_stay_turn_ρ1", run_hmm_partial_independence_leaf_stay_turn_ρ1, true),
("hmm_partial_independence_leaf_stay_spatial_ρ1", run_hmm_partial_independence_leaf_stay_spatial_ρ1, true),
("hmm_partial_independence_leaf_stay_leafturn_ρ1", run_hmm_partial_independence_leaf_stay_leafturn_ρ1, true),
("hmm_partial_independence_leaf_stay_leafspatial_ρ1", run_hmm_partial_independence_leaf_stay_leafspatial_ρ1, true),
("hmm_partial_independence_leaf_stay_turn_leafspatial_ρ1", run_hmm_partial_independence_leaf_stay_turn_leafspatial_ρ1, true),
("hmm_partial_independence_leaf_stay_turn_leafturn_ρ1", run_hmm_partial_independence_leaf_stay_turn_leafturn_ρ1, true),
("hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ1", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ1, true),
("hmm_partial_independence_leaf_stay_spatial_leafturn_ρ1", run_hmm_partial_independence_leaf_stay_spatial_leafturn_ρ1, true),

# ("hmm_partial_independence_leaf_turn_γ2_ρ1", run_hmm_partial_independence_leaf_turn_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_spatial_γ2_ρ1", run_hmm_partial_independence_leaf_spatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ1", run_hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_turn_leafturn_γ2_ρ1", run_hmm_partial_independence_leaf_turn_leafturn_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ1", run_hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ1", run_hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ1, true),

# ("hmm_partial_independence_leaf_stay_turn_γ2_ρ1", run_hmm_partial_independence_leaf_stay_turn_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_spatial_γ2_ρ1", run_hmm_partial_independence_leaf_stay_spatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_leafturn_γ2_ρ1", run_hmm_partial_independence_leaf_stay_leafturn_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ1", run_hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ1", run_hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ1", run_hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ1", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ1, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ1", run_hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ1, true),

("hmm_partial_independence_leaf_ρ", run_hmm_partial_independence_leaf_ρ, true),
("hmm_partial_independence_leaf_γ2_ρ", run_hmm_partial_independence_leaf_γ2_ρ, true),
("hmm_partial_independence_leaf_retainbelief_ρ", run_hmm_partial_independence_leaf_retainbelief_ρ, true),
("hmm_partial_independence_leaf_stay_ρ", run_hmm_partial_independence_leaf_stay_ρ, true),
("hmm_partial_independence_leaf_turn_ρ", run_hmm_partial_independence_leaf_turn_ρ, true),
("hmm_partial_independence_leaf_spatial_ρ", run_hmm_partial_independence_leaf_spatial_ρ, true),
("hmm_partial_independence_leaf_leafspatial_ρ", run_hmm_partial_independence_leaf_leafspatial_ρ, true),
("hmm_partial_independence_leaf_leafturn_ρ", run_hmm_partial_independence_leaf_leafturn_ρ, true),

("hmm_partial_independence_leaf_stay_γ2_ρ", run_hmm_partial_independence_leaf_stay_γ2_ρ, true),
("hmm_partial_independence_leaf_stay_retainbelief_ρ", run_hmm_partial_independence_leaf_stay_retainbelief_ρ, true),
("hmm_partial_independence_leaf_stay_turn_ρ", run_hmm_partial_independence_leaf_stay_turn_ρ, true),
("hmm_partial_independence_leaf_stay_spatial_ρ", run_hmm_partial_independence_leaf_stay_spatial_ρ, true),
("hmm_partial_independence_leaf_stay_leafturn_ρ", run_hmm_partial_independence_leaf_stay_leafturn_ρ, true),
("hmm_partial_independence_leaf_stay_leafspatial_ρ", run_hmm_partial_independence_leaf_stay_leafspatial_ρ, true),
("hmm_partial_independence_leaf_stay_turn_leafspatial_ρ", run_hmm_partial_independence_leaf_stay_turn_leafspatial_ρ, true),
("hmm_partial_independence_leaf_stay_turn_leafturn_ρ", run_hmm_partial_independence_leaf_stay_turn_leafturn_ρ, true),
("hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ, true),
("hmm_partial_independence_leaf_stay_spatial_leafturn_ρ", run_hmm_partial_independence_leaf_stay_spatial_leafturn_ρ, true),

# ("hmm_partial_independence_leaf_turn_γ2_ρ", run_hmm_partial_independence_leaf_turn_γ2_ρ, true),
# ("hmm_partial_independence_leaf_spatial_γ2_ρ", run_hmm_partial_independence_leaf_spatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ", run_hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_turn_leafturn_γ2_ρ", run_hmm_partial_independence_leaf_turn_leafturn_γ2_ρ, true),
# ("hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ", run_hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ", run_hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ, true),

# ("hmm_partial_independence_leaf_stay_turn_γ2_ρ", run_hmm_partial_independence_leaf_stay_turn_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_spatial_γ2_ρ", run_hmm_partial_independence_leaf_stay_spatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_leafturn_γ2_ρ", run_hmm_partial_independence_leaf_stay_leafturn_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ", run_hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ", run_hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ", run_hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ", run_hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ, true),

# Test against depletion code
# ("hmm_partial_independence_leaf_ρ0_depletion", run_hmm_partial_independence_leaf_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_retainbelief_ρ0_depletion", run_hmm_partial_independence_leaf_retainbelief_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_ρ0_depletion", run_hmm_partial_independence_leaf_stay_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_turn_ρ0_depletion", run_hmm_partial_independence_leaf_turn_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_spatial_ρ0_depletion", run_hmm_partial_independence_leaf_spatial_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_leafspatial_ρ0_depletion", run_hmm_partial_independence_leaf_leafspatial_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_leafturn_ρ0_depletion", run_hmm_partial_independence_leaf_leafturn_ρ0_depletion, true),

# ("hmm_partial_independence_leaf_stay_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_retainbelief_ρ0_depletion", run_hmm_partial_independence_leaf_stay_retainbelief_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_ρ0_depletion", run_hmm_partial_independence_leaf_stay_turn_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_ρ0_depletion", run_hmm_partial_independence_leaf_stay_spatial_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafturn_ρ0_depletion", run_hmm_partial_independence_leaf_stay_leafturn_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_ρ0_depletion", run_hmm_partial_independence_leaf_stay_leafspatial_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_ρ0_depletion", run_hmm_partial_independence_leaf_stay_turn_leafspatial_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_ρ0_depletion", run_hmm_partial_independence_leaf_stay_turn_leafturn_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ0_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_ρ0_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafturn_ρ0_depletion, true),

# ("hmm_partial_independence_leaf_turn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_turn_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_spatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_spatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_turn_leafturn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_turn_leafturn_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ0_depletion, true),

# ("hmm_partial_independence_leaf_stay_turn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_turn_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_spatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafturn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_leafturn_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ0_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ0_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ0_depletion, true),

# ("hmm_partial_independence_leaf_ρ1_depletion", run_hmm_partial_independence_leaf_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_retainbelief_ρ1_depletion", run_hmm_partial_independence_leaf_retainbelief_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_ρ1_depletion", run_hmm_partial_independence_leaf_stay_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_turn_ρ1_depletion", run_hmm_partial_independence_leaf_turn_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_spatial_ρ1_depletion", run_hmm_partial_independence_leaf_spatial_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_leafspatial_ρ1_depletion", run_hmm_partial_independence_leaf_leafspatial_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_leafturn_ρ1_depletion", run_hmm_partial_independence_leaf_leafturn_ρ1_depletion, true),

# ("hmm_partial_independence_leaf_stay_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_retainbelief_ρ1_depletion", run_hmm_partial_independence_leaf_stay_retainbelief_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_ρ1_depletion", run_hmm_partial_independence_leaf_stay_turn_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_ρ1_depletion", run_hmm_partial_independence_leaf_stay_spatial_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafturn_ρ1_depletion", run_hmm_partial_independence_leaf_stay_leafturn_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_ρ1_depletion", run_hmm_partial_independence_leaf_stay_leafspatial_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_ρ1_depletion", run_hmm_partial_independence_leaf_stay_turn_leafspatial_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_ρ1_depletion", run_hmm_partial_independence_leaf_stay_turn_leafturn_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ1_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_ρ1_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafturn_ρ1_depletion, true),

# ("hmm_partial_independence_leaf_turn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_turn_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_spatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_spatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_turn_leafturn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_turn_leafturn_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ1_depletion, true),

# ("hmm_partial_independence_leaf_stay_turn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_turn_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_spatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafturn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_leafturn_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ1_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ1_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ1_depletion, true),

# ("hmm_partial_independence_leaf_ρ_depletion", run_hmm_partial_independence_leaf_ρ_depletion, true),
# ("hmm_partial_independence_leaf_γ2_ρ_depletion", run_hmm_partial_independence_leaf_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_retainbelief_ρ_depletion", run_hmm_partial_independence_leaf_retainbelief_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_ρ_depletion", run_hmm_partial_independence_leaf_stay_ρ_depletion, true),
# ("hmm_partial_independence_leaf_turn_ρ_depletion", run_hmm_partial_independence_leaf_turn_ρ_depletion, true),
# ("hmm_partial_independence_leaf_spatial_ρ_depletion", run_hmm_partial_independence_leaf_spatial_ρ_depletion, true),
# ("hmm_partial_independence_leaf_leafspatial_ρ_depletion", run_hmm_partial_independence_leaf_leafspatial_ρ_depletion, true),
# ("hmm_partial_independence_leaf_leafturn_ρ_depletion", run_hmm_partial_independence_leaf_leafturn_ρ_depletion, true),

# ("hmm_partial_independence_leaf_stay_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_retainbelief_ρ_depletion", run_hmm_partial_independence_leaf_stay_retainbelief_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_ρ_depletion", run_hmm_partial_independence_leaf_stay_turn_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_ρ_depletion", run_hmm_partial_independence_leaf_stay_spatial_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafturn_ρ_depletion", run_hmm_partial_independence_leaf_stay_leafturn_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_ρ_depletion", run_hmm_partial_independence_leaf_stay_leafspatial_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_ρ_depletion", run_hmm_partial_independence_leaf_stay_turn_leafspatial_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_ρ_depletion", run_hmm_partial_independence_leaf_stay_turn_leafturn_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_ρ_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafturn_ρ_depletion, true),

# ("hmm_partial_independence_leaf_turn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_turn_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_spatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_spatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_turn_leafspatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_turn_leafturn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_turn_leafturn_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_spatial_leafspatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_spatial_leafturn_γ2_ρ_depletion, true),

# ("hmm_partial_independence_leaf_stay_turn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_turn_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_spatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafturn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_leafturn_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_leafspatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_turn_leafspatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_turn_leafturn_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafspatial_γ2_ρ_depletion, true),
# ("hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ_depletion", run_hmm_partial_independence_leaf_stay_spatial_leafturn_γ2_ρ_depletion, true),
]
base_dir = "../results/hmm_partial_independence_biases"
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