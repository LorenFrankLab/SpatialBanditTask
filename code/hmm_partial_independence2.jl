using DataFrames
using DataFramesMeta
using Chain
using LinearAlgebra
using EM
using SpecialFunctions
using Statistics
using StatsFuns
using Combinatorics

"""
get_contingencies
    n: Number of top stem configurations to use

    Taken from an analysis of the contingencies in all animal data:

    20/80 50/50 50/80 : 3
    20/20 20/80 50/50 : 3
    20/50 20/80 50/50 : 5
    20/20 50/50 50/80 : 7
    20/20 20/20 50/50 : 9
    20/50 20/50 50/50 : 11
    20/50 50/50 50/80 : 14
    20/50 20/50 20/80 : 16
    20/20 20/50 50/50 : 31
    20/20 20/20 20/80 : 35
    20/20 20/20 50/80 : 40
    20/50 20/50 50/80 : 47
    20/20 20/20 20/50 : 51
    20/50 20/80 50/80 : 74
    20/20 20/80 50/80 : 85
    20/20 20/50 20/80 : 118
    20/20 20/50 50/80 : 186

n=3 seems to provide the best AIC under most conditions
"""
function get_contingencies(;n=3)
    base_contingency_list =[
        [[20, 20], [20, 50], [50, 80]], # 186
        [[20, 20], [20, 50], [20, 80]], # 188
        [[20, 20], [20, 80], [50, 80]], # 85
        [[20, 50], [20, 80], [50, 80]], # 74
        [[20, 20], [20, 20], [20, 50]], # 51
        [[20, 50], [20, 50], [50, 80]], # 47
        [[20, 20], [20, 20], [50, 80]], # 40
        [[20, 20], [20, 20], [20, 80]], # 35
    ][1:n]
    contingency_list = []
    for c in base_contingency_list
        for p1 in permutations(c)
            for p2 in permutations(p1[1])
                for p3 in permutations(p1[2])
                    for p4 in permutations(p1[3])
                        push!(contingency_list, vcat(p2, p3, p4))
                    end
                end
            end
        end
    end
    contingency_list = unique(contingency_list)
    return transpose(hcat(contingency_list...))/100
end

"""
Continencies manually taken from one animal
"""
function get_contingencies_observed()
    return Matrix(hcat(
        [0.2000, 0.5000, 0.5000, 0.8000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.5000, 0.5000, 0.8000],
        [0.2000, 0.2000, 0.5000, 0.2000, 0.8000, 0.5000],
        [0.2000, 0.2000, 0.5000, 0.8000, 0.2000, 0.2000],
        [0.2000, 0.5000, 0.2000, 0.2000, 0.5000, 0.8000],
        [0.5000, 0.2000, 0.8000, 0.5000, 0.2000, 0.2000],
        [0.5000, 0.8000, 0.5000, 0.2000, 0.2000, 0.2000],
        [0.8000, 0.5000, 0.2000, 0.2000, 0.5000, 0.2000],
        [0.2000, 0.2000, 0.5000, 0.2000, 0.5000, 0.8000],
        [0.2000, 0.5000, 0.8000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.5000, 0.8000, 0.2000, 0.5000, 0.2000],
        [0.5000, 0.2000, 0.2000, 0.2000, 0.2000, 0.8000],
        [0.5000, 0.8000, 0.2000, 0.2000, 0.5000, 0.2000],
        [0.8000, 0.5000, 0.2000, 0.5000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.5000, 0.8000, 0.2000],
        [0.2000, 0.2000, 0.8000, 0.2000, 0.5000, 0.2000],
        [0.2000, 0.5000, 0.8000, 0.5000, 0.2000, 0.2000],
        [0.5000, 0.2000, 0.2000, 0.2000, 0.8000, 0.2000],
        [0.5000, 0.8000, 0.2000, 0.2000, 0.2000, 0.5000],
        [0.8000, 0.2000, 0.5000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.8000, 0.5000],
        [0.2000, 0.2000, 0.2000, 0.5000, 0.8000, 0.5000],
        [0.2000, 0.2000, 0.2000, 0.8000, 0.5000, 0.2000],
        [0.2000, 0.2000, 0.5000, 0.2000, 0.2000, 0.8000],
        [0.2000, 0.2000, 0.5000, 0.8000, 0.2000, 0.5000],
        [0.2000, 0.2000, 0.5000, 0.8000, 0.5000, 0.2000],
        [0.2000, 0.2000, 0.8000, 0.5000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.8000, 0.5000, 0.5000, 0.2000],
        [0.2000, 0.5000, 0.2000, 0.2000, 0.2000, 0.8000],
        [0.2000, 0.5000, 0.2000, 0.2000, 0.8000, 0.5000],
        [0.2000, 0.8000, 0.2000, 0.2000, 0.2000, 0.5000],
        [0.5000, 0.2000, 0.2000, 0.2000, 0.5000, 0.8000],
        [0.5000, 0.2000, 0.2000, 0.2000, 0.8000, 0.5000],
        [0.5000, 0.2000, 0.2000, 0.5000, 0.8000, 0.2000],
        [0.5000, 0.2000, 0.2000, 0.8000, 0.2000, 0.2000],
        [0.5000, 0.2000, 0.2000, 0.8000, 0.5000, 0.2000],
        [0.5000, 0.2000, 0.5000, 0.2000, 0.2000, 0.8000],
        [0.5000, 0.2000, 0.5000, 0.8000, 0.2000, 0.2000],
        [0.5000, 0.2000, 0.8000, 0.2000, 0.2000, 0.2000],
        [0.5000, 0.8000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.8000, 0.2000, 0.2000, 0.2000, 0.2000, 0.5000],
        [0.8000, 0.5000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.8000, 0.5000, 0.2000, 0.2000, 0.2000, 0.5000],
        [0.8000, 0.5000, 0.5000, 0.2000, 0.2000, 0.2000]
        )')
end
function get_contingencies_base()
    return Matrix(hcat(
        [0.8000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.8000, 0.2000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.8000, 0.2000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.8000, 0.2000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.8000, 0.2000],
        [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.8000],     
        )')
end

import Base.length
length(df::U) where U <: DataFrame = size(df, 1)
length(df::U) where U <: SubDataFrame = size(df, 1)

function hmm_partial_independence2_lik_stem_inner!(Qstem, Q, prevs, prevl, βgo, βstay, stay_bias, γ2)
    Qstem .= βgo .* mean.(Q)
    
    # stay bias
    if (prevs > 0)
        Qstem[prevs] = βstay * ((1 - γ2) * Q[prevs][3 - prevl] + γ2 * Q[prevs][prevl]) + stay_bias
    end
end

function hmm_partial_independence2_lik_turn_inner!(Qstem, prevs, turn_bias, spatial_bias)
    Qstem[mod1(prevs + 1, 3)] += turn_bias
    Qstem[mod1(prevs + 1, 3)] += spatial_bias[prevs]
end

function hmm_partial_independence2_lik_leaf_inner!(Qleaf, Q, stemchoice, βleaf, leaf_turn_bias, leaf_spatial_bias)
    # Leaf choice
    Qleaf .= Q[stemchoice]
    Qleaf .= Qleaf .* βleaf
    # Add turn bias
    Qleaf[1] += leaf_turn_bias
    # Add per-stem turn biases
    Qleaf[1] += leaf_spatial_bias[stemchoice]
end

leafpairs = Dict(
    1=>2,
    2=>1,
    3=>4,
    4=>3,
    5=>6,
    6=>5,
)

"""
hmm_partial_independence2_lik

HMM Likelihood function

Parameterizes a balance between two fully
separate joint and independent HMMs.

ϕ: nstates x 6 emission probabilities
volatility<float>: Assumed chance of a switch
ρ<float>: Weighting of joint vs. independent estimation
βgo<float>: Scaling for alternate stems
βstay<float>: Scaling for current stem
βleaf<float>: Beta weight for leaf choice softmax
turn_bias<float>: Offset added to (leftward?) choice
spatial_bias<[float, float, float]>: Per-stem offset added to (leftward?) choice
leaf_turn_bias<float>: Offset added to 'first' leaf on entering a new stem
leaf_spatial_bias<[float, float, float]>: Per-stem leaf turn bias
γ2<float>: Fraction of current stem's value derived from leaf we're leaving (vs. leaf we're going to)
depletion_factor<float>: Fraction of value retained when remaining at the same leaf for multiple trials
retain_belief<float>: Fraction of belief state carried over between sessions
rewscaled<bool>: If true, reward is +1/-1 instead of 0/1
add_leaf<bool>: Whether to include likelihood for the leaf choice on a stem switch
record<bool>: Whether to return a record of estimated Q-values and entropy measures

Returns:
    negative likelihood
If 'record':
    Q: Inferred leaf Q-values at the start of each trial, ignoring biases
    Qstem: Stem Q-values at each trial, including biases
    state_entropy
    reward_entropy
"""
function hmm_partial_independence2_lik(df, ϕ::Array{Float64, 2}, volatility, ρ, βgo, βstay::V, βleaf, stay_bias::W, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, depletion_factor, retain_belief, delay_turn_bias::Bool, rewscaled::Bool, add_leaf::Bool, record::Bool) where {V, W}
    U = promote_type(eltype(βstay), eltype(stay_bias))  # this is a bit of a hack so that we can optionally have either of these fixed at 0
    # Re-organize so 
    ϕ1 = ϕ
    ϕ2 = ϕ[:,1]

    nstates = size(ϕ, 1)
    ntrials = length(df)

    statePrior = ones(U,nstates) * 1/nstates

    # volatility and state-state transition matrix T
    T = ones(U,nstates,nstates) * volatility / (nstates - 1)
    T[diagind(T)] .= 1 - volatility

    lik::U = 0.

    α = zeros(U, nstates)
    α1 = zeros(U, nstates)
    α2 = zeros(U, nstates)
    α3 = zeros(U, nstates)
    α4 = zeros(U, nstates)
    α5 = zeros(U, nstates)
    α6 = zeros(U, nstates)
    αtemp = zeros(U, nstates)
    α1temp = zeros(U, nstates)
    α2temp = zeros(U, nstates)
    α3temp = zeros(U, nstates)
    α4temp = zeros(U, nstates)
    α5temp = zeros(U, nstates)
    α6temp = zeros(U, nstates)
    # Really a 3x2 matrix, split up here for speed
    Q = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Qstem = zeros(U, 3)
    Qleaf = zeros(U, 2)
    # Qeff = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Qstem = zeros(U, 3)
    Qtemp = zeros(U, 6)
    ϕ1T = ϕ1'
    ϕ2T = ϕ2'
    if record
        Q_record = zeros(ntrials, 6)
        Qstem_record = zeros(ntrials, 3)
        Qleaf_record = zeros(ntrials, 2)
        state_entropy = zeros(ntrials)
        reward_entropy = zeros(ntrials)
        reward_entropy_leaf_1 = zeros(ntrials)
        reward_entropy_leaf_2 = zeros(ntrials)
        reward_entropy_leaf_3 = zeros(ntrials)
        reward_entropy_leaf_4 = zeros(ntrials)
        reward_entropy_leaf_5 = zeros(ntrials)
        reward_entropy_leaf_6 = zeros(ntrials)
        reward_entropy_current_leaf = zeros(ntrials)
        reward_entropy_pair_prev_leaf = Vector{Union{Float64, Missing}}(undef, ntrials)
        reward_entropy_pair_current_leaf = zeros(ntrials)
        stem_1_p = zeros(ntrials)
        stem_2_p = zeros(ntrials)
        stem_3_p = zeros(ntrials)
        stem_stay_p = zeros(ntrials)
        stem_go_turn_p = zeros(ntrials)
        stem_turn_alone_p = zeros(ntrials)
        leaf_1_p = zeros(ntrials)
        leaf_2_p = zeros(ntrials)
        stem_1_var = zeros(ntrials)
        stem_2_var = zeros(ntrials)
        stem_3_var = zeros(ntrials)
        leaf_1_var = zeros(ntrials)
        leaf_2_var = zeros(ntrials)
        stem_stay_var = zeros(ntrials)
        stem_go_turn_var = zeros(ntrials)
        stem_turn_alone_var = zeros(ntrials)
    end
    lp_stay = 0  # keep track of these so we can use them for stem_stay_p etc
    lp_go_turn = 0
    lp_turn_alone = 0
    i = 1

    depletion = ones(U, 6)

    # (dates, sessions, leaf, leafchoice, stemchoice, reward) = hmm_partial_independence2_lik_extract_df(df)
    dates = df.daynum
    sessions = df.daysessionnum
    leaf = df.leaf
    leafchoice = df.leafchoice
    stemchoice = df.stemchoice
    if rewscaled
        reward = df.rewscaled
    else
        reward = df.reward
    end

    for date in unique(dates)
        d_sessions = sessions[dates .== date]
        α .= statePrior
        α1 .= statePrior
        α2 .= statePrior
        α3 .= statePrior
        α4 .= statePrior
        α5 .= statePrior
        α6 .= statePrior
        for session in unique(d_sessions)
            # Subset of trials for the session/date
            t_ind = findall((dates .== date) .& (sessions .== session))
            # α is the joint probability of all observations, and the state at time t
            # For predictive purposes, α at time t is prior to the outcome information for trial t            

            # Initial α state
            # We're resetting this to the prior for each session
            # α(z_1k) = π_k p(x_1 | ϕ_k)
            α .= (retain_belief .* α) .+ ((1 - retain_belief) .* statePrior)
            α1 .= (retain_belief .* α1) .+ ((1 - retain_belief) .* statePrior)
            α2 .= (retain_belief .* α2) .+ ((1 - retain_belief) .* statePrior)
            α3 .= (retain_belief .* α3) .+ ((1 - retain_belief) .* statePrior)
            α4 .= (retain_belief .* α4) .+ ((1 - retain_belief) .* statePrior)
            α5 .= (retain_belief .* α5) .+ ((1 - retain_belief) .* statePrior)
            α6 .= (retain_belief .* α6) .+ ((1 - retain_belief) .* statePrior)
            prevs::Int = 0
            prevl::Int = 0
            # If rewarded, then emission = reward probs, otherwise 1 - reward probs
            # Note that we're normalizing α with a scaling factor (Bishop, p.627)
            # Since we only care about marginal belief at the current timestep,
            # no need to compute c

            # reset depletion
            depletion .= 1
            @inbounds for t in t_ind
                # Q = expectation over states
                if rewscaled
                    mul!(Qtemp, ϕ1T, α)
                    Qtemp *= ρ
                    Qtemp[1] += (1 - ρ) * ϕ2T * α1
                    Qtemp[2] += (1 - ρ) * ϕ2T * α2
                    Qtemp[3] += (1 - ρ) * ϕ2T * α3
                    Qtemp[4] += (1 - ρ) * ϕ2T * α4
                    Qtemp[5] += (1 - ρ) * ϕ2T * α5
                    Qtemp[6] += (1 - ρ) * ϕ2T * α6
                    Qtemp .*= depletion .* 2.0
                    Qtemp .-= 1.0
                else
                    mul!(Qtemp, ϕ1T, α)
                    Qtemp *= ρ
                    Qtemp[1] += (1 - ρ) * ϕ2T * α1
                    Qtemp[2] += (1 - ρ) * ϕ2T * α2
                    Qtemp[3] += (1 - ρ) * ϕ2T * α3
                    Qtemp[4] += (1 - ρ) * ϕ2T * α4
                    Qtemp[5] += (1 - ρ) * ϕ2T * α5
                    Qtemp[6] += (1 - ρ) * ϕ2T * α6
                    Qtemp .*= depletion
                end
                Q[1][1] = Qtemp[1]
                Q[1][2] = Qtemp[2]
                Q[2][1] = Qtemp[3]
                Q[2][2] = Qtemp[4]
                Q[3][1] = Qtemp[5]
                Q[3][2] = Qtemp[6]
                hmm_partial_independence2_lik_stem_inner!(Qstem, Q, prevs, prevl, βgo, βstay, stay_bias, γ2)
                if (prevs > 0)
                    # Probability of stay/switch
                    # mod1(prevs + 1, 3) and mod1(prevs + 2, 3) give us the other two stems
                    # log(exp(Q_x) / sum(exp.(Q))) -> Q_x - logsumexp(Q)
                    if delay_turn_bias  # Whether we add on the turn bias before or after we calculate stay/go
                        lp_stay = Qstem[prevs] - logsumexp(view(Qstem, :))
                        lp_go = logsumexp(view(Qstem, [mod1(prevs + 1, 3), mod1(prevs + 2, 3)])) - logsumexp(view(Qstem, :))
                        hmm_partial_independence2_lik_turn_inner!(Qstem, prevs, turn_bias, spatial_bias)
                    else
                        hmm_partial_independence2_lik_turn_inner!(Qstem, prevs, turn_bias, spatial_bias)
                        lp_stay = Qstem[prevs] - logsumexp(view(Qstem, :))
                        lp_go = logsumexp(view(Qstem, [mod1(prevs + 1, 3), mod1(prevs + 2, 3)])) - logsumexp(view(Qstem, :))
                    end
                    # Now probability is either p(stay) or p(go) * p(left/right)
                    if prevs == stemchoice[t]
                        lik += lp_stay
                    else
                        lp_turn_alone = Qstem[stemchoice[t]] - logsumexp(Qstem[[mod1(prevs + 1, 3), mod1(prevs + 2, 3)]])
                        lp_go_turn = lp_go + lp_turn_alone
                        lik += lp_go_turn
                    end
                else  # First trial, just a three-way choice
                    lik += Qstem[stemchoice[t]] - logsumexp(Qstem)
                end

                if (add_leaf && (prevs != stemchoice[t]))  # If stem switch, add leaf bias
                    hmm_partial_independence2_lik_leaf_inner!(Qleaf, Q, stemchoice[t], βleaf, leaf_turn_bias, leaf_spatial_bias)
                    lik += Qleaf[leafchoice[t]] - logsumexp(Qleaf)
                end

                # Everything below is just for meta-info on decision variables
                # Note that this is after we've calculated Q-values, but before we've updated contingencies
                if record
                    stem_1_p[i] = exp(Qstem[1] - logsumexp(Qstem))
                    stem_2_p[i] = exp(Qstem[2] - logsumexp(Qstem))
                    stem_3_p[i] = exp(Qstem[3] - logsumexp(Qstem))
                    if (add_leaf && (prevs != stemchoice[t]))
                        leaf_1_p[i] = exp(Qleaf[1] - logsumexp(Qleaf))
                        leaf_2_p[i] = exp(Qleaf[2] - logsumexp(Qleaf))
                    end
                    if (prevs > 0)
                        stem_stay_p[i] = exp(lp_stay)
                        if prevs != stemchoice[t]
                            stem_go_turn_p[i] = exp(lp_go_turn)
                            stem_turn_alone_p[i] = exp(lp_turn_alone)
                        end
                    end

                    Q_record[i, :] .= Qtemp
                    Qstem_record[i, :] .= Qstem
                    Qleaf_record[i, :] .= Qleaf
                    state_entropy[i] = -sum(α .* log.(α))

                    # First, find the probability of each leaf being .2/.5/.8
                    # Next find entropy of that dist, and mean across all 6 leaves
                    # Large extra allocations happening here
                    reward_probs = ρ .* vcat([
                        sum((ϕ1 .== .2) .* α; dims=1)
                        sum((ϕ1 .== .5) .* α; dims=1)
                        sum((ϕ1 .== .8) .* α; dims=1)
                        ])
                    αs = [α1 α2 α3 α4 α5 α6]
                    reward_probs .+= (1 - ρ) .* vcat([
                        sum((ϕ2 .== .2) .* αs; dims=1)
                        sum((ϕ2 .== .5) .* αs; dims=1)
                        sum((ϕ2 .== .8) .* αs; dims=1)
                        ])
                    # First is entropy across all leaves
                    reward_entropy[i] = sum([-sum(reward_probs[:,j] .* log.(reward_probs[:,j])) for j in 1:6])
                    # Now individual leaves - this represents epistemic uncertainty
                    # e.g. uncertainty about the true mean, NOT uncertainty about expected reward
                    reward_entropy_leaf_1[i] = -sum(reward_probs[:,1] .* log.(reward_probs[:,1]))
                    reward_entropy_leaf_2[i] = -sum(reward_probs[:,2] .* log.(reward_probs[:,2]))
                    reward_entropy_leaf_3[i] = -sum(reward_probs[:,3] .* log.(reward_probs[:,3]))
                    reward_entropy_leaf_4[i] = -sum(reward_probs[:,4] .* log.(reward_probs[:,4]))
                    reward_entropy_leaf_5[i] = -sum(reward_probs[:,5] .* log.(reward_probs[:,5]))
                    reward_entropy_leaf_6[i] = -sum(reward_probs[:,6] .* log.(reward_probs[:,6]))
                    # Entropy of the chosen leaf, before we observe it
                    reward_entropy_current_leaf[i] = -sum(reward_probs[:,leaf[t]] .* log.(reward_probs[:,leaf[t]]))
                    # Entropy of the pair of our previous leaf, or where we would end up without a stem switch
                    if t-1 in t_ind
                        reward_entropy_pair_prev_leaf[i] = -sum(reward_probs[:,leafpairs[leaf[t-1]]] .* log.(reward_probs[:,leafpairs[leaf[t-1]]]))
                    else
                        reward_entropy_pair_prev_leaf[i] = missing
                    end
                    # Entropy of the pair leaf of the chosen leaf (unclear utility of this)
                    reward_entropy_pair_current_leaf[i] = -sum(reward_probs[:,leafpairs[leaf[t]]] .* log.(reward_probs[:,leafpairs[leaf[t]]]))

                    state_probs = zeros(729)
                    stem_1_choice_probs = zeros(729)
                    stem_2_choice_probs = zeros(729)
                    stem_3_choice_probs = zeros(729)
                    stem_stay_choice_probs = zeros(729)
                    stem_go_turn_choice_probs = zeros(729)
                    stem_turn_alone_choice_probs = zeros(729)
                    leaf_1_choice_probs = zeros(729)
                    leaf_2_choice_probs = zeros(729)
                    for i in 1:729
                        i1 = mod1(i, 3)
                        i2 = mod1(fld1(i,3), 3)
                        j1 = mod1(fld1(i,9), 3)
                        j2 = mod1(fld1(i,27), 3)
                        k1 = mod1(fld1(i,81), 3)
                        k2 = mod1(fld1(i,243), 3)
                        p11 = reward_probs[i1, 1]
                        p12 = reward_probs[i2, 2]
                        p21 = reward_probs[j1, 3]
                        p22 = reward_probs[j2, 4]
                        p31 = reward_probs[k1, 5]
                        p32 = reward_probs[k2, 6]
                        Q[1][1] = [0.2, 0.5, 0.8][i1] * depletion[1]
                        Q[1][2] = [0.2, 0.5, 0.8][i2] * depletion[2]
                        Q[2][1] = [0.2, 0.5, 0.8][j1] * depletion[3]
                        Q[2][2] = [0.2, 0.5, 0.8][j2] * depletion[4]
                        Q[3][1] = [0.2, 0.5, 0.8][k1] * depletion[5]
                        Q[3][2] = [0.2, 0.5, 0.8][k2] * depletion[6]

                        state_probs[i] = p11 * p12 * p21 * p22 * p31 * p32
    
                        hmm_partial_independence2_lik_stem_inner!(Qstem, Q, prevs, prevl, βgo, βstay, stay_bias, γ2)
                        if (prevs > 0)
                            # Probability of stay/switch
                            if delay_turn_bias
                                lp_stay = Qstem[prevs] - logsumexp(view(Qstem, :))
                                lp_go = logsumexp(view(Qstem, [mod1(prevs + 1, 3), mod1(prevs + 2, 3)])) - logsumexp(view(Qstem, :))
                                hmm_partial_independence2_lik_turn_inner!(Qstem, prevs, turn_bias, spatial_bias)
                            else
                                hmm_partial_independence2_lik_turn_inner!(Qstem, prevs, turn_bias, spatial_bias)
                                lp_stay = Qstem[prevs] - logsumexp(view(Qstem, :))
                                lp_go = logsumexp(view(Qstem, [mod1(prevs + 1, 3), mod1(prevs + 2, 3)])) - logsumexp(view(Qstem, :))
                            end
                            stem_stay_choice_probs[i] = lp_stay
                            if prevs != stemchoice[t]
                                stem_go_turn_choice_probs[i] = lp_go + Qstem[stemchoice[t]] - logsumexp(Qstem[[mod1(prevs + 1, 3), mod1(prevs + 2, 3)]])
                                stem_turn_alone_choice_probs[i] = Qstem[stemchoice[t]] - logsumexp(Qstem[[mod1(prevs + 1, 3), mod1(prevs + 2, 3)]])
                            end
                        end
                        # Assuming a three-way choice
                        stem_1_choice_probs[i] = Qstem[1] - logsumexp(Qstem)
                        stem_2_choice_probs[i] = Qstem[2] - logsumexp(Qstem)
                        stem_3_choice_probs[i] = Qstem[3] - logsumexp(Qstem)
                        if (add_leaf && (prevs != stemchoice[t]))
                            hmm_partial_independence2_lik_leaf_inner!(Qleaf, Q, stemchoice[t], βleaf, leaf_turn_bias, leaf_spatial_bias)
                            leaf_1_choice_probs[i] = Qleaf[1] - logsumexp(Qleaf)
                            leaf_2_choice_probs[i] = Qleaf[2] - logsumexp(Qleaf)
                        end
                    end
                    exp_stem_1_choice_probs = exp.(stem_1_choice_probs)
                    exp_stem_2_choice_probs = exp.(stem_2_choice_probs)
                    exp_stem_3_choice_probs = exp.(stem_3_choice_probs)
                    exp_stem_stay_choice_probs = exp.(stem_stay_choice_probs)
                    exp_stem_go_turn_choice_probs = exp.(stem_go_turn_choice_probs)
                    exp_stem_turn_alone_choice_probs = exp.(stem_turn_alone_choice_probs)
                    exp_leaf_1_choice_probs = exp.(leaf_1_choice_probs)
                    exp_leaf_2_choice_probs = exp.(leaf_2_choice_probs)
                    stem_1_mu = sum(state_probs .* exp_stem_1_choice_probs)
                    stem_2_mu = sum(state_probs .* exp_stem_2_choice_probs)
                    stem_3_mu = sum(state_probs .* exp_stem_3_choice_probs)
                    stem_stay_mu = sum(state_probs .* exp_stem_stay_choice_probs)
                    stem_go_turn_mu = sum(state_probs .* exp_stem_go_turn_choice_probs)
                    stem_turn_alone_mu = sum(state_probs .* exp_stem_turn_alone_choice_probs)
                    leaf_1_mu = sum(state_probs .* exp_leaf_1_choice_probs)
                    leaf_2_mu = sum(state_probs .* exp_leaf_2_choice_probs)
                    for j in 1:729
                        stem_1_var[i] += state_probs[j] * (exp_stem_1_choice_probs[j] - stem_1_mu)^2
                        stem_2_var[i] += state_probs[j] * (exp_stem_2_choice_probs[j] - stem_2_mu)^2
                        stem_3_var[i] += state_probs[j] * (exp_stem_3_choice_probs[j] - stem_3_mu)^2
                        stem_stay_var[i] += state_probs[j] * (exp_stem_stay_choice_probs[j] - stem_stay_mu)^2
                        stem_go_turn_var[i] += state_probs[j] * (exp_stem_go_turn_choice_probs[j] - stem_go_turn_mu)^2
                        stem_turn_alone_var[i] += state_probs[j] * (exp_stem_turn_alone_choice_probs[j] - stem_turn_alone_mu)^2
                        leaf_1_var[i] += state_probs[j] * (exp_leaf_1_choice_probs[j] - leaf_1_mu)^2
                        leaf_2_var[i] += state_probs[j] * (exp_leaf_2_choice_probs[j] - leaf_2_mu)^2
                    end
                end

                # Update state prediction
                mul!(αtemp, T, α)
                mul!(α1temp, T, α1)
                mul!(α2temp, T, α2)
                mul!(α3temp, T, α3)
                mul!(α4temp, T, α4)
                mul!(α5temp, T, α5)
                mul!(α6temp, T, α6)
                α .= αtemp
                α1 .= α1temp
                α2 .= α2temp
                α3 .= α3temp
                α4 .= α4temp
                α5 .= α5temp
                α6 .= α6temp

                # Does the depletion matter at all? Or does it get normalized out?
                # ϕ_depleted = ϕ .* repeat(depletion, 1, 72)'
                if (reward[t] == 1)
                    α .*= view(ϕ1, :, leaf[t]) .* depletion[leaf[t]]
                    if leaf[t] == 1
                        α1 .*= ϕ2 .* depletion[leaf[t]]
                    elseif leaf[t] == 2
                        α2 .*= ϕ2 .* depletion[leaf[t]]
                    elseif leaf[t] == 3
                        α3 .*= ϕ2 .* depletion[leaf[t]]
                    elseif leaf[t] == 4
                        α4 .*= ϕ2 .* depletion[leaf[t]]
                    elseif leaf[t] == 5
                        α5 .*= ϕ2 .* depletion[leaf[t]]
                    else
                        α6 .*= ϕ2 .* depletion[leaf[t]]
                    end
                else
                    αtemp .= view(ϕ1, :, leaf[t]) .* depletion[leaf[t]]
                    α .*= 1 .- αtemp
                    if leaf[t] == 1
                        α1temp .= ϕ2 .* depletion[leaf[t]]
                        α1 .*= 1 .- α1temp
                    elseif leaf[t] == 2
                        α2temp .= ϕ2 .* depletion[leaf[t]]
                        α2 .*= 1 .- α2temp
                    elseif leaf[t] == 3
                        α3temp .= ϕ2 .* depletion[leaf[t]]
                        α3 .*= 1 .- α3temp
                    elseif leaf[t] == 4
                        α4temp .= ϕ2 .* depletion[leaf[t]]
                        α4 .*= 1 .- α4temp
                    elseif leaf[t] == 5
                        α5temp .= ϕ2 .* depletion[leaf[t]]
                        α5 .*= 1 .- α5temp
                    else
                        α6temp .= ϕ2 .* depletion[leaf[t]]
                        α6 .*= 1 .- α6temp
                    end
                end
                normalize!(α, 1)
                normalize!(α1, 1)
                normalize!(α2, 1)
                normalize!(α3, 1)
                normalize!(α4, 1)
                normalize!(α5, 1)
                normalize!(α6, 1)

                if (depletion_factor < 1.0)
                    if (prevs == stemchoice[t])
                        depletion[leaf[t]] *= depletion_factor
                    else
                        depletion .= 1
                    end
                end

                prevl = leafchoice[t]
                prevs = stemchoice[t]

                i += 1
            end
        end
    end

    if record
        # Construct a dataframe of all recorded Q-values, choice probabilities, etc
        record_df = DataFrame(
            # Base Q values
            Q1 = Q_record[:,1],
            Q2 = Q_record[:,2],
            Q3 = Q_record[:,3],
            Q4 = Q_record[:,4],
            Q5 = Q_record[:,5],
            Q6 = Q_record[:,6],
            # Stem values, incorporating biases
            Qstem1 = Qstem_record[:,1],
            Qstem2 = Qstem_record[:,2],
            Qstem3 = Qstem_record[:,3],
            # Leaf values, incorporating biases
            Qleaf1 = Qleaf_record[:,1],
            Qleaf2 = Qleaf_record[:,2],
            # Entropy of belief state
            state_entropy = state_entropy,
            # Entropy of reward distribution
            reward_entropy = reward_entropy,
            reward_entropy_leaf_1 = reward_entropy_leaf_1,
            reward_entropy_leaf_2 = reward_entropy_leaf_2,
            reward_entropy_leaf_3 = reward_entropy_leaf_3,
            reward_entropy_leaf_4 = reward_entropy_leaf_4,
            reward_entropy_leaf_5 = reward_entropy_leaf_5,
            reward_entropy_leaf_6 = reward_entropy_leaf_6,
            reward_entropy_current_leaf = reward_entropy_current_leaf,
            reward_entropy_pair_prev_leaf = reward_entropy_pair_prev_leaf,
            reward_entropy_pair_current_leaf = reward_entropy_pair_current_leaf,
            # Model probabilities of choosing each stem and leaf
            stem_1_p = stem_1_p,
            stem_2_p = stem_2_p,
            stem_3_p = stem_3_p,
            stem_stay_p = stem_stay_p,
            stem_go_turn_p = stem_go_turn_p,
            stem_turn_alone_p = stem_turn_alone_p,
            leaf_1_p = leaf_1_p,
            leaf_2_p = leaf_2_p,
            # Variance of choice probabilities, based on HMM state distribution
            stem_1_var = stem_1_var,
            stem_2_var = stem_2_var,
            stem_3_var = stem_3_var,
            stem_stay_var = stem_stay_var,
            stem_go_turn_var = stem_go_turn_var,
            stem_turn_alone_var = stem_turn_alone_var,
            leaf_1_var = leaf_1_var,
            leaf_2_var = leaf_2_var,
        ) 
        record_df[!, :stem_choice_p] .= 0.0
        record_df[!, :leaf_choice_p] .= 0.0
        record_df[!, :stem_choice_var] .= 0.0
        record_df[!, :leaf_choice_var] .= 0.0
        # Label trials with probability of eventual choice
        for i in 1:nrow(record_df)
            s = df.stemchoice[i]
            record_df[i, :stem_choice_p] = record_df[i, Symbol("stem_$(s)_p")]
            record_df[i, :stem_choice_var] = record_df[i, Symbol("stem_$(s)_var")]
            l = df.leafchoice[i]
            record_df[i, :leaf_choice_p] = record_df[i, Symbol("leaf_$(l)_p")]
            record_df[i, :leaf_choice_var] = record_df[i, Symbol("leaf_$(l)_var")]
        end
        return -lik, record_df
    else
        return -lik
    end
end

# Probably shouldn't use this second version in the EM code
# As julia doens't allow specialization on kw arguments?
function hmm_partial_independence2_lik(df, ϕ; volatility=1.0/30.0, ρ=0.0, βgo=0., βstay=0., βleaf=0., stay_bias=0., turn_bias=0., spatial_bias=[0., 0, 0], leaf_turn_bias=0., leaf_spatial_bias=[0., 0, 0], γ2=0., depletion_factor=1., retain_belief=0., delay_turn_bias=false, rewscaled=false, add_leaf=true, record=false)
    hmm_partial_independence2_lik(df, ϕ, volatility, ρ, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, depletion_factor, retain_belief, delay_turn_bias, rewscaled, add_leaf, record)
end

"""
HMM likelihood function using parameters from an existing EM run

Can pass in extra parameters with `params`: these will override the EM results

Note that params should be a dictionary of symbols to values, e.g. (:βgo => 2.0)

If `subject` is provided, use parameters from subject `subject`.
Otherwise use group-level betas
"""
function hmm_partial_independence2_lik(data, ϕ, results::T; subject=0, params=nothing, delay_turn_bias=false, rewscaled=false, add_leaf=true, record=false) where T <: EMResultsAbstract
    d = Dict{Symbol, Any}()
    # The trick here is that we can pass in a dictionary of (symbol => value) as kwargs
    # Then everything not present the EMResults struct is left at its default value
    if subject == 0
        for i in eachindex(results.varnames)
            d[Symbol(results.varnames[i])] = results.betas[i]
        end
    else
        for i in eachindex(results.varnames)
            d[Symbol(results.varnames[i])] = results.x[subject, i]
        end
    end

    if haskey(d, :volatility)
        d[:volatility] = unitnorm(d[:volatility])
    end
    if haskey(d, :ρ)
        d[:ρ] = unitnorm(d[:ρ])
    end
    if haskey(d, :γ2)
        d[:γ2] = unitnorm(d[:γ2])
    end
    if haskey(d, :depletion_factor)
        d[:depletion_factor] = unitnorm(d[:depletion_factor])
    end
    if haskey(d, :retain_belief)
        d[:retain_belief] = unitnorm(d[:retain_belief])
    end
    # Combine array parameters
    if haskey(d, :spatial_1)
        d[:spatial_bias] = [d[:spatial_1], d[:spatial_2], d[:spatial_3]]
        delete!(d, :spatial_1)
        delete!(d, :spatial_2)
        delete!(d, :spatial_3)
    end
    if haskey(d, :leaf_spatial_1)
        d[:leaf_spatial_bias] = [d[:leaf_spatial_1], d[:leaf_spatial_2], d[:leaf_spatial_3]]
        delete!(d, :leaf_spatial_1)
        delete!(d, :leaf_spatial_2)
        delete!(d, :leaf_spatial_3)
    end
    # Incorporate any extra parameters
    if !isnothing(params)
        for (k, v) in params
            d[k] = v
        end
    end
    hmm_partial_independence2_lik(data, ϕ; delay_turn_bias, rewscaled, add_leaf, record, d...)
end

"""
run_hmm
full: Whether to model full covariance matrix
extended: Try to calculate p-values and group-level covariance
quiet: Silence progress

ϕ: Custom set of contingencies. If unset, use default get_contingencies(n=3)
volatility: Assumed chance of a switch
ρ: 1: subset, 0: full
βgo: Scaling for alternate stems
βstay: Scaling for current stem
βleaf: Beta weight for leaf choice softmax
stay_bias: Offset added to staying at current stem
turn_bias: Offset added to (leftward?) choice
spatial_bias: Per-stem offset added to (leftward?) choice
leaf_turn_bias: Offset added to 'first' leaf on entering a new stem
leaf_spatial_bias: Per-stem leaf turn bias
γ2: Fraction of current stem's value derived from leaf we're leaving (vs. leaf we're going to)
depletion_factor: Fraction of value retained when remaining at the same leaf for multiple trials
retain_belief: Fraction of belief in HMM state to retain between sessions
rewscaled: If true, reward is +1/-1 instead of 0/1
add_leaf: Whether to include likelihood for the leaf choice on a stem switch
ρ_default: Default value for ρ if not included in the model
subjlevel: How to split the dataset - either :daynum or :daysessionnum
"""
function run_hmm_partial_independence2(df; maxiter=100, emtol=1e-3, full=true, extended=false, quiet=false,
    ϕ=nothing,
    add_volatility=true,
    add_ρ=false,
    add_βgo=true,
    add_βstay=true,
    add_βleaf=false,
    add_stay_bias=false,
    add_turn_bias=false,
    add_spatial_bias=false,
    add_leaf_turn_bias=false,
    add_leaf_spatial_bias=false,
    add_γ2=false,
    add_depletion_factor=false,
    add_retain_belief=false,
    delay_turn_bias=false,
    rewscaled=false,
    add_leaf=true,
    ρ_default=0.0,
    loocv_data=nothing,
    loocv_subject=nothing,
    subjlevel=:daynum,
    )

    @show add_volatility
    @show add_ρ
    @show add_βgo
    @show add_βstay
    @show add_βleaf
    @show add_stay_bias
    @show add_turn_bias
    @show add_spatial_bias
    @show add_leaf_turn_bias
    @show add_leaf_spatial_bias
    @show add_γ2
    @show add_factor
    @show add_retain_belief
    @show delay_turn_bias
    @show rewscaled
    @show add_leaf
    @show ρ_default
    @show loocv_subject
    @show subjlevel

    data = copy(df)
    data[:, :sub] = data[:, subjlevel]
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...

    initbetas = Matrix{Float64}(undef, 1, 0)
    initsigma = Vector{Float64}(undef, 0)
    varnames = Vector{String}(undef, 0)

    if add_volatility
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "volatility")
    end
    if add_ρ
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "ρ")
    end
    if add_βgo
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 5)
        push!(varnames, "βgo")
    end
    if add_βstay
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 5)
        push!(varnames, "βstay")
    end
    if add_βleaf
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 5)
        push!(varnames, "βleaf")
    end
    if add_stay_bias
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 5)
        push!(varnames, "stay_bias")
    end
    if add_turn_bias
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "turn_bias")
    end
    if add_spatial_bias
        initbetas = hcat(initbetas, 0, 0, 0)
        initsigma = vcat(initsigma, 1, 1, 1)
        varnames = vcat(varnames, "spatial_1", "spatial_2", "spatial_3")
    end
    if add_leaf_turn_bias
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "leaf_turn_bias")
    end
    if add_leaf_spatial_bias
        initbetas = hcat(initbetas, 0, 0, 0)
        initsigma = vcat(initsigma, 1, 1, 1)
        varnames = vcat(varnames, "leaf_spatial_1", "leaf_spatial_2", "leaf_spatial_3")
    end
    if add_γ2
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "γ2")
    end
    if add_depletion_factor
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "depletion_factor")
    end
    if add_retain_belief
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "retain_belief")
    end

    @show varnames

    function fn(params, data)
        if ϕ === nothing
            ϕ = get_contingencies()
        end

        i = 1
    
        if add_volatility
            volatility = unitnorm(params[i])
            i += 1
        else
            volatility = 0.0
        end

        if add_ρ
            ρ = unitnorm(params[i]) # weight for partial independence2
            i += 1
        else
            ρ = ρ_default
        end

        if add_βgo
            βgo = params[i] # beta for go choice
            i += 1
        else
            βgo = 0.0
        end

        if add_βstay
            βstay = params[i] # beta for stay choice
            i += 1
        else
            βstay = 0.0
        end

        if add_βleaf
            βleaf = params[i] # beta for leaf choice on switch
            i += 1
        else
            βleaf = 0.0
        end

        if add_stay_bias
            stay_bias = params[i] # beta for leaf choice on switch
            i += 1
        else
            stay_bias = 0.0
        end

        if add_turn_bias
            turn_bias = params[i] # beta for leaf choice on switch
            i += 1
        else
            turn_bias = 0.0
        end

        if add_spatial_bias
            spatial_bias = params[i:i+2] # beta for leaf choice on switch
            i += 3
        else
            spatial_bias = [0.0, 0.0, 0.0]
        end

        if add_leaf_turn_bias
            leaf_turn_bias = params[i] # beta for leaf choice on switch
            i += 1
        else
            leaf_turn_bias = 0.0
        end

        if add_leaf_spatial_bias
            leaf_spatial_bias = params[i:i+2] # beta for leaf choice on switch
            i += 3
        else
            leaf_spatial_bias = [0.0, 0.0, 0.0]
        end

        if add_γ2
            γ2 = unitnorm(params[i])
            i += 1
        else
            γ2 = 0.0
        end

        if add_depletion_factor
            depletion_factor = unitnorm(params[i])
            i += 1
        else
            depletion_factor = 1.0
        end

        if add_retain_belief
            retain_belief = unitnorm(params[i])
            i += 1
        else
            retain_belief = 0.0
        end


        return hmm_partial_independence2_lik(data, ϕ, volatility, ρ, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, depletion_factor, retain_belief, delay_turn_bias, rewscaled, add_leaf, false)
    end

    if !isnothing(loocv_data)
        if !isnothing(loocv_subject)
            return loocv_singlesubject(data,subs,loocv_subject,loocv_data.x,X,loocv_data.betas,loocv_data.sigma,fn; emtol, full, maxiter)
        else
            return loocv(data,subs,loocv_data.x,X,loocv_data.betas,loocv_data.sigma,fn; emtol, full, maxiter)
        end
    else
        (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol, full, maxiter, quiet);
        if extended
            try
                @info "Running emerrors"
                (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
                return EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
            catch err
                if isa(err, SingularException)
                    @warn "emerrors failed to run. Re-check fitting. Returning EMResults"
                    return EMResults(varnames,betas,sigma,x,l,h,opt_rec)
                else
                    rethrow()
                end
            end
        else
            return EMResults(varnames,betas,sigma,x,l,h,opt_rec)
        end
    end
end

"""
Re-runs the HMM to compute Q-values and other decision variables

Takes as input the rat data and output from EM

A few variables are not encoded in the EM results, so they are passed in as kwargs:
add_leaf: Whether to include likelihood for the leaf choice on a stem switch
rewscaled: If true, reward is +1/-1 instead of 0/1
delay_turn_bias: Whether to delay the addition of the turn bias
subjlevel: How to split the dataset - either :daynum or :daysessionnum
"""
function find_Q_vals_hmm_partial_independence2(df, results; add_leaf=true, rewscaled, delay_turn_bias, params=nothing, subjlevel=:daynum)
    data = copy(df)
    data[:, :sub] = data[:, subjlevel]
    nsubjs = maximum(data.sub)
    liks = zeros(nsubjs)
    dfs = []
    for i in 1:nsubjs
        (liks[i], df) = hmm_partial_independence2_lik(view(data, data.sub .== i, :), get_contingencies(), results;
        subject=i, add_leaf, rewscaled, delay_turn_bias, params, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

# Non-Depletion
run_hmm_partial_independence2_leaf_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_retainbelief_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_retain_belief=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_turn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_leafspatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_spatial_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_leafturn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_turn_bias=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_leaf_stay_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_retainbelief_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_leaf_turn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafspatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafturn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafspatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafturn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_leaf_stay_turn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_leaf_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_retainbelief_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_retain_belief=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_turn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_leafspatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_spatial_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_leafturn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_turn_bias=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_leaf_stay_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_retainbelief_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_leaf_turn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafspatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafturn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafspatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafturn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_leaf_stay_turn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_leaf_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_retainbelief_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_retain_belief=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_leafspatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_spatial_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_leafturn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_turn_bias=true, add_ρ=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_retainbelief_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_ρ=true, kwargs...)

run_hmm_partial_independence2_leaf_turn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafspatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafturn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafspatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafturn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_turn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)

# Depletion
run_hmm_partial_independence2_leaf_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_depletion_factor=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_leaf_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_retainbelief_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_retain_belief=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_leafspatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_leafturn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_retainbelief_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_turn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafspatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafturn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafspatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafturn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_turn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_depletion_factor=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_leaf_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_retainbelief_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_retain_belief=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_leafspatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_leafturn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_retainbelief_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_turn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafspatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafturn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafspatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafturn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_turn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_depletion_factor=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_leaf_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_retainbelief_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_retain_belief=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_leafspatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_leafturn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_leaf_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_retainbelief_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_turn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafspatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_turn_leafturn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafspatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_spatial_leafturn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_leaf_stay_turn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafturn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_leafspatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafspatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_turn_leafturn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafspatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_leaf_stay_spatial_leafturn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)

# No Leaf
run_hmm_partial_independence2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_retainbelief_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_stay_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_turn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_spatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_stay_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_stay_retainbelief_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_stay_turn_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_stay_spatial_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_turn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_spatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_stay_turn_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)
run_hmm_partial_independence2_stay_spatial_γ2_ρ0(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, kwargs...)

run_hmm_partial_independence2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_retainbelief_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_stay_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_turn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_spatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_stay_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_stay_retainbelief_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_stay_turn_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_stay_spatial_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_turn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_spatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_stay_turn_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)
run_hmm_partial_independence2_stay_spatial_γ2_ρ1(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, kwargs...)

run_hmm_partial_independence2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_ρ=true, kwargs...)
run_hmm_partial_independence2_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_retainbelief_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_stay_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_turn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_spatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_ρ=true, kwargs...)

run_hmm_partial_independence2_stay_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_stay_retainbelief_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_stay_turn_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_ρ=true, kwargs...)

run_hmm_partial_independence2_turn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_spatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_stay_turn_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_ρ=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_γ2_ρ(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_ρ=true, kwargs...)

# # Depletion
run_hmm_partial_independence2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_retainbelief_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_turn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_spatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_stay_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_retainbelief_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_turn_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_turn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_spatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_turn_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_γ2_ρ0_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=0.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_retainbelief_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_turn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_spatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_stay_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_retainbelief_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_turn_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_turn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_spatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_turn_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_γ2_ρ1_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, ρ_default=1.0, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_retainbelief_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_turn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_spatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_stay_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_retainbelief_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_turn_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_ρ=true, add_depletion_factor=true, kwargs...)

run_hmm_partial_independence2_turn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_spatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_turn_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)
run_hmm_partial_independence2_stay_spatial_γ2_ρ_depletion(data; kwargs...) = run_hmm_partial_independence2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_ρ=true, add_depletion_factor=true, kwargs...)