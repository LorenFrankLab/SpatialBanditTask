using DataFrames
using DataFramesMeta
using Chain
using CSV
using LinearAlgebra
using EM
using SpecialFunctions
using Statistics
using StatsFuns
using Combinatorics

# Including these to fix an issue with Julia 1.6
@static if (VERSION <= v"1.6")
    function normalize!(a::AbstractArray, p::Real=2)
        nrm = norm(a, p)
        __normalize!(a, nrm)
    end

    @inline function __normalize!(a::AbstractArray, nrm::Real)
        # The largest positive floating point number whose inverse is less than infinity
        δ = inv(prevfloat(typemax(nrm)))

        if nrm ≥ δ # Safe to multiply with inverse
            invnrm = inv(nrm)
            rmul!(a, invnrm)

        else # scale elements to avoid overflow
            εδ = eps(one(nrm))/δ
            rmul!(a, εδ)
            rmul!(a, inv(nrm*εδ))
        end

        a
    end
end

animals = [
    "senor",
    "chimi",
    "j16",
    "peanut",
    "wilbur",
]

function cont_combination_old(s)
    parse.(Int, split(s, "0"; keepempty=false)) ./ 10
end

function load_animal(animal; depletion=false)
    filepath = "/Users/ari/Dropbox/Princeton/SpatialBanditTask"
    
    # csv_file to estimate value for
    if depletion
        csv_file = animal * "_clean_contingencies_only_parsed_depletion_data.csv"
        # mkdir(fullfile(filepath,["hmm_decay_",animal]))
    else
        csv_file = animal * "_clean_contingencies_only_parsed_data.csv"
        # mkdir(fullfile(filepath,['hmm_',animal]))
    end
    fullpath = joinpath(filepath, "data", csv_file)
    df = DataFrame(CSV.File(fullpath, drop=[1]))  # Drop index column

    # recode some variables
    df.rewscaled = 2 * df.reward .- 1
    df.stemchoice = [df[i,:stem][1] - 'A' + 1 for i in 1:nrow(df)]
    df.leafchoice = 1 .+ mod.(df.leaf.+1,2)

    # Number days 1-n
    dates = unique(df.date)
    df.daynum = [minimum(findall(df[i,:date] .== dates)) for i in 1:nrow(df)]

    # Code sessions contiguously across days
    datesessions = string.(df.date) .* string.(df.session)
    uniq_datesessions = unique(datesessions)
    df.daysessionnum = [minimum(findall(datesessions[i] .== uniq_datesessions)) for i in 1:nrow(df)]

    # Stem switches
    df[!, :stemswitch] .= false
    prevstem = 0
    prevsession = 0
    for i in 1:size(df, 1)
        session = df[i, :session]
        if (session == prevsession)
            stem = df[i, :stem]
            if (stem != prevstem)
                df[i, :stemswitch] = true
            end
            prevstem = stem
        end
        prevsession = session
    end

    df.cont_combo = cont_combination_old.(string.(df.contingency))
    df.contingency = string.(df.contingency)

    return df
end

function load_all_data()
    dfs = []
    for animal in animals
        df = load_animal(animal)
        df[!, :sub] .= animal
        push!(dfs, df)
    end
    vcat(dfs...)
end

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

function hmm_lik_stem_inner!(Qstem, Q, prevs, prevl, βgo, βstay, stay_bias, turn_bias, spatial_bias, γ2)
    Qstem .= βgo .* mean.(Q)

    # spatial bias
    # If-else was providing a speedup, may be fixed with views?
    
    if (prevs == 1)
        Qstem[2] += turn_bias
        Qstem[2] += spatial_bias[1]
    elseif (prevs == 2)
        Qstem[3] += turn_bias
        Qstem[3] += spatial_bias[2]
    elseif (prevs == 3)
        Qstem[1] += turn_bias
        Qstem[1] += spatial_bias[3]
    end

    # stay bias
    if (prevs == 1)
        # Qstem[prevs] = βstay * Q[prevs][3-prevl] + stay_bias
        if (prevl == 1)
            Qstem[1] = βstay * (Q[1][2] + γ2 * Q[1][1]) + stay_bias
        else
            # Qstem[prevs] = βstay * Q[prevs][1] + stay_bias
            Qstem[1] = βstay * (Q[1][1] + γ2 * Q[1][2]) + stay_bias
        end
        # Qeff[prevs] = βstay .* Qeff[prevs] .+ stay_bias
    elseif (prevs == 2)
        if (prevl == 1)
            Qstem[2] = βstay * (Q[2][2] + γ2 * Q[2][1]) + stay_bias
        else
            Qstem[2] = βstay * (Q[2][1] + γ2 * Q[2][2]) + stay_bias
        end
    elseif (prevs == 3)
        if (prevl == 1)
            Qstem[3] = βstay * (Q[3][2] + γ2 * Q[3][1]) + stay_bias
        else
            Qstem[3] = βstay * (Q[3][1] + γ2 * Q[3][2]) + stay_bias
        end
    end
end

function hmm_lik_leaf_inner!(Qleaf, Q, stemchoice, βleaf, leaf_turn_bias, leaf_spatial_bias)
    # Leaf choice
    Qleaf .= Q[stemchoice]
    # Add turn bias
    Qleaf[1] += leaf_turn_bias
    # Add per-stem turn biases
    if (stemchoice == 1)
        Qleaf[1] += leaf_spatial_bias[1]
    elseif (stemchoice == 2)
        Qleaf[1] += leaf_spatial_bias[2]
    else
        Qleaf[1] += leaf_spatial_bias[3]
    end
    Qleaf .*= βleaf
end

"""
hmm_lik

HMM Likelihood function

βgo<float>: Scaling for alternate stems
βstay<float>: Scaling for current stem
βleaf<float>: Beta weight for leaf choice softmax
turn_bias<float>: Offset added to (leftward?) choice
spatial_bias<[float, float, float]>: Per-stem offset added to (leftward?) choice
leaf_turn_bias<float>: Offset added to 'first' leaf on entering a new stem
leaf_spatial_bias<[float, float, float]>: Per-stem leaf turn bias
volatility<float>: Assumed chance of a switch
γ2<float>: Discount on timestep 2 for other leaf
depletion_factor<float>: Fraction of value retained when remaining at the same leaf for multiple trials
ϕ: nstates x 6 emission probabilities
add_leaf<bool>: Whether to include likelihood for the leaf choice on a stem switch
record<bool>: Whether to return a record of estimated Q-values and entropy measures

Returns:
    negative likelihood
If 'return':
    Q: Inferred leaf Q-values at the start of each trial, ignoring biases
    Qstem: Stem Q-values at each trial, including biases
    state_entropy
    reward_entropy
"""
function hmm_lik(df, βgo::U, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, volatility, γ2, depletion_factor, retain_belief, ϕ::Array{Float64, 2}, add_leaf, record::Bool) where U
    nstates = size(ϕ, 1)
    ntrials = length(df)

    statePrior = ones(U,nstates) * 1/nstates

    # volatility and state-state transition matrix T
    T = ones(U,nstates,nstates) * volatility / (nstates - 1)
    T[diagind(T)] .= 1 - volatility

    lik::U = 0.

    α = zeros(U, nstates)
    αtemp = zeros(U, nstates)
    # Really a 3x2 matrix, split up here for speed
    Q = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Qstem = zeros(U, 3)
    Qleaf = zeros(U, 2)
    # Qeff = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Qstem = zeros(U, 3)
    Qtemp = zeros(U, 6)
    ϕT = ϕ'
    if record
        Q_record = zeros(ntrials, 6)
        Qstem_record = zeros(ntrials, 3)
        state_entropy = zeros(ntrials)
        reward_entropy = zeros(ntrials)
        stem_1_var = zeros(ntrials)
        stem_2_var = zeros(ntrials)
        stem_3_var = zeros(ntrials)
        leaf_1_var = zeros(ntrials)
        leaf_2_var = zeros(ntrials)
    end
    i = 1

    depletion = ones(U, 6)

    # (dates, sessions, leaf, leafchoice, stemchoice, reward) = hmm_lik_extract_df(df)
    dates = df.date
    sessions = df.session
    leaf = df.leaf
    leafchoice = df.leafchoice
    stemchoice = df.stemchoice
    reward = df.reward

    for date in unique(dates)
        d_sessions = sessions[dates .== date]
        α .= statePrior
        for session in unique(d_sessions)
            # Subset of trials for the session/date
            t_ind = findall((dates .== date) .& (sessions .== session))
            # α is the joint probability of all observations, and the state at time t
            # For predictive purposes, α at time t is prior to the outcome information for trial t            

            # Initial α state
            # We're resetting this to the prior for each session
            # α(z_1k) = π_k p(x_1 | ϕ_k)
            α .= (retain_belief .* α) .+ ((1 - retain_belief) .* statePrior)
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
                # Q .= reshape(ϕT * α, (2, 3))'
                Qtemp .= (ϕT * α) .* depletion
                Q[1][1] = Qtemp[1]
                Q[1][2] = Qtemp[2]
                Q[2][1] = Qtemp[3]
                Q[2][2] = Qtemp[4]
                Q[3][1] = Qtemp[5]
                Q[3][2] = Qtemp[6]
                hmm_lik_stem_inner!(Qstem, Q, prevs, prevl, βgo, βstay, stay_bias, turn_bias, spatial_bias, γ2)
                lik += Qstem[stemchoice[t]] - logsumexp(Qstem)
                if (add_leaf && (prevs != stemchoice[t]))
                    hmm_lik_leaf_inner!(Qleaf, Q, stemchoice[t], βleaf, leaf_turn_bias, leaf_spatial_bias)
                    lik += Qleaf[leafchoice[t]] - logsumexp(Qleaf)
                end

                if record
                    Q_record[i, :] .= Qtemp
                    state_entropy[i] = -sum(α .* log.(α))

                    # First, find the probability of each leaf being .2/.5/.8
                    # Next find entropy of that dist, and mean across all 6 leaves
                    # Large extra allocations happening here
                    reward_probs = vcat([
                        sum((ϕ .== .2) .* α; dims=1)
                        sum((ϕ .== .5) .* α; dims=1)
                        sum((ϕ .== .8) .* α; dims=1)
                        ])
                    reward_entropy[i] = mean([-sum(reward_probs[:,j] .* log.(reward_probs[:,j])) for j in 1:6])

                    state = 1
                    state_probs = zeros(729)
                    stem_1_choice_probs = zeros(729)
                    stem_2_choice_probs = zeros(729)
                    stem_3_choice_probs = zeros(729)
                    leaf_1_choice_probs = zeros(729)
                    leaf_2_choice_probs = zeros(729)
                    for i1 in 1:3
                        p11 = reward_probs[i1, 1]
                        Q[1][1] = [0.2, 0.5, 0.8][i1] * depletion[1]
                        for i2 in 1:3
                            p12 = reward_probs[i2, 2]
                            Q[1][2] = [0.2, 0.5, 0.8][i2] * depletion[2]
                            for j1 in 1:3
                                p21 = reward_probs[j1, 3]
                                Q[2][1] = [0.2, 0.5, 0.8][j1] * depletion[3]
                                for j2 in 1:3
                                    p22 = reward_probs[j2, 4]
                                    Q[2][2] = [0.2, 0.5, 0.8][j2] * depletion[4]
                                    for k1 in 1:3
                                        p31 = reward_probs[k1, 5]
                                        Q[3][1] = [0.2, 0.5, 0.8][k1] * depletion[5]
                                        for k2 in 1:3
                                            p32 = reward_probs[k2, 6]
                                            Q[3][2] = [0.2, 0.5, 0.8][k2] * depletion[6]

                                            state_probs[state] = p11 * p12 * p21 * p22 * p31 * p32
                        
                                            hmm_lik_stem_inner!(Qstem, Q, prevs, prevl, βgo, βstay, stay_bias, turn_bias, spatial_bias, γ2)
                                            stem_1_choice_probs[state] = Qstem[1] - logsumexp(Qstem)
                                            stem_2_choice_probs[state] = Qstem[2] - logsumexp(Qstem)
                                            stem_3_choice_probs[state] = Qstem[3] - logsumexp(Qstem)
                                            if (add_leaf && (prevs != stemchoice[t]))
                                                hmm_lik_leaf_inner!(Qleaf, Q, stemchoice[t], βleaf, leaf_turn_bias, leaf_spatial_bias)
                                                leaf_1_choice_probs[state] = Qleaf[1] - logsumexp(Qleaf)
                                                leaf_2_choice_probs[state] = Qleaf[2] - logsumexp(Qleaf)
                                            end
                                            state += 1
                                        end
                                    end
                                end
                            end
                        end
                    end
                    exp_stem_1_choice_probs = exp.(stem_1_choice_probs)
                    exp_stem_2_choice_probs = exp.(stem_2_choice_probs)
                    exp_stem_3_choice_probs = exp.(stem_3_choice_probs)
                    exp_leaf_1_choice_probs = exp.(leaf_1_choice_probs)
                    exp_leaf_2_choice_probs = exp.(leaf_2_choice_probs)
                    stem_1_mu = sum(state_probs .* exp_stem_1_choice_probs)
                    stem_2_mu = sum(state_probs .* exp_stem_2_choice_probs)
                    stem_3_mu = sum(state_probs .* exp_stem_3_choice_probs)
                    leaf_1_mu = sum(state_probs .* exp_leaf_1_choice_probs)
                    leaf_2_mu = sum(state_probs .* exp_leaf_2_choice_probs)
                    for j in 1:729
                        stem_1_var[i] += state_probs[j] * (exp_stem_1_choice_probs[j] - stem_1_mu)^2
                        stem_2_var[i] += state_probs[j] * (exp_stem_2_choice_probs[j] - stem_2_mu)^2
                        stem_3_var[i] += state_probs[j] * (exp_stem_3_choice_probs[j] - stem_3_mu)^2
                        leaf_1_var[i] += state_probs[j] * (exp_leaf_1_choice_probs[j] - leaf_1_mu)^2
                        leaf_2_var[i] += state_probs[j] * (exp_leaf_2_choice_probs[j] - leaf_2_mu)^2
                    end
                end

                # Update state prediction
                α .= T * α

                # Does the depletion matter at all? Or does it get normalized out?
                # ϕ_depleted = ϕ .* repeat(depletion, 1, 72)'
                if (reward[t] == 1)
                    α .*= view(ϕ, :, leaf[t]) .* depletion[leaf[t]]
                else
                    αtemp .= view(ϕ, :, leaf[t]) .* depletion[leaf[t]]
                    α .*= 1 .- αtemp
                end
                normalize!(α, 1)

                if (depletion_factor < 1.0)
                    if (prevs == stemchoice[t])
                        if (leaf[t] == 1)
                            depletion[1] *= depletion_factor
                        elseif (leaf[t] == 2)
                            depletion[2] *= depletion_factor
                        elseif (leaf[t] == 3)
                            depletion[3] *= depletion_factor
                        elseif (leaf[t] == 4)
                            depletion[4] *= depletion_factor
                        elseif (leaf[t] == 5)
                            depletion[5] *= depletion_factor
                        else
                            depletion[6] *= depletion_factor
                        end
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
        return -lik, Q_record, Qstem_record, state_entropy, reward_entropy, stem_1_var, stem_2_var, stem_3_var, leaf_1_var, leaf_2_var
    else
        return -lik
    end
end

"""
run_hmm

βleaf: Beta weight for leaf choice softmax
turn_bias: Offset added to (leftward?) choice
spatial_bias: Per-stem offset added to (leftward?) choice
leaf_turn_bias: Offset added to 'first' leaf on entering a new stem
leaf_spatial_bias: Per-stem leaf turn bias
γ2: Discount on timestep 2 for other leaf
depletion_factor: Fraction of value retained when remaining at the same leaf for multiple trials
add_leaf: Whether to include likelihood for the leaf choice on a stem switch
"""
function run_hmm(df; maxiter=100, emtol=1e-3, full=true, extended=false,
    add_βleaf=false,
    add_stay_bias=false,
    add_turn_bias=false,
    add_spatial_bias=false,
    add_leaf_turn_bias=false,
    add_leaf_spatial_bias=false,
    add_γ2=false,
    add_depletion_factor=false,
    add_retain_belief=false,
    add_leaf=true)

    data = copy(df)
    data[:, :sub] = data[:, :daynum]
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...

    initbetas = [0. 0]
    initsigma = [5., 5]
    varnames = ["βgo", "βstay"]

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

    initbetas = hcat(initbetas, 0)
    push!(initsigma, 1)
    push!(varnames, "volatility")

    function fn(params, data)
        βgo = params[1]
        βstay = params[2]
        i = 3

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
            γ2 = 0.5 + 0.5 * erf(params[i] / sqrt(2))
            i += 1
        else
            γ2 = 0.0
        end

        if add_depletion_factor
            depletion_factor = 0.5 + 0.5 * erf(params[i] / sqrt(2))
            i += 1
        else
            depletion_factor = 1.0
        end

        if add_retain_belief
            retain_belief = 0.5 + 0.5 * erf(params[i] / sqrt(2))
            i += 1
        else
            retain_belief = 0.0
        end

        volatility = 0.5 + 0.5 * erf(params[i] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
        ϕ = get_contingencies()
        return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, volatility, γ2, depletion_factor, retain_belief, ϕ, add_leaf, false)
    end

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=emtol, full=full, maxiter=maxiter);
    if extended
        (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
        return EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
    else
        return EMResults(varnames,betas,sigma,x,l,h,opt_rec)
    end
end

run_hmm_leaf(data; kwargs...) = run_hmm(data; add_βleaf=true, kwargs...)
run_hmm_leaf_stay(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, kwargs...)
run_hmm_leaf_stay_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, kwargs...)
run_hmm_leaf_stay_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, kwargs...)
run_hmm_leaf_stay_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)

run_hmm_leaf_stay_turn(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, kwargs...)
run_hmm_leaf_stay_spatial(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, kwargs...)
run_hmm_leaf_stay_turn_leafspatial(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_hmm_leaf_stay_turn_leafturn(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, kwargs...)
run_hmm_leaf_stay_spatial_leafspatial(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_hmm_leaf_stay_spatial_leafturn(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, kwargs...)

run_hmm_leaf_stay_turn_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_spatial_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_turn_leafspatial_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_turn_leafturn_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_spatial_leafspatial_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_spatial_leafturn_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)

run_hmm_leaf_stay_turn_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)
run_hmm_leaf_stay_spatial_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)
run_hmm_leaf_stay_turn_leafspatial_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)
run_hmm_leaf_stay_turn_leafturn_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)
run_hmm_leaf_stay_spatial_leafspatial_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)
run_hmm_leaf_stay_spatial_leafturn_depletion_retainbelief(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, add_retain_belief=true, kwargs...)

run_hmm_leaf_stay_turn_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, kwargs...)
run_hmm_leaf_stay_spatial_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, kwargs...)
run_hmm_leaf_stay_turn_leafspatial_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_hmm_leaf_stay_turn_leafturn_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
run_hmm_leaf_stay_spatial_leafspatial_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_hmm_leaf_stay_spatial_leafturn_γ2(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)

run_hmm_leaf_stay_turn_γ2_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_spatial_γ2_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_turn_leafspatial_γ2_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_turn_leafturn_γ2_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_spatial_leafspatial_γ2_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_hmm_leaf_stay_spatial_leafturn_γ2_depletion(data; kwargs...) = run_hmm(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)