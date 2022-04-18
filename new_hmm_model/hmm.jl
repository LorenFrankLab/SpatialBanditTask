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

function load_animal(animal)
    filepath = "/Users/ari/Dropbox/Princeton/SpatialBanditTask"
    
    # csv_file to estimate value for
    # if depletion_flag
        # csv_file = animal * "_clean_contingencies_only_parsed_depletion_data.csv"
        # mkdir(fullfile(filepath,["hmm_decay_",animal]))
    # else
    csv_file = animal * "_clean_contingencies_only_parsed_data.csv"
        # mkdir(fullfile(filepath,['hmm_',animal]))
    # end
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

function hmm_lik(df, βgo::U, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ::Array{Float64, 2}, add_leaf) where U
    nstates = size(ϕ, 1)

    statePrior = ones(U,nstates) * 1/nstates

    # volatility and state-state transition matrix T
    T = ones(U,nstates,nstates) * volatility / (nstates - 1)
    T[diagind(T)] .= 1 - volatility

    lik::U = 0.

    α = zeros(U, nstates)
    # Really a 3x2 matrix, split up here for speed
    Q = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    # Qeff = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Qstem = zeros(U, 3)
    Qtemp = zeros(U, 6)
    ϕT = ϕ'

    # (dates, sessions, leaf, leafchoice, stemchoice, reward) = hmm_lik_extract_df(df)
    dates = df.date
    sessions = df.session
    leaf = df.leaf
    leafchoice = df.leafchoice
    stemchoice = df.stemchoice
    reward = df.reward

    for date in unique(dates)
        d_sessions = sessions[dates .== date]
        for session in unique(d_sessions)
            # Subset of trials for the session/date
            t_ind = findall((dates .== date) .& (sessions .== session))
            # α is the joint probability of all observations, and the state at time t
            # For predictive purposes, α at time t is prior to the outcome information for trial t            

            # Initial α state
            # We're resetting this to the prior for each session
            # α(z_1k) = π_k p(x_1 | ϕ_k)
            α .= statePrior
            prevs::Int = 0
            prevl::Int = 0
            # If rewarded, then emission = reward probs, otherwise 1 - reward probs
            # Note that we're normalizing α with a scaling factor (Bishop, p.627)
            # Since we only care about marginal belief at the current timestep,
            # no need to compute c
            @inbounds for t in t_ind
                # Q = expectation over states
                # Q .= reshape(ϕT * α, (2, 3))'
                Qtemp = ϕT * α
                Q[1][1] = Qtemp[1]
                Q[1][2] = Qtemp[2]
                Q[2][1] = Qtemp[3]
                Q[2][2] = Qtemp[4]
                Q[3][1] = Qtemp[5]
                Q[3][2] = Qtemp[6]
                # Qeff[1] .= Q[1]
                # Qeff[2] .= Q[2]
                # Qeff[3] .= Q[3]
                Qstem .= βgo .* mean.(Q)

                # spatial bias
                # If-else was providing a speedup, may be fixed with views?
                
                if (prevs == 1)
                    Qstem[2] += turn_bias
                    # Qeff[2] .+= turn_bias
                    Qstem[2] += spatial_bias[1]
                    # Qeff[2] .+= spatial_bias[1]
                elseif (prevs == 2)
                    Qstem[3] += turn_bias
                    # Qeff[3] .+= turn_bias
                    Qstem[3] += spatial_bias[2]
                    # Qeff[3] .+= spatial_bias[2]
                elseif (prevs == 3)
                    Qstem[1] += turn_bias
                    # Qeff[1] .+= turn_bias
                    Qstem[1] += spatial_bias[3]
                    # Qeff[1] .+= spatial_bias[3]
                end

                # stay bias
                if (prevs == 1)
                    # Qstem[prevs] = βstay * Q[prevs][3-prevl] + stay_bias
                    if (prevl == 1)
                        Qstem[1] = βstay * Q[1][2] + stay_bias
                    else
                        # Qstem[prevs] = βstay * Q[prevs][1] + stay_bias
                        Qstem[1] = βstay * Q[1][1] + stay_bias
                    end
                    # Qeff[prevs] = βstay .* Qeff[prevs] .+ stay_bias
                elseif (prevs == 2)
                    if (prevl == 1)
                        Qstem[2] = βstay * Q[2][2] + stay_bias
                    else
                        Qstem[2] = βstay * Q[2][1] + stay_bias
                    end
                elseif (prevs == 3)
                    if (prevl == 1)
                        Qstem[3] = βstay * Q[3][2] + stay_bias
                    else
                        Qstem[3] = βstay * Q[3][1] + stay_bias
                    end
                end
                # Stem choice: Current stem is modeled as the βstay * opposite leaf + stay_bias
                # Other stems are modeled as mean
                if (stemchoice[t] == 1)
                    lik += Qstem[1] - logsumexp(Qstem)
                elseif (stemchoice[t] == 2)
                    lik += Qstem[2] - logsumexp(Qstem)
                else
                    lik += Qstem[3] - logsumexp(Qstem)
                end
                # Leaf choice
                if (add_leaf && (prevs != stemchoice[t]))
                    # log likelihood of leaf choice on switches only (meaningless for stay)
                    lik += βleaf * Q[stemchoice[t]][leafchoice[t]]
                    lik -= logsumexp(βleaf .* Q[stemchoice[t]]);
                    # actual leaf choice minus logsumexp of all leaves
                end

                prevl = leafchoice[t]
                prevs = stemchoice[t]

                # Update state prediction
                α .= T * α

                if (reward[t] == 1)
                    α .*= view(ϕ, :, leaf[t])
                else
                    α .*= 1 .- view(ϕ, :, leaf[t])
                end
                normalize!(α, 1)
            end
        end
    end

    return -lik
end

"""
Returns:
    negative likelihood
    Q: Inferred leaf Q-values at the start of each trial, ignoring biases
    Qstem: Stem Q-values at each trial, including biases
"""
function hmm_Q(df, βgo::U, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ::Array{Float64, 2}, add_leaf) where U
    nstates = size(ϕ, 1)

    statePrior = ones(U,nstates) * 1/nstates

    # volatility and state-state transition matrix T
    T = ones(U,nstates,nstates) * volatility / (nstates - 1)
    T[diagind(T)] .= 1 - volatility

    lik::U = 0.

    α = zeros(U, nstates)
    # Really a 3x2 matrix, split up here for speed
    ntrials = size(df, 1)
    Q = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Q_record = zeros(ntrials, 6)
    # Qeff = [zeros(U, 2), zeros(U, 2), zeros(U, 2)]
    Qstem = zeros(U, 3)
    Qstem_record = zeros(ntrials, 3)
    Qtemp = zeros(U, 6)
    ϕT = ϕ'

    # (dates, sessions, leaf, leafchoice, stemchoice, reward) = hmm_lik_extract_df(df)
    dates = df.date
    sessions = df.session
    leaf = df.leaf
    leafchoice = df.leafchoice
    stemchoice = df.stemchoice
    reward = df.reward

    i = 1
    for date in unique(dates)
        d_sessions = sessions[dates .== date]
        for session in unique(d_sessions)
            # Subset of trials for the session/date
            t_ind = findall((dates .== date) .& (sessions .== session))
            ntrials::Int = length(t_ind)   
            # α is the joint probability of all observations, and the state at time t
            # For predictive purposes, α at time t is prior to the outcome information for trial t            

            # Initial α state
            # We're resetting this to the prior for each session
            # α(z_1k) = π_k p(x_1 | ϕ_k)
            α .= statePrior
            prevs::Int = 0
            prevl::Int = 0
            # If rewarded, then emission = reward probs, otherwise 1 - reward probs
            # Note that we're normalizing α with a scaling factor (Bishop, p.627)
            # Since we only care about marginal belief at the current timestep,
            # no need to compute c
            @inbounds for t in t_ind
                # Q = expectation over states
                # Q .= reshape(ϕT * α, (2, 3))'
                Qtemp = ϕT * α
                Q_record[i, :] .= Qtemp
                i += 1
                Q[1][1] = Qtemp[1]
                Q[1][2] = Qtemp[2]
                Q[2][1] = Qtemp[3]
                Q[2][2] = Qtemp[4]
                Q[3][1] = Qtemp[5]
                Q[3][2] = Qtemp[6]
                # Qeff[1] .= Q[1]
                # Qeff[2] .= Q[2]
                # Qeff[3] .= Q[3]
                Qstem .= βgo .* mean.(Q)

                # spatial bias
                # If-else was providing a speedup, may be fixed with views?
                
                if (prevs == 1)
                    Qstem[2] += turn_bias
                    # Qeff[2] .+= turn_bias
                    Qstem[2] += spatial_bias[1]
                    # Qeff[2] .+= spatial_bias[1]
                elseif (prevs == 2)
                    Qstem[3] += turn_bias
                    # Qeff[3] .+= turn_bias
                    Qstem[3] += spatial_bias[2]
                    # Qeff[3] .+= spatial_bias[2]
                elseif (prevs == 3)
                    Qstem[1] += turn_bias
                    # Qeff[1] .+= turn_bias
                    Qstem[1] += spatial_bias[3]
                    # Qeff[1] .+= spatial_bias[3]
                end

                # stay bias
                if (prevs == 1)
                    # Qstem[prevs] = βstay * Q[prevs][3-prevl] + stay_bias
                    if (prevl == 1)
                        Qstem[1] = βstay * Q[1][2] + stay_bias
                    else
                        # Qstem[prevs] = βstay * Q[prevs][1] + stay_bias
                        Qstem[1] = βstay * Q[1][1] + stay_bias
                    end
                    # Qeff[prevs] = βstay .* Qeff[prevs] .+ stay_bias
                elseif (prevs == 2)
                    if (prevl == 1)
                        Qstem[2] = βstay * Q[2][2] + stay_bias
                    else
                        Qstem[2] = βstay * Q[2][1] + stay_bias
                    end
                elseif (prevs == 3)
                    if (prevl == 1)
                        Qstem[3] = βstay * Q[3][2] + stay_bias
                    else
                        Qstem[3] = βstay * Q[3][1] + stay_bias
                    end
                end

                Qstem_record[i, :] .= Qstem
                # Stem choice: Current stem is modeled as the βstay * opposite leaf + stay_bias
                # Other stems are modeled as mean
                if (stemchoice[t] == 1)
                    lik += Qstem[1] - logsumexp(Qstem)
                elseif (stemchoice[t] == 2)
                    lik += Qstem[2] - logsumexp(Qstem)
                else
                    lik += Qstem[3] - logsumexp(Qstem)
                end
                # Leaf choice
                if (prevs != stemchoice[t])
                    # log likelihood of leaf choice on switches only (meaningless for stay)
                    lik += βleaf * Q[stemchoice[t]][leafchoice[t]]
                    lik -= logsumexp(βleaf .* Q[stemchoice[t]]);
                    # actual leaf choice minus logsumexp of all leaves
                end

                prevl = leafchoice[t]
                prevs = stemchoice[t]

                # Update state prediction
                α .= T * α

                if (reward[t] == 1)
                    α .*= view(ϕ, :, leaf[t])
                else
                    α .*= 1 .- view(ϕ, :, leaf[t])
                end
                normalize!(α, 1)
            end
        end
    end

    return -lik, Q_record, Qstem_record
end


"""
hmm_lik
    params: βgo, βstay, βleaf, ...biases, volatility 
    data
"""
function hmm_lik_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = 0.0 # beta for leaf choice on switch
    stay_bias = 0.0
    turn_bias = 0.0
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[3] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, false)
end

function hmm_lik_stay_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = 0.0 # beta for leaf choice on switch
    stay_bias = params[3]
    turn_bias = 0.0
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[4] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, false)
end

function hmm_lik_stay_turn_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = 0.0 # beta for leaf choice on switch
    stay_bias = params[3]
    turn_bias = params[4]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[5] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, false)
end

function hmm_lik_stay_turn_observed_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = 0.0 # beta for leaf choice on switch
    stay_bias = params[3]
    turn_bias = params[4]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[5] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies_observed()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, false)
end

function hmm_lik_stay_spatial_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = 0.0 # beta for leaf choice on switch
    stay_bias = params[3]
    turn_bias = 0.0
    spatial_bias = params[4:6]
    volatility = 0.1 + 0.1 * erf(params[7] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, false)
end

function hmm_lik_leaf_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = 0.0
    turn_bias = 0.0
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[4] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, true)
end

function hmm_lik_leaf_stay_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = 0.0
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[5] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, true)
end

function hmm_lik_leaf_stay_turn_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, true)
end

function hmm_lik_leaf_stay_turn_observed_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies_observed()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, true)
end

function hmm_lik_leaf_stay_spatial_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = 0.0
    spatial_bias = params[5:7]
    volatility = 0.1 + 0.1 * erf(params[8] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies()
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ, true)
end

function run_hmm_em(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0] 
    initsigma = [5., 5, 5, 1]
    varnames = ["βgo", "βstay", "βleaf", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end

function run_hmm_em_stay(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end

function run_hmm_em_stay_turn(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_fn
    # data[!, :sub] = string.(data.date)
    data[:, :sub] = data[:, :daynum]
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end

function run_hmm_em_stay_spatial(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_spatial_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "spatial_bias_1", "spatial_bias_2", "spatial_bias_3", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end

# estimate parameters
# original task
# function hmm_test()
#     data = load_animal("peanut")
#     full = true
#     fn = hmm_lik_stay_spatial_fn
#     data[:, :sub] = data[:, :daynum]
#     subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects, poor variable naming
#     NS = length(subs) #number of subjects/days
#     X = ones(NS) # (group level design matrix); #x group level design matrix... matrix across all days of this task)

#     # fit the full model
#     initbetas = [0. 0 0 0 0 0 0 0] 
#     initsigma = [5., 5, 5, 5, 1, 1, 1, 1]

#     @time em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter = 20)
#     @time em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter = 20);
# end
