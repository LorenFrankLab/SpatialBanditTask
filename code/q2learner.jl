using Statistics
using StatsFuns
using SpecialFunctions
using EM
# this is the likelihood function for the actual model

"""
qlik

Q-Learner Likelihood function

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
initial_Q<float>: Value between 0 and 1 (will be rescaled) to initialize Q-values with at start of each session
α<float>: Learning rate
decay<float>: Decay rate for Q-values (to initial_Q), 1 for full decay, 0 for no decay
decay_Q<float>: Decay target for Q-values
rewscaled<bool>: If true, reward is +1/-1 instead of 0/1
add_leaf<bool>: Whether to include likelihood for the leaf choice on a stem switch
record<bool>: Whether to return a record of estimated Q-values and entropy measures

Returns:
    negative likelihood
If 'record':
    Q: Inferred leaf Q-values at the start of each trial, ignoring biases
"""
function q2lik(data, βgo, βstay::V, βleaf, stay_bias::W, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, depletion_factor, depletion_tuning, retain_belief, initial_Q, decay, decay_Q, α, delay_turn_bias::Bool, rewscaled::Bool, add_leaf::Bool, record::Bool) where {V, W}
    U = promote_type(eltype(βstay), eltype(stay_bias))  # this is a bit of a hack so that we can optionally have either of these fixed at 0
    # mode = "likelihood"

    # rename the variables for easy acccess
    if rewscaled
        r = data.rewscaled
    else
        r = data.reward
    end
    c1 = data.stemchoice
    c2 = data.leafchoice

    # initialize 
    lik = 0.
    lik_go = 0.
    lik_stay = 0.
    lp_stay = 0.
    lp_turn = 0.
    lp_go = 0.
    prevs::Int = 0 #stem
    prevl::Int = 0 #leaf
    stem_stay_p = zeros(U, length(c1))
    stem_turn_alone_p = zeros(U, length(c1))
    stem_go_turn_p = zeros(U, length(c1))
    stem_1_p = zeros(U, length(c1))
    stem_2_p = zeros(U, length(c1))
    stem_3_p = zeros(U, length(c1))
    leaf_1_p = zeros(U, length(c1))
    leaf_2_p = zeros(U, length(c1))

    Q = zeros(U,3,2,length(c1)+1)
    Qstem = zeros(U,3,length(c1))
    Qleaf = zeros(U,2,length(c1))
    depletion = ones(U,3,2)
    Q_depletion = zeros(U,3,2)
    if rewscaled
        Q .= (2.0 .* initial_Q) .- 1.0
        Qstem .= (2.0 .* initial_Q) .- 1.0
        Qleaf .= (2.0 .* initial_Q) .- 1.0
    else
        Q .= initial_Q
        Qstem .= initial_Q
        Qleaf .= initial_Q
    end

    for i = eachindex(c1)
        if ((i>1) && (data.daysessionnum[i] != data.daysessionnum[i-1]))
            if rewscaled
                Q[:,:,i] .*= retain_belief
                Q[:,:,i] .+= (1 .- retain_belief) .* (2.0 * initial_Q - 1.0)
            else
                Q[:,:,i] .*= retain_belief
                Q[:,:,i] .+= (1 .- retain_belief) .* initial_Q
            end
            prevs = 0
            prevl = 0
            depletion .= 1.0
        end

        # Adjust for depletion
        @views Q_depletion .= Q[:, :, i] .* depletion

        # this is the (scaled) value of switching to each alternative stem
        # mean averages over both leaves
        # 
        # Seems to be faster breaking it up into the three calculations
        @views Qstem[1, i] = mean(Q_depletion[1, :]) * βgo
        @views Qstem[2, i] = mean(Q_depletion[2, :]) * βgo
        @views Qstem[3, i] = mean(Q_depletion[3, :]) * βgo

        if prevs > 0
            # this is the (scaled) value of staying with the current stem
            # uses only the value of the next leaf
            # plus the bias toward staying
            Qstem[prevs, i] = βstay * ((1.0 - γ2) * Q_depletion[prevs,3-prevl] + γ2 * Q_depletion[prevs,prevl]) + stay_bias
            # spatial bias
            # 1 -> 2, 2->3, 3->1
            # Probability of stay/switch
            # mod1(prevs + 1, 3) and mod1(prevs + 2, 3) give us the other two stems
            # log(exp(Q_x) / sum(exp.(Q))) -> Q_x - logsumexp(Q)
            if delay_turn_bias  # Compute stay/go before we add a turn bias
                @views lp_stay = Qstem[prevs, i] - logsumexp(Qstem[:, i])
                @views lp_go = logsumexp(Qstem[[mod1(prevs + 1, 3), mod1(prevs + 2, 3)], i]) - logsumexp(Qstem[:, i])
                Qstem[mod1(prevs + 1, 3), i] += spatial_bias[prevs] + turn_bias
            else # Or after we add the turn bias
                Qstem[mod1(prevs + 1, 3), i] += spatial_bias[prevs] + turn_bias
                @views lp_stay = Qstem[prevs, i] - logsumexp(Qstem[:, i])
                @views lp_go = logsumexp(Qstem[[mod1(prevs + 1, 3), mod1(prevs + 2, 3)], i]) - logsumexp(Qstem[:, i])
            end
            if prevs == c1[i]  # If a stay trial, just the stay/go likelihood
                ll = lp_stay
            else  # If a go trial, p(go) times the left/right turn choice
                @views lp_turn = Qstem[c1[i], i] - logsumexp(Qstem[[mod1(prevs + 1, 3), mod1(prevs + 2, 3)], i])
                ll = lp_go + lp_turn
            end
            stem_stay_p[i] = exp(lp_stay)
            stem_turn_alone_p[i] = exp(lp_turn)
            stem_go_turn_p[i] = exp(lp_go + lp_turn)

            @views stem_1_p[i] = exp(Qstem[1, i] - logsumexp(Qstem[:, i]))
            @views stem_2_p[i] = exp(Qstem[2, i] - logsumexp(Qstem[:, i]))
            @views stem_3_p[i] = exp(Qstem[3, i] - logsumexp(Qstem[:, i]))
        else  # First trial, no biases and just three-way choice
            @views ll = Qstem[c1[i], i] - logsumexp(Qstem[:, i])
        end

        # log likelihood of stem choice (log of logistic)
        lik += ll
        if ((i>1) && c1[i] != prevs)
            lik_go += ll
        else
            lik_stay += ll
        end

        ## Leaf
        if ((prevs != c1[i]) && add_leaf)
            @views Qleaf[:, i] .= Q_depletion[c1[i], :]
            @views Qleaf[:, i] .*= βleaf
            # Add leaf bias
            Qleaf[1, i] += leaf_turn_bias + leaf_spatial_bias[c1[i]]
            # log likelihood of leaf choice on switches only
            @views lik += Qleaf[c2[i], i] - logsumexp(Qleaf[ :, i]);

            @views leaf_1_p[i] = Qleaf[1, i] - logsumexp(Qleaf[:, i]);
            @views leaf_2_p[i] = Qleaf[2, i] - logsumexp(Qleaf[:, i]);
        end

        # learn about the chosen leaf
        # Decay towards initial_Q
        @views Q[:,:,i+1] .= (1 - decay) .* Q[:, :, i] .+ decay .* decay_Q

        if (depletion_factor < 1.0)
            # May be a way to handle this more efficiently
            if rewscaled
                # At 0.0, r1 should be 1.0, r2 should be small
                # At 1.0, r1 should be large. r2 should be 1.0
                d = depletion[c1[i], c2[i]]
                Qi = Q[c1[i],c2[i],i]
                Qi_adj = (Qi + 1) / 2 # rescale to 0.0-1.0
                r2 = 2*Qi_adj*(1-d)/(1-d*Qi_adj) - 1 # back to -1/1
                if r[i] > 0
                    Q[c1[i],c2[i],i+1] = (1 - α) * Qi + α
                else
                    Q[c1[i],c2[i],i+1] = (1 - α) * Qi - α * r2
                end
            else
                # At 0.0, d1 should be 1.0, d2 should be small
                # At 1.0, d1 should be large, d2 should be 1.0
                d = depletion[c1[i], c2[i]]
                Qi = Q[c1[i],c2[i],i]
                if r[i] > 0
                    Q[c1[i],c2[i],i+1] = (1 - α) * Qi + α
                else
                    Q[c1[i],c2[i],i+1] = (1 - α) * Qi + α * Qi*(1-d)/(1-d*Qi)
                end
            end

            if (prevs == c1[i])
                depletion[c2[i]] *= depletion_factor
            else
                depletion .= 1
            end
        else
            # Normal Q-learning update
            Q[c1[i],c2[i],i+1] = (1-α) * Q[c1[i],c2[i],i] + α * r[i]
        end

        prevs = c1[i]
        prevl = c2[i]
    end
    
    if record
        record_df = DataFrame(
            # Base Q values
            Q1 = Q[1, 1, 1:length(c1)],
            Q2 = Q[1, 2, 1:length(c1)],
            Q3 = Q[2, 1, 1:length(c1)],
            Q4 = Q[2, 2, 1:length(c1)],
            Q5 = Q[3, 1, 1:length(c1)],
            Q6 = Q[3, 2, 1:length(c1)],
            Qstem1 = Qstem[1, :],
            Qstem2 = Qstem[2, :],
            Qstem3 = Qstem[3, :],
            Qleaf1 = Qleaf[1, :],
            Qleaf2 = Qleaf[2, :],
            stem_stay_p = stem_stay_p,
            stem_turn_alone_p = stem_turn_alone_p,
            stem_go_turn_p = stem_go_turn_p,
            stem_1_p = stem_1_p,
            stem_2_p = stem_2_p,
            stem_3_p = stem_3_p,
            leaf_1_p = leaf_1_p,
            leaf_2_p = leaf_2_p,
        )
        record_df[!, :stem_choice_p] .= 0.0
        record_df[!, :leaf_choice_p] .= 0.0
        # Label trials with probability of eventual choice
        for i in 1:nrow(record_df)
            s = c1[i]
            record_df[i, :stem_choice_p] = record_df[i, Symbol("stem_$(s)_p")]
            l = c2[i]
            record_df[i, :leaf_choice_p] = record_df[i, Symbol("leaf_$(l)_p")]
        end
        return -lik, record_df
    else
        return -lik
    end
    # elseif mode=="lik_stay_go"
    #     return [-lik_go,-lik_stay] 
    # elseif mode=="sim"
    #     return Q[:,:,1:length(c1)]
    # elseif mode=="sim_effective"
    #     return Qeff[:,:,1:length(c1)]
    # end
end

"""
Q Learning likelihood function for non-optimization usage, e.g. just computing the likelihood for a particular set of parameters
"""
function q2lik(data; βgo=0.0, βstay=0.0, α=0.0, βleaf=0.0, stay_bias=0.0, turn_bias=0.0, spatial_bias=[0.0, 0.0, 0.0], leaf_turn_bias=0.0, leaf_spatial_bias=[0.0, 0.0, 0.0], γ2=0.0, depletion_factor=1.0, depletion_tuning=1.0, retain_belief=0.0, initial_Q=0.75, decay=0.0, decay_Q=0.75, delay_turn_bias=false, rewscaled=false, add_leaf=true, record=false)
    q2lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, depletion_factor, depletion_tuning, retain_belief, initial_Q, decay, decay_Q, α, delay_turn_bias, rewscaled, add_leaf, record)
end

"""
Q-Learning likelihood function using parameters from an existing EM run

Can pass in extra parameters with `params`: these will override the EM results

Note that params should be a dictionary of symbols to values, e.g. (:βgo => 2.0)

If `subject` is provided, use parameters from subject `subject`.
Otherwise use group-level betas
"""
function q2lik(data, results::T; subject=0, params=nothing, delay_turn_bias=false, rewscaled=false, add_leaf=true, record=false) where T <: EMResultsAbstract
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
    if haskey(d, :α)
        d[:α] = unitnorm(d[:α])
    end
    if haskey(d, :γ2)
        d[:γ2] = unitnorm(d[:γ2])
    end
    if haskey(d, :depletion_factor)
        d[:depletion_factor] = unitnorm(d[:depletion_factor])
    end
    if haskey(d, :depletion_tuning)
        d[:depletion_tuning] = unitnorm(d[:depletion_tuning])
    end
    if haskey(d, :retain_belief)
        d[:retain_belief] = unitnorm(d[:retain_belief])
    end
    if haskey(d, :initial_Q)
        d[:initial_Q] = unitnorm(d[:initial_Q])
    end
    if haskey(d, :decay)
        d[:decay] = unitnorm(d[:decay])
    end
    if haskey(d, :decay_Q)
        d[:decay_Q] = unitnorm(d[:decay_Q])
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
    q2lik(data; delay_turn_bias, rewscaled, add_leaf, record, d...)
end

"""
run_q
full: Whether to model full covariance matrix
extended: Try to calculate p-values and group-level covariance
quiet: Silence progress

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
retain_belief: Fraction of Q-value estimates to retain between sessions
initial_Q: Allow Q-values to initialize to a value above minimum (a fixed value can also be provided for this)
decay: Decay rate for Q-values (to initial_Q)
rewscaled: If true, reward is +1/-1 instead of 0/1
add_leaf: Whether to include likelihood for the leaf choice on a stem switch
"""
function run_q2(df; maxiter=100, emtol=1e-3, full=true, extended=false, quiet=false,
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
    add_depletion_tuning=false,
    add_retain_belief=false,
    add_initial_Q=false,
    add_decay=false,
    add_decay_Q=false,
    initial_Q=nothing,
    delay_turn_bias=false,
    depletion_tuning=1.0,
    rewscaled=false,
    add_leaf=true,
    subjlevel=:daynum,
    )

    @show add_βgo
    @show add_βstay
    @show add_βleaf
    @show add_stay_bias
    @show add_turn_bias
    @show add_spatial_bias
    @show add_leaf_turn_bias
    @show add_leaf_spatial_bias
    @show add_γ2
    @show add_depletion_factor
    @show add_retain_belief
    @show add_initial_Q
    @show add_decay
    @show add_decay_Q
    @show initial_Q
    @show delay_turn_bias
    @show depletion_tuning
    @show rewscaled
    @show add_leaf
    @show subjlevel

    data = copy(df)
    data[:, :sub] = data[:, subjlevel]
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...

    initbetas = Matrix{Float64}(undef, 1, 0)
    initsigma = Vector{Float64}(undef, 0)
    varnames = Vector{String}(undef, 0)

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
    if add_depletion_tuning
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "depletion_tuning")
    end
    if add_retain_belief
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "retain_belief")
    end
    if add_initial_Q
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "initial_Q")
    end
    if add_decay
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "decay")
    end
    if add_decay_Q
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "decay_Q")
    end

    initbetas = hcat(initbetas, 0)
    push!(initsigma, 1)
    push!(varnames, "α")

    @show varnames

    function fn(params, data)
        i = 1

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

        if add_depletion_tuning
            f_depletion_tuning = unitnorm(params[i])
            i += 1
        else
            f_depletion_tuning = depletion_tuning
        end

        if add_retain_belief
            retain_belief = unitnorm(params[i])
            i += 1
        else
            retain_belief = 0.0
        end

        if add_initial_Q
            f_initial_Q = unitnorm(params[i])
            i += 1
        elseif !isnothing(initial_Q)
            f_initial_Q = initial_Q
        else
            f_initial_Q = 0.0
        end

        if add_decay
            decay = unitnorm(params[i])
            i += 1
        else
            decay = 0.0
        end

        if add_decay_Q
            decay_Q = unitnorm(params[i])
            i += 1
        else
            decay_Q = f_initial_Q
        end

        α = unitnorm(params[i])
        i += 1

        return q2lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, depletion_factor, f_depletion_tuning, retain_belief, f_initial_Q, decay, decay_Q, α, delay_turn_bias, rewscaled, add_leaf, false)
    end

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=emtol, full=full, maxiter=maxiter, quiet=quiet);
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

function find_Q_vals_q2learner(df, results; add_leaf=true, rewscaled, delay_turn_bias, params=nothing, subjlevel=:daynum)
    data = copy(df)
    data[:, :sub] = data[:, subjlevel]
    nsubjs = maximum(data.sub)
    liks = zeros(nsubjs)
    dfs = []
    for i in 1:nsubjs
        (liks[i], df) = q2lik(view(data, data.sub .== i, :), results;
        subject=i, add_leaf, rewscaled, delay_turn_bias, params, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end

run_q2_leaf(data; kwargs...) = run_q2(data; add_βleaf=true, kwargs...)
run_q2_leaf_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_γ2=true, kwargs...)
run_q2_leaf_retainbelief(data; kwargs...) = run_q2(data; add_βleaf=true, add_retain_belief=true, kwargs...)
run_q2_leaf_stay(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, kwargs...)
run_q2_leaf_stay_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_retainbelief(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, kwargs...)

run_q2_leaf_stay_turn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, kwargs...)
run_q2_leaf_stay_spatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, kwargs...)
run_q2_leaf_stay_leafturn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, kwargs...)
run_q2_leaf_stay_leafspatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q2_leaf_stay_turn_leafturn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, kwargs...)

run_q2_leaf_stay_turn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_spatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_leafturn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_leafspatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)

run_q2_leaf_initialQ(data; kwargs...) = run_q2(data; add_βleaf=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_retainbelief(data; kwargs...) = run_q2(data; add_βleaf=true, add_retain_belief=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_retainbelief(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, kwargs...)

run_q2_leaf_initialQ_stay_turn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_γ2(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)

run_q2_leaf_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, kwargs...)
run_q2_leaf_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_γ2=true, kwargs...)
run_q2_leaf_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_retain_belief=true, kwargs...)
run_q2_leaf_stay_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, kwargs...)
run_q2_leaf_stay_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_retain_belief=true, kwargs...)

run_q2_leaf_stay_turn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, kwargs...)
run_q2_leaf_stay_spatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, kwargs...)
run_q2_leaf_stay_leafturn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, kwargs...)
run_q2_leaf_stay_leafspatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, kwargs...)

run_q2_leaf_stay_turn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_spatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_leafturn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_leafspatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)

run_q2_leaf_initialQ_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_retain_belief=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)

# Depletion
run_q2_leaf_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_retainbelief_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_retain_belief=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_retainbelief_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_stay_turn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafturn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafspatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_stay_turn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafturn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafspatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_initialQ_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_retainbelief_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_retainbelief_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_γ2_depletion(data; kwargs...) = run_q2(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_retain_belief=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_stay_turn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafturn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafspatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_stay_turn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafturn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_leafspatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafspatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_turn_leafturn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafspatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_stay_spatial_leafturn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_initialQ_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_leaf_initialQ_stay_turn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafturn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_leafspatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafspatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_turn_leafturn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafspatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_leaf_initialQ_stay_spatial_leafturn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

# No Leaf
run_q2_base(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, kwargs...)
run_q2_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_γ2=true, kwargs...)
run_q2_retainbelief(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, kwargs...)
run_q2_stay(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, kwargs...)
run_q2_stay_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, kwargs...)
run_q2_stay_retainbelief(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, kwargs...)

run_q2_stay_turn(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, kwargs...)
run_q2_stay_spatial(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, kwargs...)
run_q2_stay_turn_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, kwargs...)
run_q2_stay_spatial_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, kwargs...)

run_q2_initialQ(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_initial_Q=true, kwargs...)
run_q2_initialQ_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_retainbelief(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_retainbelief(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, kwargs...)

run_q2_initialQ_stay_turn(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_spatial(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_turn_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_spatial_γ2(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)

run_q2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, kwargs...)
run_q2_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_γ2=true, kwargs...)
run_q2_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_retain_belief=true, kwargs...)
run_q2_stay_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, kwargs...)
run_q2_stay_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, kwargs...)
run_q2_stay_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, kwargs...)

run_q2_stay_turn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, kwargs...)
run_q2_stay_spatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, kwargs...)
run_q2_stay_turn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, kwargs...)
run_q2_stay_spatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, kwargs...)

run_q2_initialQ_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_initial_Q=true, kwargs...)
run_q2_initialQ_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_retain_belief=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_retainbelief_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, kwargs...)

run_q2_initialQ_stay_turn_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_spatial_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_turn_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)
run_q2_initialQ_stay_spatial_γ2_decay(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, kwargs...)

# No leaf with depletion
run_q2_base_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_depletion_factor=true, kwargs...)
run_q2_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_retainbelief_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, add_depletion_factor=true, kwargs...)
run_q2_stay_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_depletion_factor=true, kwargs...)
run_q2_stay_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_stay_retainbelief_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_depletion_factor=true, kwargs...)

run_q2_stay_turn_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_stay_spatial_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_stay_turn_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_stay_spatial_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)

run_q2_initialQ_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_retainbelief_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_retainbelief_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_initialQ_stay_turn_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_spatial_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_turn_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_spatial_γ2_depletion(data; kwargs...) = run_q2(data; add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_depletion_factor=true, kwargs...)
run_q2_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_retain_belief=true, add_depletion_factor=true, kwargs...)
run_q2_stay_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_depletion_factor=true, kwargs...)
run_q2_stay_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_stay_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_depletion_factor=true, kwargs...)

run_q2_stay_turn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_depletion_factor=true, kwargs...)
run_q2_stay_spatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_depletion_factor=true, kwargs...)
run_q2_stay_turn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_q2_stay_spatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_depletion_factor=true, kwargs...)

run_q2_initialQ_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_retainbelief_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_retain_belief=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)

run_q2_initialQ_stay_turn_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_spatial_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_turn_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)
run_q2_initialQ_stay_spatial_γ2_decay_depletion(data; kwargs...) = run_q2(data; add_decay=true, add_leaf=false, add_βleaf=false, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, add_initial_Q=true, add_depletion_factor=true, kwargs...)