using Statistics
using StatsFuns
using SpecialFunctions
using DataFrames
using LinearAlgebra
using EM

""" Run EM optimization with option for extended results
"""
function em_opt_extended(trials,subs,X,betas,sigma,lik_fn,varnames;
    emtol=1e-3, maxiter=100, full=true, extended=true, quiet=false, startx=nothing)
    _startx = []
    if !isnothing(startx)
        _startx = startx
    end
    (betas,sigma,x,l,h,opt_rec) = em(trials,subs,X,betas,sigma,lik_fn; emtol=emtol, full=full, maxiter=maxiter, quiet=quiet, startx=_startx);
    if extended
        try
            @info "Running emerrors"
            (standarderrors,pvalues,covmtx) = emerrors(trials,subs,x,X,h,betas,sigma,lik_fn)
            return EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
        catch err
            if isa(err, SingularException) || isa(err, DomainError) || isa(err, LAPACKException)
            @warn err
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

function beta_var(α, β)
    return α * β / ((α + β)^2 * (α + β + 1))
end

""" Beta-Bernoulli likelihood function
data: DataFrame with columns:
    :reward - 1.0 for reward, 0.0 for no reward
    :stemchoice - 1, 2, or 3 for stem choice
    :leafchoice - 1 or 2 for leaf choice
    :daysessionnum - integer session number (necessary to reset beliefs on session change)
Parameters:
    βgo - inverse temperature for switching stems
    βstay - inverse temperature for staying on same stem
    βleaf - inverse temperature for leaf choice on switch
    stay_bias - bias toward staying on same stem
    turn_bias - bias toward turning left when switching stems
    spatial_bias - vector of length 3, left bias when at each stem
    leaf_turn_bias - bias toward left leaf choice
    leaf_spatial_bias - vector of length 3, bias for each stem's left leaf
    beta_decay - decay rate for beta distribution
    a_baseline - baseline value for α (for initialization and decay)
    b_baseline - baseline value for β (for initialization and decay)
    γ2 - weighting for leaf value when staying on same stem
    depletion_factor - factor by which reward depletes on repeated choices
    retain_belief - fraction of belief retained across sessions
    delay_turn_bias - if true, apply turn bias after computing stay/go probabilities
    rewscaled - if true, rescale Q values to be centered around 0
    add_leaf - if true, include leaf choice in likelihood
    record - if true, return a DataFrame with trial-by-trial variables
"""
function beta_lik(data, βgo, βstay::V, βleaf, stay_bias::W, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, beta_decay, a_baseline, b_baseline, γ2, depletion_factor, retain_belief, delay_turn_bias::Bool, rewscaled::Bool, add_leaf::Bool, record::Bool) where {V, W}
    U = promote_type(eltype(βstay), eltype(stay_bias))  # this is a bit of a hack so that we can optionally have either of these fixed at 0
    # mode = "likelihood"

    # rename the variables for easy acccess
    r = data.reward
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

    # Beta dist variables
    betadist_α = zeros(U, 3, 2, length(c1)+1)
    betadist_β = zeros(U, 3, 2, length(c1)+1)
    betadist_α[:, :, 1] .= a_baseline
    betadist_β[:, :, 1] .= b_baseline

    # Q-values
    # Choice-level representation
    # Q is the 'true' expected value, not including depletion
    # Qdep (from Q) includes depletion for that trial
    # Qstem (from Qdep) takes a (weighted) average of each stem, and adds stay/turn biases
    # Qstem_nobias (from Qdep) takes a (weighted) average of each stem, without any biases
    # Qstem_pre_update_post_bias is the value with biases from the final choice, but before a value update
    # Qleaf (from Qdep) incorporates a turn bias
    Q = zeros(U,3,2,length(c1)+1)
    Qdep = zeros(U,3,2,length(c1))
    Qstem = zeros(U,3,length(c1))
    Qstem_nobias = zeros(U,3,length(c1))
    Qstem_pre_update_post_bias = zeros(U,3,length(c1))
    Qleaf = zeros(U,2,length(c1))
    Q[:, :, 1] .= betadist_α[:, :, 1] ./ (betadist_α[:, :, 1] + betadist_β[:, :, 1])
    if rewscaled
        @views Q[:, :, 1] .-= 0.5
    end

    depletion = ones(U,3,2,length(c1)+1)

    for i in eachindex(c1)
        if ((i>1) && (data.daysessionnum[i] != data.daysessionnum[i-1]))
            # If retain_belief > 0, carry over some of the previous belief
            betadist_α[:, :, i] .= (1 - retain_belief) * 1.0 .+ retain_belief * betadist_α[:, :, i-1]
            betadist_β[:, :, i] .= (1 - retain_belief) * 1.0 .+ retain_belief * betadist_β[:, :, i-1]
            @views Q[:, :, i] .= betadist_α[:, :, i] ./ (betadist_α[:, :, i] + betadist_β[:, :, i])
            if rewscaled
                @views Q[:, :, i] .-= 0.5
            end
            prevs = 0
            prevl = 0
            @views depletion[:, :, i] .= 1
        end

        # this is the (scaled) value of switching to each alternative stem
        # mean averages over both leaves
        # 
        # Seems to be faster breaking it up into the three calculations
        @views Qdep[:, :, i] .= Q[:, :, i] .* depletion[:, :, i]
        @views Qstem_nobias[1, i] = mean(Qdep[1, :, i]) * βgo
        @views Qstem_nobias[2, i] = mean(Qdep[2, :, i]) * βgo
        @views Qstem_nobias[3, i] = mean(Qdep[3, :, i]) * βgo
        @views Qstem[:, i] = Qstem_nobias[:, i]
        @views Qstem_pre_update_post_bias[:, i] = Qstem_nobias[:, i]

        if prevs > 0
            # this is the (scaled) value of staying with the current stem
            # uses only the value of the next leaf
            # plus the bias toward staying
            Qstem_nobias[prevs, i] = βstay * ((1.0 - γ2) * Qdep[prevs,3-prevl,i] + γ2 * Qdep[prevs,prevl,i])
            Qstem_pre_update_post_bias[prevs, i] = Qstem_nobias[prevs, i]
            Qstem[prevs, i] = Qstem_nobias[prevs, i] + stay_bias
            # Bias relative to current trial instead of last
            Qstem_pre_update_post_bias[c1[i], i] += stay_bias
            # @views Qstem[:, i] .*= βgo
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
            Qstem_pre_update_post_bias[mod1(c1[i] + 1, 3), i] += spatial_bias[prevs] + turn_bias
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
            # @views Qstem[:, i] .*= βgo
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
            @views Qleaf[:, i] .= Qdep[c1[i], :, i]
            @views Qleaf[:, i] .*= βleaf
            # Add leaf bias
            Qleaf[1, i] += leaf_turn_bias + leaf_spatial_bias[c1[i]]
            # log likelihood of leaf choice on switches only
            @views lik += Qleaf[c2[i], i] - logsumexp(Qleaf[ :, i]);

            @views leaf_1_p[i] = Qleaf[1, i] - logsumexp(Qleaf[:, i]);
            @views leaf_2_p[i] = Qleaf[2, i] - logsumexp(Qleaf[:, i]);
        end

        @views betadist_α[:, :, i+1] .= betadist_α[:, :, i]
        @views betadist_β[:, :, i+1] .= betadist_β[:, :, i]

        # Decay existing estimates
        betadist_α[:, :, i+1] .= (betadist_α[:, :, i+1] .- a_baseline) .* (1.0 - beta_decay * 0.5) .+ a_baseline
        betadist_β[:, :, i+1] .= (betadist_β[:, :, i+1] .- b_baseline) .* (1.0 - beta_decay * 0.5) .+ b_baseline

        # learn about the chosen leaf
        if r[i] == 1.0
            betadist_α[c1[i], c2[i], i+1] += 1
        else
            mu = betadist_α[c1[i], c2[i], i] / (betadist_α[c1[i], c2[i], i] + betadist_β[c1[i], c2[i], i])
            d = depletion[c1[i], c2[i], i]
            target = mu*(1-d)/(1-d*mu)
            betadist_β[c1[i], c2[i], i+1] += d*(1-target)
        end

        # Update estimated reward probabilities
        @views Q[:, :, i+1] .= betadist_α[:, :, i+1] ./ (betadist_α[:, :, i+1] .+ betadist_β[:, :, i+1])
        if rewscaled
            @views Q[:, :, i+1] .-= 0.5
        end

        @views depletion[:, :, i+1] .= depletion[:, :, i]
        if (prevs == c1[i])
            depletion[c1[i], c2[i], i+1] *= depletion_factor
        else
            depletion[:, :, i+1] .= 1
        end

        prevs = c1[i]
        prevl = c2[i]
    end
    
    if record
        betadist_var = beta_var.(betadist_α, betadist_β)
        record_df = DataFrame(
            # means
            betadist_α1 = betadist_α[1, 1, 1:length(c1)],
            betadist_α2 = betadist_α[1, 2, 1:length(c1)],
            betadist_α3 = betadist_α[2, 1, 1:length(c1)],
            betadist_α4 = betadist_α[2, 2, 1:length(c1)],
            betadist_α5 = betadist_α[3, 1, 1:length(c1)],
            betadist_α6 = betadist_α[3, 2, 1:length(c1)],
            betadist_β1 = betadist_β[1, 1, 1:length(c1)],
            betadist_β2 = betadist_β[1, 2, 1:length(c1)],
            betadist_β3 = betadist_β[2, 1, 1:length(c1)],
            betadist_β4 = betadist_β[2, 2, 1:length(c1)],
            betadist_β5 = betadist_β[3, 1, 1:length(c1)],
            betadist_β6 = betadist_β[3, 2, 1:length(c1)],
            betadist_var1 = betadist_var[1, 1, 1:length(c1)],
            betadist_var2 = betadist_var[1, 2, 1:length(c1)],
            betadist_var3 = betadist_var[2, 1, 1:length(c1)],
            betadist_var4 = betadist_var[2, 2, 1:length(c1)],
            betadist_var5 = betadist_var[3, 1, 1:length(c1)],
            betadist_var6 = betadist_var[3, 2, 1:length(c1)],
            # variances
            # Base Q values
            Q1 = Q[1, 1, 1:length(c1)],
            Q2 = Q[1, 2, 1:length(c1)],
            Q3 = Q[2, 1, 1:length(c1)],
            Q4 = Q[2, 2, 1:length(c1)],
            Q5 = Q[3, 1, 1:length(c1)],
            Q6 = Q[3, 2, 1:length(c1)],
            Qdep1 = Qdep[1, 1, :],
            Qdep2 = Qdep[1, 2, :],
            Qdep3 = Qdep[2, 1, :],
            Qdep4 = Qdep[2, 2, :],
            Qdep5 = Qdep[3, 1, :],
            Qdep6 = Qdep[3, 2, :],
            Qstem1 = Qstem[1, :],
            Qstem2 = Qstem[2, :],
            Qstem3 = Qstem[3, :],
            Qstem1_nobias = Qstem_nobias[1, :],
            Qstem2_nobias = Qstem_nobias[2, :],
            Qstem3_nobias = Qstem_nobias[3, :],
            Qstem1_pre_update_post_bias = Qstem_pre_update_post_bias[1, :],
            Qstem2_pre_update_post_bias = Qstem_pre_update_post_bias[2, :],
            Qstem3_pre_update_post_bias = Qstem_pre_update_post_bias[3, :],
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
            depletion_leaf_1 = depletion[1, 1, 1:length(c1)],
            depletion_leaf_2 = depletion[1, 2, 1:length(c1)],
            depletion_leaf_3 = depletion[2, 1, 1:length(c1)],
            depletion_leaf_4 = depletion[2, 2, 1:length(c1)],
            depletion_leaf_5 = depletion[3, 1, 1:length(c1)],
            depletion_leaf_6 = depletion[3, 2, 1:length(c1)],
            depletion_mean_stem_1 = mean(depletion[1, :, 1:length(c1)]; dims=1)[1, :],
            depletion_mean_stem_2 = mean(depletion[2, :, 1:length(c1)]; dims=1)[1, :],
            depletion_mean_stem_3 = mean(depletion[3, :, 1:length(c1)]; dims=1)[1, :],
        )
        record_df[!, :stem_choice_p] .= 0.0
        record_df[!, :leaf_choice_p] .= 0.0
        record_df[!, :prev_global_variance] .= (
            record_df.betadist_var1 .+ record_df.betadist_var2 .+ 
            record_df.betadist_var3 .+ record_df.betadist_var4 .+ 
            record_df.betadist_var5 .+ record_df.betadist_var6)
        record_df[!, :prev_leaf_variance] .= 0.0
        record_df[!, :upcoming_leaf_variance] .= 0.0
        record_df[!, :next_choice_variance] .= 0.0
        record_df[!, :prev_local_variance] .= 0.0
        record_df[!, :next_choice_local_variance] .= 0.0
        record_df[!, :next_choice_local_variance_gamma] .= 0.0
        # Label trials with probability of eventual choice
        for i in 1:nrow(record_df)
            s = c1[i]
            record_df[i, :stem_choice_p] = record_df[i, Symbol("stem_$(s)_p")]
            l = c2[i]
            record_df[i, :leaf_choice_p] = record_df[i, Symbol("leaf_$(l)_p")]
            prevleaf = (c1[i]-1)*2 + c2[i]
            upcomingleaf = (c1[i]-1)*2 + 3-c2[i]
            record_df[i, :prev_leaf_variance] = record_df[i, Symbol("betadist_var$(prevleaf)")]
            if i < length(c1)
                nextleaf = (c1[i+1]-1)*2 + c2[i+1]
                record_df[i, :next_choice_variance] = record_df[i+1, Symbol("betadist_var$(nextleaf)")]
                @views record_df[i, :next_choice_local_variance] = sum(betadist_var[c1[i+1], :, i+1])
                @views record_df[i, :next_choice_local_variance_gamma] = (1 - γ2) * betadist_var[c1[i+1], c2[i+1], i+1] + γ2 * betadist_var[c1[i+1], 3-c2[i+1], i+1]
                record_df[i, :upcoming_leaf_variance] = record_df[i+1, Symbol("betadist_var$(upcomingleaf)")]
            end
            @views record_df[i, :prev_local_variance] = sum(betadist_var[c1[i], :, i+1])
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

function beta_lik(data; βgo=0.0, βstay=0.0, βleaf=0.0, stay_bias=0.0, turn_bias=0.0, spatial_bias=[0.0, 0.0, 0.0], leaf_turn_bias=0.0, leaf_spatial_bias=[0.0, 0.0, 0.0], beta_decay=0.0, a_baseline=1.0, b_baseline=1.0, γ2=0.0, depletion_factor=1.0, retain_belief=0.0, delay_turn_bias=false, rewscaled=false, add_leaf=true, record=false)
    beta_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, beta_decay, a_baseline, b_baseline, γ2, depletion_factor, retain_belief, delay_turn_bias, rewscaled, add_leaf, record)
end

"""
Beta bernoulli likelihood function using parameters from an existing EM run

Can pass in extra parameters with `params`: these will override the EM results

Note that params should be a dictionary of symbols to values, e.g. (:βgo => 2.0)

If `subject` is provided, use parameters from subject `subject`.
Otherwise use group-level betas
"""
function beta_lik(data, results::T; subject=0, params=nothing, delay_turn_bias=false, rewscaled=false, add_leaf=true, record=false) where T <: EMResultsAbstract
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
    if haskey(d, :beta_decay)
        d[:beta_decay] = unitnorm(d[:beta_decay])
    end
    if haskey(d, :a_baseline)
        d[:a_baseline] = exp(d[:a_baseline])
    end
    if haskey(d, :b_baseline)
        d[:b_baseline] = exp(d[:b_baseline])
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
    beta_lik(data; delay_turn_bias, rewscaled, add_leaf, record, d...)
end


""" Run EM optimization for beta-bernoulli model
Options to include/exclude various parameters
df: DataFrame containing the data

Optimization parameters:
maxiter: maximum number of EM iterations
emtol: convergence tolerance for EM
full: if true, model full group-level covariance structure
extended: if true, compute standard errors and p-values
quiet: if true, suppress output

Model parameters:
add_βgo - include βgo parameter
add_βstay - include βstay parameter
add_βleaf - include βleaf parameter
add_stay_bias - include stay bias parameter
add_turn_bias - include turn bias parameter
add_spatial_bias - include spatial bias parameters  (3 parameters)
add_leaf_turn_bias - include leaf turn bias parameter
add_leaf_spatial_bias - include leaf spatial bias parameters (3 parameters)
add_beta_decay - include beta decay parameter
add_a_baseline - include a_baseline parameter
add_b_baseline - include b_baseline parameter
add_γ2 - include γ2 parameter
add_depletion_factor - include depletion factor parameter
add_retain_belief - include retain belief parameter
delay_turn_bias - if true, apply turn bias after computing stay/go probabilities
rewscaled - if true, rescale Q values to be centered around 0
add_leaf - if true, include leaf choice in likelihood
subjlevel - symbol for column in df indicating subject/daynum (either :daynum or :daysessionnum)
"""
function run_beta_lik(df; maxiter=100, emtol=1e-3, full=true, extended=false, quiet=false,
    add_βgo=true,
    add_βstay=true,
    add_βleaf=false,
    add_stay_bias=false,
    add_turn_bias=false,
    add_spatial_bias=false,
    add_leaf_turn_bias=false,
    add_leaf_spatial_bias=false,
    add_beta_decay=false,
    add_a_baseline=false,
    add_b_baseline=false,
    add_γ2=false,
    add_depletion_factor=false,
    add_retain_belief=false,
    delay_turn_bias=false,
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
    @show add_beta_decay
    @show add_a_baseline
    @show add_b_baseline
    @show add_γ2
    @show add_depletion_factor
    @show add_retain_belief
    @show delay_turn_bias
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
    if add_beta_decay
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "beta_decay")
    end
    if add_a_baseline
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "a_baseline")
    end
    if add_b_baseline
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "b_baseline")
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

        if add_beta_decay
            beta_decay = unitnorm(params[i])
            i += 1
        else
            beta_decay = 0.0
        end

        if add_a_baseline
            a_baseline = exp(params[i])
            i += 1
        else
            a_baseline = 1.0
        end

        if add_b_baseline
            b_baseline = exp(params[i])
            i += 1
        else
            b_baseline = 1.0
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

        i += 1

        return beta_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, beta_decay, a_baseline, b_baseline, γ2, depletion_factor, retain_belief, delay_turn_bias, rewscaled, add_leaf, false)
    end

    em_opt_extended(data,subs,X,initbetas,initsigma,fn,varnames; emtol, maxiter, full, extended, quiet, startx=nothing)
end

""" Compute trial-by-trial Q values and other variables from fitted beta-bernoulli model
df: DataFrame containing the data
results: EMResults object from run_beta_lik
add_leaf - if true, include leaf choice in likelihood
delay_turn_bias - if true, apply turn bias after computing stay/go probabilities
rewscaled - if true, rescale Q values to be centered around 0
params - optional dictionary of parameters to override those in results
subjlevel - symbol for column in df indicating subject/daynum (either :daynum or
:daysessionnum)

Returns: DataFrame with original data plus trial-by-trial variables
"""

function find_Q_vals_beta_lik(df, results; add_leaf, delay_turn_bias, rewscaled, params=nothing, subjlevel=:daynum)
    data = copy(df)
    data[:, :sub] = data[:, subjlevel]
    nsubjs = maximum(data.sub)
    liks = zeros(nsubjs)
    dfs = []
    for i in 1:nsubjs
        (liks[i], df) = beta_lik(view(data, data.sub .== i, :), results;
        subject=i, add_leaf, delay_turn_bias, rewscaled, params, record=true)
        push!(dfs, df)
    end
    record_df = vcat(dfs...) # Combine all session results
    hcat(data, record_df) # Append columns to the original data
end