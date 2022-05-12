using Statistics
using StatsFuns
using SpecialFunctions
using EM
# this is the likelihood function for the actual model

function qlik(data, βgo::U, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, retain_belief, α, rewscaled::Bool, add_leaf::Bool, record::Bool) where U
    """
    learn_q: If true, learn Q values trial by trial.
             If false, use provided data[trial, :q1], data[trial, :q2] etc
    mode: "lik": Return neg likelihood
          "sim": Return Q values
    mode: 

    Q is the baseline Q value estimate for each state
    Qeff is Q plus (potentially):
        - Spatial bias
        - Stay bias
        - 

    """
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
    prevs::Int = 0 #stem
    prevl::Int = 0 #leaf

    Q = zeros(U,3,2,length(c1)+1)
    Qeff = zeros(U,3,2,length(c1)+1)
    Qstem = zeros(U,3,1)
    Qleaf = zeros(U,2)

    for i = 1:length(c1)
        if ((i>1) && (data.session[i] != data.session[i-1]))
            Q[:,:,i] = retain_belief * Q[:,:,i]
            prevs = 0
            prevl = 0
        end
        Qeff[:,:,i]=Q[:,:,i]

        # this is the (scaled) value of switching to each alternative stem
        # mean averages over both leaves
        Qstem .= βgo .* mean(Q[:,:,i],dims=2)

        # spatial bias
        # 1 -> 2, 2->3, 3->1
        if prevs > 0
            other_stem = ((prevs + 3) % 3) + 1
            Qstem[other_stem] = Qstem[other_stem] + spatial_bias[prevs] + turn_bias
            Qeff[other_stem,:,i] = Qeff[other_stem,:,i] .+ spatial_bias[prevs] .+ turn_bias
        end


        # this is the (scaled) value of staying with the current stem
        # uses only the value of the next leaf
        # plus the bias toward staying
        if (prevs > 0)
            Qstem[prevs] = βstay * (Q[prevs,3-prevl,i] + γ2 * Q[prevs,prevl,i]) + stay_bias
            Qeff[prevs,:,i] = βstay .* Qeff[prevs,:,i] .+ stay_bias
        end

        # log likelihood of stem choice (log of logistic)
        lik += Qstem[c1[i]] - logsumexp(Qstem)
        if ((i>1) && c1[i]!=c1[i-1])
            lik_go+=Qstem[c1[i]] - logsumexp(Qstem)
        else
            lik_stay+=Qstem[c1[i]] - logsumexp(Qstem)
        end

        ## Leaf
        if ((prevs != c1[i]) && add_leaf)
            Qleaf .= Q[c1[i], :, i]
            # Add leaf bias
            Qleaf[1] += leaf_turn_bias + leaf_spatial_bias[c1[i]]
            # lik += Qeff[c1[i]] - logsumexp(Qeff[setdiff(1:3,prevs)])
            # log likelihood of leaf choice on switches only
            # Qeff_all[c1[i],c2[i],i]=βleaf * Q[c1[i],c2[i],i];
            lik += βleaf * Qleaf[c2[i]] - logsumexp(βleaf .* Qleaf);
        end

        # learn about the chosen leaf
        Q[:,:,i+1] = Q[:,:,i]
        Q[c1[i],c2[i],i+1] = (1-α) * Q[c1[i],c2[i],i] + α * r[i]

        prevs = c1[i]
        prevl = c2[i]
    end
    
    if record
        return (-lik, Q[:,:,1:length(c1)])
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
Function for EM to allow params to vary from -inf to inf
"""
function qlik_em(data, βgo::U, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, retain_belief, α, rewscaled::Bool, add_leaf::Bool, record::Bool) where U
    αnorm = 0.5 + 0.5 * erf(α / sqrt(2))
    γ2norm = 0.5 + 0.5 * erf(γ2 / sqrt(2))
    retain_belief_norm = 0.5 + 0.5 * erf(retain_belief / sqrt(2))
    qlik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2norm, retain_belief_norm, αnorm, rewscaled, add_leaf, record)
end

function run_q(df; maxiter=100, emtol=1e-3, full=true, extended=false,
    add_βleaf=false,
    add_stay_bias=false,
    add_turn_bias=false,
    add_spatial_bias=false,
    add_leaf_turn_bias=false,
    add_leaf_spatial_bias=false,
    add_leaf=true,
    add_γ2=false,
    add_retain_belief=false,
    rewscaled=false,
    )

    data = copy(df)
    data[:, :sub] = data[:, :daynum]
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...

    initbetas = [0 0]
    initsigma = [5, 5]
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
    if add_retain_belief
        initbetas = hcat(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "retain_belief")
    end

    initbetas = hcat(initbetas, 0)
    push!(initsigma, 1)
    push!(varnames, "α")

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

        if add_retain_belief
            retain_belief = 0.5 + 0.5 * erf(params[i] / sqrt(2))
            i += 1
        else
            retain_belief = 0.0
        end

        α = 0.5 + 0.5 * erf(params[i] / sqrt(2))
        i += 1

        return qlik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, leaf_turn_bias, leaf_spatial_bias, γ2, retain_belief, α, rewscaled, add_leaf, false)
    end

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=emtol, full=full, maxiter=maxiter);
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

run_q_leaf(data; kwargs...) = run_q(data; add_βleaf=true, kwargs...)
run_q_leaf_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_γ2=true, kwargs...)
run_q_leaf_retainbelief(data; kwargs...) = run_q(data; add_βleaf=true, add_retain_belief=true, kwargs...)
run_q_leaf_stay(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, kwargs...)
run_q_leaf_stay_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_γ2=true, kwargs...)
run_q_leaf_stay_retainbelief(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_retain_belief=true, kwargs...)

run_q_leaf_stay_turn(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, kwargs...)
run_q_leaf_stay_spatial(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, kwargs...)
run_q_leaf_stay_turn_leafspatial(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q_leaf_stay_turn_leafturn(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, kwargs...)
run_q_leaf_stay_spatial_leafspatial(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, kwargs...)
run_q_leaf_stay_spatial_leafturn(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, kwargs...)

run_q_leaf_stay_turn_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_γ2=true, kwargs...)
run_q_leaf_stay_spatial_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_γ2=true, kwargs...)
run_q_leaf_stay_turn_leafspatial_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q_leaf_stay_turn_leafturn_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
run_q_leaf_stay_spatial_leafspatial_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_spatial_bias=true, add_γ2=true, kwargs...)
run_q_leaf_stay_spatial_leafturn_γ2(data; kwargs...) = run_q(data; add_βleaf=true, add_stay_bias=true, add_spatial_bias=true, add_leaf_turn_bias=true, add_γ2=true, kwargs...)
