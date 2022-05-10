"""
Pretty-print group-level means and SDs.

results: EMResults
mu: bool, whether to show means
sigma: bool, whether to show SDs
transforms: optionally, a dictionary of columns names to functions
    Intended for variables which are transformed prior to use in the likelihood
    For column x, x* is added applying func
"""
function show_results(results; mu=true, sigma=true, transforms=nothing)
    io = IOBuffer()
    
    varnames = results.varnames
    betas = results.betas
    if !isnothing(transforms)
        for (k,fn) in transforms
            inds = findall(varnames .== k)
            if length(inds) > 0
                ind = inds[1]
                varnames = vcat(varnames, "*"*varnames[ind])
                new_betas = zeros(size(betas, 1))
                new_betas[1] = fn(betas[1,ind])
                for j in 2:size(betas, 1)
                    new_betas[j] = fn(betas[1,ind] + betas[j,ind]) - new_betas[1]
                end
                betas = hcat(betas, new_betas)
            end
        end
    end
    
    varnames_short = copy(varnames)
    for i in eachindex(varnames_short)
        if length(eachindex(varnames_short[i])) > 5
            varnames_short[i] = varnames_short[i][1:nextind(varnames[i],0,5)]
        end
        varnames_short[i] *= " "^(6 - length(eachindex(varnames_short[i])))
    end
    
    print(io, "<pre>")
    print(io, "$(varnames)\n")
    print(io, "</pre>")

    if mu
        print(io, "β:<br/>")
        print(io, "<pre>")
        print(io, "$(" " * join(varnames_short))\n")
        show(io, MIME"text/plain"(), round.(betas; digits=2))        
        if results isa EMResultsExtended
            pvalues = reshape(results.pvalues,size(results.betas'))'
            print(io, "<br/>p:<br/>")
            show(io, MIME"text/plain"(), round.(pvalues, RoundUp; digits=2))
        end
        print(io, "</pre>")
    end
    
    if sigma
        print(io, "σ²:<br/>")
        print(io, "<pre>")
        show(io, MIME"text/plain"(), round.(results.sigma; digits=2))
        print(io, "</pre>")
    end
    
    my_string = String(take!(io))
    my_string = replace(my_string, r"[0-9]+×[0-9]+ Matrix{Float64}:\n" => "")
    my_string = replace(my_string, r"(-[0-9\.]+)" => s"<span style='color: red'>\1</span>")
    my_string = replace(my_string, "\n" => "<br/>")
    
    my_string
end

function group_cov(results)
    io = IOBuffer()
    
    x = repeat(sqrt.(diag(results.sigma)), outer=[1, size(results.sigma)[1]])
    cov_scaled = results.sigma ./ x ./ x'
    
    print(io, "Scaled Group-Level Covariances:<br/>")
    print(io, "<pre>")
    print(io, "$(results.varnames)\n")
    show(io, MIME"text/plain"(), round.(cov_scaled; digits=2))
    print(io, "</pre>")
    
    my_string = String(take!(io))
    my_string = replace(my_string, r"[0-9]+×[0-9]+ Matrix{Float64}:\n" => "")
    my_string = replace(my_string, r"(-[0-9\.]+)" => s"<span style='color: red'>\1</span>")
    my_string = replace(my_string, "\n" => "<br/>")
    
    my_string
end

function subject_cov(results; scale=true)
    io = IOBuffer()
    
    if scale
        avg_h = dropdims(mean(results.h,dims=3),dims=3)
        x = repeat(sqrt.(diag(avg_h)), outer=[1, size(avg_h)[1]])
        cov_scaled = avg_h ./ x ./ x'

        print(io, "Scaled Average Subject Covariances:<br/>")
        print(io, "<pre>")
        print(io, "$(results.varnames)\n")
        show(io, MIME"text/plain"(), round.(cov_scaled; digits=2))
        print(io, "</pre>")
    else
        avg_h = dropdims(mean(results.h,dims=3),dims=3)
        print(io, "Average Subject Covariances:<br/>")
        print(io, "<pre>")
        print(io, "$(results.varnames)\n")
        show(io, MIME"text/plain"(), round.(avg_h; digits=2))
        print(io, "</pre>")
    end
    
    my_string = String(take!(io))
    my_string = replace(my_string, r"[0-9]+×[0-9]+ Matrix{Float64}:\n" => "")
    my_string = replace(my_string, r"(-[0-9\.]+)" => s"<span style='color: red'>\1</span>")
    my_string = replace(my_string, "\n" => "<br/>")
    
    my_string
end