using DataFrames
using CSV
using SpecialFunctions

# Including this to fix an issue with Julia 1.6
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

# Old version separated trials by zeros
# Works for base but not depletion condition
function cont_combination_old(s)
    parse.(Int, split(s, "0"; keepempty=false)) ./ 10
end

# New version assumes strings are properly padded to 12 characters
function cont_combination_new(s)
    [
        parse(Int, s[1:2]),
        parse(Int, s[3:4]),
        parse(Int, s[5:6]),
        parse(Int, s[7:8]),
        parse(Int, s[9:10]),
        parse(Int, s[11:12]),
    ] ./ 10
end

"""
Return a dataframe for the animal

    animal<string>: Name of the animal
    depletion<bool, default false>: Whether to use the non-depletion or depletion data for this anmal

    Coding:
    reward: 0/1
    rewscaled: -1/1

    stem: A/B/C
    stemchoice: 1-3
    leaf: 1-6
    leafchoice: 1-2 (conditional on stem)

    trial: 0-179
    session: 1-?, per-day session number, e.g. 1 is first session each day
    daynum: 1-?, contiguous day numbering
    daysessionnum: 1-?, unique session number continued across days

"""
function load_animal(animal, filepath; depletion=false)
    # csv_file to estimate value for
    if depletion
        csv_file = animal * "_decay_default_original_csv_format.csv"
        fullpath = joinpath(filepath, "data/new", csv_file)
        # mkdir(fullfile(filepath,["hmm_decay_",animal]))
    else
        csv_file = animal * "_clean_contingencies_only_parsed_data.csv"
        fullpath = joinpath(filepath, "data", csv_file)
        # mkdir(fullfile(filepath,['hmm_',animal]))
    end
    df = DataFrame(CSV.File(fullpath, drop=[1]))  # Drop index column

    if animal == "senor"
        df.date = parse.(Int, [x[6:end] for x in df.date])
    end

    annotate_data!(df; depletion)
    return df
end

"""
Version for remote format
"""
function load_animal_dj(animal, filepath)
    df = DataFrame(CSV.File(filepath, drop=[1]))  # Use filepath instead of fullpath, drop index column
    annotate_data!(df)
    return df
end

function annotate_data!(df; depletion=false)
    # recode some variables
    df.rewscaled = 2 * df.reward .- 1
    df.stemchoice = [df[i,:stem][1] - 'A' + 1 for i in 1:nrow(df)]
    df.leafchoice = 1 .+ mod.(df.leaf.+1,2)

    # Number days 1-n
    dates = unique(df.date)
    df.daynum = [minimum(findall(df[i,:date] .== dates)) for i in 1:nrow(df)]

    # Code sessions contiguously across days
    datesessions = string.(df.date) .* "_" .* string.(df.session)
    uniq_datesessions = unique(datesessions)
    df.daysessionnum = [minimum(findall(datesessions[i] .== uniq_datesessions)) for i in 1:nrow(df)]

    # Stem switches
    df[!, :stemswitch] .= false
    prevstem = 0
    prevsession = 0
    for i in axes(df, 1)
        session = df[i, :daysessionnum]
        stem = df[i, :stem]
        if (session == prevsession)
            if (stem != prevstem)
                df[i, :stemswitch] = true
            end
        end
        prevstem = stem
        prevsession = session
    end

    # Mark length of runs pre- and post-switch
    df[!, :n_post_switch] .= -1
    switch_happened = false
    n_post_switch = 0
    prev_session = 0
    for i in axes(df, 1)
        session = df[i, :daysessionnum]
        if (session != prev_session)
            switch_happened = false
            n_post_switch = -1
        else
            if df[i, :stemswitch]
                switch_happened = true
                n_post_switch = -1
            end
            if switch_happened
                n_post_switch += 1
            end
        end
        df[i, :n_post_switch] = n_post_switch
        prev_session = session
    end

    # backwards sweep
    df[!, :n_pre_switch] .= -1
    switch_happened = false
    n_pre_switch = 0
    prev_session = 0
    for i in reverse(axes(df, 1))
        session = df[i, :daysessionnum]
        if (session != prev_session)
            switch_happened = false
            n_pre_switch = -1
        else
            if df[i, :stemswitch]
                switch_happened = true
                n_pre_switch = -1
            end
            if switch_happened
                n_pre_switch += 1
            end
        end
        df[i, :n_pre_switch] = n_pre_switch
        prev_session = session
    end

    # Old method of separating contingencies by zeros
    # df.cont_combo_old = cont_combination_old.(string.(df.contingency))
    df.contingency = string.(df.contingency)

    # A few contingencies are truncated if they have a leading 0
    # Pad them out
    short_contingencies = findall(length.(df.contingency) .== 9)
    for i in short_contingencies
        df.contingency[i] = "0$(df.contingency[i])"
    end
    short_contingencies = findall(length.(df.contingency) .== 10)
    for i in short_contingencies
        df.contingency[i] = "0$(df.contingency[i])"
    end
    short_contingencies = findall(length.(df.contingency) .== 11)
    for i in short_contingencies
        df.contingency[i] = "0$(df.contingency[i])"
    end
    df.cont_combo = cont_combination_new.(df.contingency)

    # For depletion data, we may want to label the true (underlying) contingencies
    if depletion
        # Fix contingencies
        firstrows = combine(first, groupby(df, :daysessionnum))
        firstrows.contingency_base .= firstrows.contingency
        # Merge
        x = leftjoin(df[:, [:daysessionnum, :contingency]], firstrows[: ,[:daysessionnum, :contingency_base]], on=:daysessionnum)
        # Get rid of union type with null?
        df.contingency_base .= String.(x.contingency_base)
    else
        df.contingency_base .= df.contingency
    end
    df.cont_base_combo = cont_combination_new.(df.contingency_base)

    return df
end

"""
Load both depletion and non-depletion data together.
Depletion data is shifted to have unique daynum numbers following
non-depletion sessions
"""
function load_animal_combined(animal, filepath)
    nondep = load_animal(animal, filepath, depletion=false)
    nondep.depletion .= false
    nondep.daysessionnum_cond .= nondep.daysessionnum
    dep = load_animal(animal, filepath, depletion=true)
    dep.daynum .+= maximum(nondep.daynum)
    dep.daysessionnum_cond .= dep.daysessionnum
    dep.daysessionnum .+= maximum(nondep.daysessionnum)
    dep.depletion .= true
    combined = vcat(nondep, dep)
    return combined
end

# This loads all animals together,
# with data shifted to have unique daynum and daysessionnum numbers
# across both non-depletion and depletion sessions
function load_allanimals_combined(filepath)
    combined = []
    for animal in animals
        nondep = load_animal(animal, filepath, depletion=false)
        nondep.depletion .= false
        nondep.animal .= animal
        nondep.daysessionnum_cond .= nondep.daysessionnum
        if length(combined) > 0
            nondep.daynum .+= maximum(combined.daynum)
            nondep.daysessionnum .+= maximum(combined.daysessionnum)
            combined = vcat(combined, nondep)
        else
            combined = nondep
        end
        dep = load_animal(animal, filepath, depletion=true)
        dep.daynum .+= maximum(combined.daynum)
        dep.animal .= animal
        dep.daysessionnum_cond .= dep.daysessionnum
        dep.daysessionnum .+= maximum(combined.daysessionnum)
        dep.depletion .= true
        combined = vcat(combined, dep)
    end
    return combined
end

function load_allanimals_depletion(filepath)
    animal = animals[1]
    dep = load_animal(animal, filepath, depletion=true)
    dep.depletion .= true
    dep.animal .= animal
    dep.daysessionnum_cond .= dep.daysessionnum
    combined = dep
    for animal in animals[2:end]
        dep = load_animal(animal, filepath, depletion=true)
        dep.depletion .= true
        dep.animal .= animal
        dep.daysessionnum_cond .= dep.daysessionnum
        dep.daynum .+= maximum(combined.daynum)
        dep.daysessionnum .+= maximum(combined.daysessionnum)
        combined = vcat(combined, dep)
    end
    return combined
end

function load_all_data(filepath)
    dfs = []
    for animal in animals
        df = load_animal(animal, filepath)
        df[!, :sub] .= animal
        push!(dfs, df)
    end
    vcat(dfs...)
end

function unitnorm(x)
    0.5 + 0.5 * erf(x / sqrt(2))
end

function invunitnorm(x)
    sqrt(2) * erfinv(2x - 1)
end