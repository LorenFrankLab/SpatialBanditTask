using DataFrames
using CSV

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

function cont_combination_old(s)
    parse.(Int, split(s, "0"; keepempty=false)) ./ 10
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