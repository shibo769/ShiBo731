using DataFrames
using CSV
using Plots
using Dates
using Statistics
using Weave
using Distributions
cd("D:\\Courses\\731\\HWs\\HW1")
""" Loading data """
data = CSV.read("SPData.csv", DataFrame)
data.Date = reverse(data.Date)
rename!(data, [:Date,:Close])
data.Close = reverse(data.Close)

""" Set Parameters """
WindowSize = 100
Δ = 1 / 252
λ₁ = 0.94
λ₂ = 0.97
logret = log.(data.Close[2:end]) - log.(data.Close[1:end-1])

""" MA """
result = DataFrame()
result[!, :Date] = data.Date[101:end]
MA = []
for i in 1:length(logret)-WindowSize+1
    append!(MA, var(logret[i:i+WindowSize-1]) / Δ)
end
result[!, :MA] = MA
plot(result.Date,result.MA, xformatter = Plots.dateformatter, xrotation = 45, xlabel = "Date",
     title = "Moving Average σ", labels = "MA")

""" EWMA """
σ₀ = var(logret[1:WindowSize])
EWMA₉₄ = [σ₀]
EWMA₉₇ = [σ₀]
logret² = logret[WindowSize+1:end] .^ 2
for i in 2:length(logret)-WindowSize+1
    temp1 = λ₁*EWMA₉₄[i-1] + (1-λ₁)*logret²[i-1]
    append!(EWMA₉₄, temp1)
    temp2 = λ₂*EWMA₉₇[i-1] + (1-λ₂)*logret²[i-1]
    append!(EWMA₉₇, temp2)
end

EWMA₉₄ = EWMA₉₄ / Δ
EWMA₉₇ = EWMA₉₇ / Δ
""" Part 1 Continue """
result[!, :EWMA94] = EWMA₉₄
result[!, :EWMA97] = EWMA₉₇

plot(result.Date, [result.MA, result.EWMA94, result.EWMA97], xlabel = "Date",
     ylabel = "σ", title = "MA and EWMA time-series plot",
     xrotation = 45, ylims = [0,1], legend = true,
     labels = ["MA" "EWMA-94" "EWMA-97"])

# weave("HW1.jl", out_path = :pwd, doctype="md2pdf")
""" Part 1 (a) finished"""
using ARCHModels
ind = findall(x -> x == "2020/2/28", Array(data.Date))
Garch = fit(GARCH{1, 1}, logret[1:1152])
ω,β,α =  Garch.spec.coefs
Vₗ = ω / (1-α-β) / Δ
nsim = 100

Vₐ = var(logret[1152+1:end]) / Δ

Zₜ = rand(Normal(0,1), (59, nsim))
simu_result = DataFrame()

for i in 2:61
    simu_array = [Vₗ]
    append!(simu_array, simu_array .+ (ω .+ simu_array[i-1] .* (α .* Zₜ[1:end, i-1].^2 .+ β)))
    simu_result = hcat(simu_result, simu_array, makeunique = true)
end

plot(Array(simu_result), title = "Two Month Sample Path for Annulized σ",
     ylabel = "Annulized σ", xlabel = "Date", legend = false)

plot(Array(simu_result))
