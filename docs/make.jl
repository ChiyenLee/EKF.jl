using Documenter
using EKF

makedocs(
    sitename = "EKF.jl",
    format = Documenter.HTML(prettyurls = false),
    pages = ["Introduction" => "index.md", "API" => "api.md"],
)

deploydocs(repo = "github.com/ChiyenLee/EKF.jl.git", devbranch = "main")
