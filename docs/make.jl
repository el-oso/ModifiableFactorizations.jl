using Documenter, DocumenterVitepress, ModifiableFactorizations

makedocs(;
    sitename = "ModifiableFactorizations.jl",
    modules = [ModifiableFactorizations],
    repo = Documenter.Remotes.GitHub("el-oso", "ModifiableFactorizations.jl"),
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/el-oso/ModifiableFactorizations.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Construction" => "construction.md",
        "Updating and downdating" => "updating.md",
        "Q representations" => "q_representations.md",
        "Benchmarks" => "benchmarks.md",
        "Provenance" => "provenance.md",
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/ModifiableFactorizations.jl",
    push_preview = true,
)
