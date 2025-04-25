module DataIO

using Match
using GZip
using Printf

using SparseArrays

export root
export read_ply, read_obj
export read_mtx, read_features, read_barcodes, read_matrix, expand_matrix, read_dge

# ------------------------------------------------------------------------
# globals

const root = "$(homedir())/root/data/seqspace"
panic(msg) = error(msg)

# ------------------------------------------------------------------------
# geometric file formats

"""
    const plytype = Dict{String, Type}

Lookup table for keywords of PLY file format matched to type information.
"""
const plytype = Dict{String, Type}(
    "char"   => Int8,    "int8"    => Int8,
    "short"  => Int16,   "int16"   => Int16,
    "int"    => Int32,   "int32"   => Int32,
    "long"   => Int64,   "int64"   => Int64,

    "uchar"   => UInt8,  "uint8"   => UInt8,
    "ushort"  => UInt16, "uint16"  => UInt16,
    "uint"    => UInt32, "uint32"  => UInt32,
    "ulong"   => UInt64, "uint64"  => UInt64,

    "float"  => Float32, "float32" => Float32,
    "double" => Float64, "float64" => Float64
)

readwords(io::IO) = split(readline(io))

"""
    struct PLYProp
        name :: Symbol
        type :: Type
        len  :: Union{Type,Nothing}
    end

Data structure that encapsulates a property of a PLY file.
If `len` is not nothing, it is assumed to be a collection of properties, i.e. a list.
"""
struct PLYProp
    name :: Symbol
    type :: Type
    len  :: Union{Type,Nothing}
end

PLYProp(word) = if word[1] == "list"
                    PLYProp(Symbol(word[4]), plytype[word[3]], plytype[word[2]])
                else
                    PLYProp(Symbol(word[2]), plytype[word[1]], nothing)
                end

"""
    fill!(prop::Array{PLYProp}, io::IO)

Read and parse a single line from `io` stream.
Fill the interpreted property into `prop`.
"""
function fill!(prop::Array{PLYProp}, io::IO)
    word = readwords(io)
    while word[1] == "property" && length(word) >= 3
        push!(prop, PLYProp(word[2:end]))
        word = readwords(io)
    end

    return word
end

"""
    readprops(io::IO, props::Array{PLYProp,1})

Read a collection of properties from `io` stream into `props` array.
Return a named tuple of parsed properties.
"""
function readprops(io::IO, props::Array{PLYProp,1})
    data = (p) -> begin
        if isnothing(p.len)
            return read(io,p.type)
        else
            len = read(io,p.len)
            return [read(io,p.type) for _ in 1:len]
        end
    end

    return (; (prop.name => data(prop) for prop in props)...)
end

"""
    read_ply(io::IO)

Read an oriented mesh from `io` stream assuming bytes are formatted in PLY file format.
"""
function read_ply(io::IO)
    # magic
    line = readline(io)
    if line != "ply"
        panic("can not interpret byte stream as PLY data")
    end

    # file format
    word = readwords(io)
    if word[1] != "format" || word[3] != "1.0"
        panic("unrecognized format of PLY file")
    end
    # FIXME: we just ignore this for now and assume little endian binary
    fmt = Symbol(word[2])

    # data layout
    n₀, n₂ = 0, 0
    field₀ = PLYProp[] # list of fields of vertices
    field₂ = PLYProp[] # list of fields of faces

    @label getline # ----------------------------
    word = readwords(io)

    @label doline
    if word[1] == "end_header" 
        @goto endloop
    end

    if word[1] != "element" || length(word) != 3
        @goto getline
    end

    @match word[2] begin
        "vertex" => begin
            n₀   = parse(Int, word[3])
            word = fill!(field₀, io)
            @goto doline
        end
        "face" => begin
            n₂   = parse(Int, word[3])
            word = fill!(field₂, io)
            @goto doline
        end
    end

    @goto  getline

    @label endloop # ----------------------------

    return (
        verts = [readprops(io,field₀) for _ in 1:n₀], 
        faces = [readprops(io,field₂) for _ in 1:n₂]
    )
end

"""
    read_obj(io::IO)

Read an oriented mesh from `io` stream assuming bytes are formatted in OBJ file format.
"""
function read_obj(io::IO)
    bufᵥ = []
    bufₙ = []
    bufₜ = []

    for ln in eachline(io)
        if startswith(ln, "#") || length(ln) == 0
            continue
        end

        w = split(ln)
        @match w[1] begin
            "v"  => push!(bufᵥ, [parse(Float32, x) for x in w[2:end]])
            "vₙ" => push!(bufₙ, [parse(Float32, x) for x in w[2:end]])
            "f"  => push!(bufₜ, [parse(Int32, split(x,"//")[1]) for x in w[2:end]])
             _   => continue
        end
    end

    return (
        verts = hcat(bufᵥ...)',
        norms = hcat(bufₙ...)',
        faces = hcat(bufₜ...)',
    )
end

# ------------------------------------------------------------------------
# matrix market exchange format

"""
    read_mtx(io::IO)

Read a matrix from `io` stream assuming bytes are formatted in matrix exchance file format.
"""
function read_mtx(io::IO)
    header = readwords(io)
    if header[1] != "%%MatrixMarket"
        panic("can not interpret given IO stream as matrix market exchange")
    end

    T = 
    if header[4] == "integer"
        Int
    elseif header[4] == "real"
        Float64
    elseif header[4] == "complex"
        Complex
    elseif header[4] == "pattern"
        Bool
    else
        panic("unrecognized element type '$(header[4])' in matrix file")
    end

    D = 
    if header[2] == "matrix"
        2
    elseif header[2] == "vector"
        1
    else
        panic("unrecognized data type '$(header[2])' in matrix file")
    end

    data = 
    if header[3] == "array"
        line = readwords(io)
        while line[1][1] == '%'
            line = readwords(io)
        end

        m, n  = parse.(Int, line[1:2])
        array = Array{T,D}(undef, m, n)

        for i in 1:m
            for j in 1:n
                line = getline(io)
                array[i,j] = parse(T, line)
            end
        end

        array
    elseif header[3] == "coordinate"
        line = readwords(io)
        while line[1][1] == '%'
            line = readwords(io)
        end

        m, n, numel = parse.(Int, line[1:3])
        row = Array{Int,1}(undef, numel)
        col = Array{Int,1}(undef, numel)
        val = Array{T,1}(undef, numel)

        for (i,line) in enumerate(eachline(io))
            word = split(line)
            row[i] = parse(Int, word[1])
            col[i] = parse(Int, word[2])
            val[i] = parse(T, word[3])
        end

        sparse(row, col, val, m, n)
    else
        panic("unrecognized data format '$(header[3])'")
    end
end

# NOTE: assumes Gene Expression is the only feature type
read_features(io::IO) = [split(line)[2] for line in eachline(io)]
read_barcodes(io::IO) = [line for line in eachline(io)]

# ------------------------------------------------------------------------
# scRNAseq file formats

"""
    read_matrix(io::IO; type=Float64, named_cols=false, named_rows=false, start_cols=1, start_rows=1)

Read a white-space delimited matrix from `io` stream of element type `type`.
If `named_cols` is true, the first line is assumed to be column labels.
If `named_rows` is true, the first column of each row is assumed to be denote the label.
"""
function read_matrix(io::IO; type=Float64, named_cols=false, named_rows=false, start_cols=1, start_rows=1)
    cols = named_cols ? readwords(io)[start_cols:end] : nothing

    x = position(io)
    ncols = if named_cols
        length(cols)
    else
        words = readwords(io)
        n = length(words) - (named_rows ? 1 : 0)
        seek(io, x)

        n
    end
    nrows = sum(1 for row in eachline(io))
    seek(io, x)

    data = Array{type, 2}(undef, nrows, ncols)
    rows = named_rows ? Array{String, 1}(undef, nrows) : nothing

    for (i,row) in enumerate(eachline(io))
        entry = split(row)
        if named_rows
            rows[i] = entry[1]
            entry   = entry[2:end]
        end
        data[i,:] = parse.(type, entry) 
    end

    return data, rows, cols
end

"""
    expand_matrix(io::IO, dir::String)

Read a matrix in MTX format from `io` stream and dump as 3 text files into directory `dir`.
"""
function expand_matrix(io::IO, dir::String)
    if isdir(dir)
        error("directory exists")
    end
    mkdir(dir)

    data, genes, barcodes = read_matrix(io; type=Int, named_cols=true, start_cols=3, named_rows=true)

    open("$dir/barcodes.tsv", "w") do fd
        write(fd, barcodes[1])
        for barcode ∈ barcodes[2:end]
            write(fd, "\n", barcode)
        end
    end

    feature(i, gene) = ((@sprintf "Gene%04d" i), "\t", gene, "\t", "Gene Expression") 
    open("$dir/features.tsv", "w") do fd
        write(fd, feature(1, genes[1])...)
        for (i, gene) in enumerate(genes[2:end])
            write(fd, "\n", feature(i+1, gene)...)
        end
    end

    nnz(x) = sum(x[:] .> 0)
    open("$dir/matrix.mtx", "w") do fd
        write(fd, "%%MatrixMarket matrix coordinate integer general", "\n")
        write(fd, "%", "\n")
        write(fd, string(size(data,1)), " ", string(size(data,2)), " ", string(nnz(data)))
        for j in 1:size(data,2)
            for i in findall(data[:,j] .> 0)
                write(fd, "\n", string(i), " ", string(j), " ", string(data[i,j]))
            end
        end
    end
end

function read_dge(io::IO; type=Float64, named_cols=false, named_rows=false, start_cols=1, start_rows=1)
    # Read column headers if needed
    cols = nothing
    if named_cols
        first_line = readline(io)
        if named_rows
            cols = split(first_line)[(start_cols+1):end]  # Skip the first column (row names)
        else
            cols = split(first_line)[start_cols:end]
        end
    end

    # Store current file position to avoid consuming lines
    x = position(io)

    # Count number of rows without consuming data
    nrows = sum(1 for _ in eachline(io)) - start_rows
    seek(io, x)  # Go back to start

    # Initialize data storage
    data = Array{type,2}(undef, nrows, length(cols))
    rows = named_rows ? Array{String,1}(undef, nrows) : nothing

    # Read data line by line
    for (i, row) in enumerate(eachline(io))
        if i <= start_rows
            continue
        end
        idx = i - start_rows  # Adjust index for storage
    
        entry = split(row)  # Split on whitespace
    
        if named_rows
            rows[idx] = entry[1]      # First column is gene name
            entry = entry[2:end]        # Remove the gene name
        end
    
        data[idx, :] = parse.(type, entry)
    end

    return data, rows, cols
end

# ------------------------------------------------------------------------
# point of entry for testing

function test()
    mesh = open("$root/gut/mesh_apical_stab_000153.ply") do io
        read_ply(io)
    end

    bdntp, genes = GZip.open("$root/dvex/bdtnp.txt.gz") do io
        data, _, cols = read_matrix(io; named_cols=true)
        return data, cols
    end

    scrna, genes = GZip.open("$root/dvex/dge_normalized.txt.gz") do io
        data, rows, _ = read_matrix(io; named_rows=true, named_cols=true)
        return data, rows
    end
end

end
