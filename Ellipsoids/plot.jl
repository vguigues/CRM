using Plots

"""
plot_elipsoid(Q, y, k; num_points=100)

plot an elipsoid
- Q::AbstractMatrix: size(Q) = (n,n), elipsoid matrix
- y::AbstractVector: size(y) = (m,n), elipsoid center
- k::Real: elipsoid radius
- num_points::Int64=100: number of points plotted
"""
function plot_elipsoid(p1, Q::AbstractMatrix, y::AbstractVector, k::Real,
    elip_label::String; num_points::Int64=100)
    θ = range(0, 2π, length=num_points)
    circle = [[cos(t), sin(t)] for t in θ]

    L = cholesky(Q).L
    T = k * inv(L)

    ellip_points = [T * p + y for p in circle]

    xs = getindex.(ellip_points, 1)
    ys = getindex.(ellip_points, 2)

    plot!(p1, xs, ys, label=elip_label, lw=2)
end

"""
plot_all_elipsoids(y::AbstractMatrix, Q::AbstractArray, k::AbstractVector)

Builds figure with the set of elipsoids plotted
- y::AbstractMatrix:    elipsoid centers 
- Q::AbstractArray:     elipsoid matrices 
- k::AbstractVector:    elipsoid radii
"""
function plot_all_elipsoids(p1, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector)
    m, n = size(y)
    @assert n == 2 "Only works for 2D"

    for i in 1:m
        Q_i = Q[i, :, :]
        y_i = y[i, :]
        k_i = k[i]

        plot_elipsoid(p1, Q_i, y_i, k_i, "Ellipsoid $i")
        # scatter!([y_i[1]], [y_i[2]], label="", marker=(:circle, 4))
    end
end


function plot_1ball(p1::Plots.Plot)
    θ = LinRange(0, 2π, 100)
    x_circle = cos.(θ)
    y_circle = sin.(θ)
    plot!(p1, x_circle, y_circle, label="1-ball", linestyle=:dash, color=:blue)
end