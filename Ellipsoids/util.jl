

"""
g(i::Int, x::AbstractVector, y::AbstractArray, Q::AbstractArray, k::AbstractVector) 

Returns the distance of x to the elipsoid (y,Q,k)
- x::AbstractVector: size(x) = n, current solution
- y::AbstractVector: size(y) = (m,n), elipsoid center
- Q::AbstractMatrix: size(Q) = (n,n), elipsoid matrix
- k::Real: elipsoid radius
"""
function g(x::AbstractVector, y::AbstractVector, Q::AbstractMatrix, k::Real)
	return (x - y)' * Q * (x - y) - k^2
end

"""
g_prime(i::Int, x::AbstractVector, y::AbstractVector, Q::AbstractMatrix)

Returns the gradient of the distance function of  x to the elipsoid (y,Q,k)

- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (n), elipsoid center
- Q::AbstractArray:     size(Q) = (n,n), elipsoid matrix
- k::Real:             elipsoid radius
"""
function g_prime(x::AbstractVector, y::AbstractArray, Q::AbstractArray)
	return 2.0 * (Q * (x - y))
end

"""
max_violation(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector, η::Real, ϵ::Real)

Computes the maximum violation for the current solution
- x::AbstractVector:    size(x) = n, current solution
- y::AbstractArray:     size(y) = (m,n), elipsoid centers
- Q::AbstractArray:     size(Q) = (m,n,n), elipsoid matrices
- k::AbstractVector:    size(k) = m, elipsoid radii  
- ϵ::Real:              precision parameter
"""

function min_violation(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector, ϵ::Real)
	m, n = size(y)
	min_violation = Inf
	for i in 1:m
		violation = (x - y[i, :])' * Q[i, :, :] * (x - y[i, :]) - (k[i] + ϵ)^2
		if violation < min_violation
			min_violation = violation
		end
	end
	return min_violation
end

function max_violation(x::AbstractVector, y::AbstractMatrix, Q::AbstractArray, k::AbstractVector, ϵ::Real)
	m, n = size(y)
	max_violation = -Inf
	for i in 1:m
		violation = (x - y[i, :])' * Q[i, :, :] * (x - y[i, :]) - (k[i] + ϵ)^2
		if violation > max_violation
			max_violation = violation
		end
	end
	return max_violation
end





"""
elipsoid_projection(x::AbstractVector, y::AbstractVector, Q::AbstractMatrix, k::Real, env::Gurobi.Env, ϵ::Real)

Computes the exact projection of x onto the elipsoid (y,Q,k)

- x::AbstractVector:    Current point
- y::AbstractVector:    Elipsoid center
- Q::AbstractMatrix:    Elipsoid Matrix
- k::Real:              Elipsoid radius
- env::Gurobi.Env:      Gurobi environment to solve projection
"""
function elipsoid_projection(x::AbstractVector, y::AbstractVector, Q::AbstractMatrix, k::Real, env::Gurobi.Env)
	if (x .- y)' * Q * (x .- y) <= k^2
		return x
	end
	n = size(x, 1)
	model = direct_model(Gurobi.Optimizer(env))
	set_silent(model)
	# set_optimizer_attribute(model, "NumericFocus", 3)
	# set_optimizer_attribute(model, "ScaleFlag", 2)
	set_optimizer_attribute(model, "ObjScale", k^2)

	@variable(model, y_var[1:n])

	@objective(model, Min, 0.5 * sum((y_var[j] - x[j])^2 for j in 1:n)) # 0.5 * ⟨y_var - x, y_var - x⟩
	@expression(model, diff[j=1:n], y_var[j] - y[j])
	@expression(model, quad_expr, dot(diff, Q * diff)) # ⟨y_var - y, Q*(y_var - y)⟩
	@constraint(model, quad_expr <= k^2)
	optimize!(model)

	status = termination_status(model)

	if status == MOI.INFEASIBLE
		println("model infeasible")
		write_to_file(model, "model_proj.lp")
		error("Exiting due to infeasibility.")
	elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
		set_optimizer_attribute(model, "DualReductions", 0)
		optimize!(model)
		status = termination_status(model)
		if status == MOI.INFEASIBLE
			println("model infeasible after DualReductions")
			write_to_file(model, "model_proj.lp")
			error("Exiting due to infeasibility.")
		end
	elseif status == MOI.NUMERICAL_ERROR
		println("Numerical ERROR")
		set_optimizer_attribute(model, "BarHomogeneous", 0)
		set_optimizer_attribute(model, "OutputFlag", 1)
		optimize!(model)
		status = termination_status(model)
		if status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED
			write_to_file(model, "model_proj.lp")
			error("No solution found even after reopt")
		end
	elseif status != MOI.OPTIMAL && status != MOI.LOCALLY_SOLVED
		println("Unknown status = $status")
		write_to_file(model, "model_proj.lp")
		error("Exiting due to unknown status.")
	end

	return value.(y_var)

end


