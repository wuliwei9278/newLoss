n = 400
y = zeros(n)
# to add intercept term, add x0
x0 = ones(n)
x1 = zeros(n)
x2 = zeros(n)
alpha = 0.02
for i = 1:n
	if rand() > 0.5
		y[i] = 1
		# mixture of two normal distributions, the first one is N(-10, 1) 
		# second one is N(0, 1), alpha represents the probability to be the first one
		if rand() > alpha
			x1[i] = randn()
		else 
			x1[i] = randn() - 10
		end
	else
		y[i] = 0
		x1[i] = randn()
	end
	x2[i] = randn()
end

function objetive(y, x0, x1, x2, Beta, Gamma)
	sum1 = 0.
	sum2 = 0.
	for i = 1:length(y)
		score = Gamma * (x0[i] * Beta[1] + x1[i] * Beta[2] + x2[i] * Beta[3])
		sum1 += y[i] * exp(score)
		sum2 += exp(score)
	end
	return log(sum1) - log(sum2)
end

function comp_grad(y, x0, x1, x2, Beta, Gamma)
	a = zeros(3)
	b = 0.
	c = zeros(3)
	d = 0.
	for i = 1:length(y)
		score = Gamma * (x0[i] * Beta[1] + x1[i] * Beta[2] + x2[i] * Beta[3])
		a += y[i] * exp(score) * [x0[i], x1[i], x2[i]]
		b += y[i] * exp(score)
		c += exp(score) * [x0[i], x1[i], x2[i]]
		d += exp(score)
	end
	return a / b - c / d
end

# the model has three parameters, beta0, beta1, beta2
Beta = randn(3)
Gamma = 1.0
step_size = 0.1
println(objetive(y, x0, x1, x2, Beta, Gamma))
for i = 1:3000
	grad = comp_grad(y, x0, x1, x2, Beta, Gamma)
	Beta += step_size * grad
	println(objetive(y, x0, x1, x2, Beta, Gamma))
end

