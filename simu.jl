function generate_data(n, alpha)
	y = zeros(n)
	# to add intercept term, add x0
	x0 = ones(n)
	x1 = zeros(n)
	x2 = zeros(n)
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
	return y, x0, x1, x2
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

function compute_precision(K, Beta, Gamma, y_test, x0_test, x1_test, x2_test)
	precision = zeros(length(K))
	score = zeros(length(y_test))
	for i = 1:length(y_test)
		score[i] = Gamma * (x0_test[i] * Beta[1] + x1_test[i] * Beta[2] + x2_test[i] * Beta[3])
	end	
	p = sortperm(score, rev = true)
	for c = 1:K[length(K)]
		i = p[c]
		if y_test[i] == 1.0
			for k in length(K):-1:1
				if c <= K[k]
					precision[k] += 1
				else
					break
				end
			end
		end
	end
	precision = precision ./ K
	return precision
end

function main()
	n = 1000
	alpha = 0.02
	# generate training data
	y_train, x0_train, x1_train, x2_train = generate_data(n, alpha)
	# see how many outliers
	println(length(x1_train[x1_train .<=-5]))
	# see how many one in total 
	println(length(y_train[y_train.==1]))

	# generate test data
	y_test, x0_test, x1_test, x2_test = generate_data(n, alpha)
	println(length(x1_test[x1_test .<=-5]), " ", length(y_test[y_test.==1]))

	# the model has three parameters, beta0, beta1, beta2
	Beta = randn(3)
	Gamma = 1.0
	step_size = 10
	#println(objetive(y_train, x0_train, x1_train, x2_train, Beta, Gamma))
	for i = 1:800
		prev_obj = objetive(y_train, x0_train, x1_train, x2_train, Beta, Gamma)
		Beta_old = Beta
		grad = comp_grad(y_train, x0_train, x1_train, x2_train, Beta_old, Gamma)
		for iter = 1:20
			Beta = Beta_old + step_size * grad
			new_obj = objetive(y_train, x0_train, x1_train, x2_train, Beta, Gamma)
			println(i, " Line Search iter ", iter, " Prev Obj ", prev_obj, " New Obj ", new_obj)
			if (new_obj > prev_obj)
				break
			else
				step_size /= 2.0
			end
		end
		
		#grad = comp_grad(y_train, x0_train, x1_train, x2_train, Beta, Gamma)
		#Beta += step_size * grad
		#println(objetive(y_train, x0_train, x1_train, x2_train, Beta, Gamma))
		
	end

	K = [1, 5, 10, 20, 50, 100]
	precision = compute_precision(K, Beta, Gamma, y_test, x0_test, x1_test, x2_test)
	recall = K / length(y_test)
	println(precision)
	println(recall)
end

# For one particular simulation
# training 8, 491
# test 11, 477
# 300 loops
# [1.0,1.0,0.9,0.7,0.54,0.55]

# 500 loops
#[1.0,1.0,1.0,0.8,0.56,0.54]
#[1.0,1.0,1.0,0.8,0.64,0.54]
#[1.0,1.0,1.0,0.8,0.62,0.55]
#[1.0,0.6,0.5,0.35,0.44,0.5] ### maybe a local optimal, happen ~1 out of eight 
# where Beta = [-0.490045, 2.44711, -2.13094]

# 800 loops
#[1.0,1.0,1.0,0.8,0.56,0.53]
#[1.0,1.0,1.0,0.85,0.54,0.53]
#[1.0,1.0,1.0,0.8,0.58,0.54]
# where Beta = [0.0536968, -2.19919, 1.09057]


