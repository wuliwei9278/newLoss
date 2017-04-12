using GLM
using DataFrames
function generate_data(n, alpha, prop, shift)
	# prop, proportion of zero
	# alpha, out of one, proportion of shifted to left
	y = zeros(n)
	# to add intercept term, add x0
	x0 = ones(n)
	x1 = zeros(n)
	x2 = zeros(n)
	for i = 1:n
		if rand() > prop
			y[i] = 1
			# mixture of two normal distributions, the first one is N(-10, 1) 
			# second one is N(0, 1), alpha represents the probability to be the first one
			if rand() > alpha
				x1[i] = randn() * 0.1 + 1
			else 
				x1[i] = randn() - shift
			end
			x2[i] = randn() * 0.1 
		else
			y[i] = 0
			x1[i] = randn() 
			x2[i] = randn() 
		end
		#x2[i] = randn()
	end
	return y, x0, x1, x2
end



function objetive(y, x0, x1, x2, Beta, Gamma, lambda)
	sum1 = 0.
	sum2 = 0.
	for i = 1:length(y)
		score = Gamma * (x0[i] * Beta[1] + x1[i] * Beta[2] + x2[i] * Beta[3])
		sum1 += y[i] * exp(score)
		sum2 += exp(score)
	end
	return log(sum1) - log(sum2) - lambda / 2.0 * norm(Beta)^2
end

function comp_grad(y, x0, x1, x2, Beta, Gamma, lambda)
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
	return a / b - c / d - lambda * Beta
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
	#precision = precision ./ K
	return precision
end

#using PyPlot
#n = 5000; alpha = 0.01; prop = 0.5; shift = 5; lambda = 1; Gamma = 1;
#y_train, x0_train, x1_train, x2_train = generate_data(n, alpha, prop, shift)
#scatter(x1_train[y_train.==0], x2_train[y_train.==0], color="blue")
#scatter(x1_train[y_train.==1], x2_train[y_train.==1], color="red")


function main(n, alpha, prop, shift, lambda, Gamma)
	#n = 1000
	#alpha = 0.02
	#prop = 0.5
	#shift = 10
	#lambda = 1
	# generate training data
	y_train, x0_train, x1_train, x2_train = generate_data(n, alpha, prop, shift)
	# see how many outliers
	#println(length(x1_train[x1_train .<=-5]))
	# see how many one in total 
	#println(length(y_train[y_train.==1]))
	println(length(x1_train[x1_train .<= -shift + 3]), " ", length(y_train[y_train.==1]))

	# generate test data
	y_test, x0_test, x1_test, x2_test = generate_data(n, alpha, prop, shift)
	println(length(x1_test[x1_test .<= -shift + 3]), " ", length(y_test[y_test.==1]))

	# the model has three parameters, beta0, beta1, beta2
	Beta = 0.01*randn(3)
	#Gamma = 1.0
	step_size = 10
	for i = 1:800
		prev_obj = objetive(y_train, x0_train, x1_train, x2_train, Beta, Gamma, lambda)
		Beta_old = Beta
		grad = comp_grad(y_train, x0_train, x1_train, x2_train, Beta_old, Gamma, lambda)
		for iter = 1:20
			Beta = Beta_old + step_size * grad
			new_obj = objetive(y_train, x0_train, x1_train, x2_train, Beta, Gamma, lambda)
			if (new_obj > prev_obj)
				break
			else
				step_size /= 2.0
			end
		end
		
		
	end

	K = [1, 5, 10, 20, 50, 100]
	precision = compute_precision(K, Beta, Gamma, y_test, x0_test, x1_test, x2_test)
	#recall = K / length(y_test)
	println(Int.(precision))
	#println(recall)
	println(Beta)


	data = DataFrame(x1 = x1_train, x2 = x2_train, y = y_train);
	Logistic = glm(@formula(y ~ x1 + x2), data, Bernoulli(), LogitLink());
	newX = DataFrame(x1 = x1_test, x2 = x2_test);
	score_l = predict(Logistic, newX);
	p_l = sortperm(score_l, rev = true);
	precision_l = zeros(length(K));
	for c = 1:K[length(K)]
		i = p_l[c]
		if y_test[i] == 1.0
			for k in length(K):-1:1
				if c <= K[k]
					precision_l[k] += 1
				else
					break
				end
			end
		end
	end
	println(Int.(precision_l))
	#precision_l = precision_l ./ K;
	#println(Int.(precision_l .* K))
	println(coef(Logistic))
end





