from sampler import ProbabilityModel, UnivariateNormal, MultiVariateNormal, Categorical, MixtureModel
import numpy as np
import matplotlib.pyplot as plt

# (1) Categorical distribution
ap = np.array([0.2,0.4,0.3,0.1])
CateDist = Categorical(ap)
cate_samples = CateDist.sample_many(1000)
categories, counts = np.unique(cate_samples, return_counts=True)
plt.bar([str(i) for i in categories], counts)
plt.title("Categorical Sampling Histogram")
plt.xlabel("Category")
plt.ylabel("Counts")
# plt.savefig("Fig0_1.pdf")
plt.show()

# (2) Univariate Normal Distribution
mu = 10
sigma = 1
UniNormal = UnivariateNormal(mu, sigma)
uninormal_samples = UniNormal.sample_many(8000)
plt.hist(uninormal_samples)
plt.title("Univariate Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Counts")
#plt.savefig("Fig0_2.pdf")
plt.show()


# (3) Multivarite Normal Distribution
Mu = np.array([1, 1])
Sigma = np.array([1, 0.5, 0.5, 1]).reshape((2,2))
MultiNormal = MultiVariateNormal(Mu, Sigma)
multinormal_samples = MultiNormal.sample_many(5000)
x = multinormal_samples[:,0]
y = multinormal_samples[:,1]
plt.scatter(x, y)
plt.title("2D Multivariate Normal Distribution")
plt.xlabel("X Value")
plt.ylabel("Y Value")
# plt.savefig("Fig0_3.pdf")
plt.show()


# (4) General Mixture Distribution
ap = np.array([0.25,0.25,0.25,0.25])
Mu = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
# covariance matrix
Sigma = np.identity(2)
pm = (MultiVariateNormal(Mu[0,:], Sigma), MultiVariateNormal(Mu[1,:], Sigma), 
		MultiVariateNormal(Mu[2,:], Sigma), MultiVariateNormal(Mu[3,:], Sigma))
MixDist = MixtureModel(ap, pm)
mix_samples = MixDist.sample_many(10000)
x = mix_samples[:,0]
y = mix_samples[:,1]
plt.scatter(x, y)
plt.title("Mixture of Four 2D Gaussian Distribution")
plt.xlabel("X Value")
plt.ylabel("Y Value")
plt.savefig("Fig0_4.pdf")
plt.show()

# estimate the percentage of data points within the circle centered at (0.1, 0.2)
def calculate_distance(data_point):
	#compute the Euclidean distance from a point to the center (0.1,0.2)
	return np.sqrt((data_point[0] - 0.1)**2 + (data_point[1] - 0.2)**2)
#calculate distance from the circle center for each data point (each row)
dist_array = np.apply_along_axis(calculate_distance, 1, mix_samples)
#percentage (probability) a data point is within the range of the unit circle
probability = sum(dist_array <= 1)/float(len(dist_array))
print probability


