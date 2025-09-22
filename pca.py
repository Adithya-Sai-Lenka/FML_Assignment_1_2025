import numpy as np
import matplotlib.pyplot as plt
import os

data = np.genfromtxt('dataset1.csv', delimiter=',')

means = np.mean(data, axis=0)


# PCA
# Covariance Matrix
cov_matrix = np.zeros((data.shape[1], data.shape[1]))
for i in range(data.shape[0]):
    cov_matrix += np.outer(data[i] - means, data[i] - means)
cov_matrix /= data.shape[0]

eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)

print(np.max(eigen_values)*100/(np.sum(eigen_values)), "% of variance explained by the first principal component.", sep='')
print(np.min(eigen_values)*100/(np.sum(eigen_values)), "% of variance explained by the second principal component.", sep='')

os.makedirs('plots/pca_images', exist_ok=True)

plt.figure(figsize=(8, 8))
plt.scatter(np.matmul(data-means, eigen_vectors[:,-1]), np.matmul(data-means, eigen_vectors[:,-2]), alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SScatter Plot of Dataset along the Eigen Directions')
plt.grid(True)
plt.savefig('plots/pca_images/linear_kernel.png')
plt.close()

# Kernel PCA

# Quadratic Kernel
kernel_matrix = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(i, data.shape[0]):
        kernel_matrix[i, j] = (np.dot(data[i], data[j]) + 1) ** 2
        kernel_matrix[j, i] = kernel_matrix[i, j]

centered_kernel_matrix = kernel_matrix - np.mean(kernel_matrix, axis=0) - np.mean(kernel_matrix, axis=1)[:, np.newaxis] + np.mean(kernel_matrix)

eigen_values, eigen_vectors = np.linalg.eigh(centered_kernel_matrix)

alpha_1 = eigen_vectors[:,-1]/np.sqrt(data.shape[0]*eigen_values[-1])
alpha_2 = eigen_vectors[:,-2]/np.sqrt(data.shape[0]*eigen_values[-2])

x = np.dot(centered_kernel_matrix, alpha_1)
y = np.dot(centered_kernel_matrix, alpha_2)
plt.figure(figsize=(8, 8))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Degree 2 Polynomial Kernel PCA')
plt.grid(True)
plt.savefig('plots/pca_images/quadratic_kernel.png')
plt.close()

# Cubic Kernel
kernel_matrix = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(i, data.shape[0]):
        kernel_matrix[i, j] = (np.dot(data[i], data[j]) + 1) ** 3
        kernel_matrix[j, i] = kernel_matrix[i, j]

centered_kernel_matrix = kernel_matrix - np.mean(kernel_matrix, axis=0) - np.mean(kernel_matrix, axis=1)[:, np.newaxis] + np.mean(kernel_matrix)

eigen_values, eigen_vectors = np.linalg.eigh(centered_kernel_matrix)

alpha_1 = eigen_vectors[:,-1]/np.sqrt(data.shape[0]*eigen_values[-1])
alpha_2 = eigen_vectors[:,-2]/np.sqrt(data.shape[0]*eigen_values[-2])

x = np.dot(centered_kernel_matrix, alpha_1)
y = np.dot(centered_kernel_matrix, alpha_2)
plt.figure(figsize=(8, 8))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Degree 3 Polynomial Kernel PCA')
plt.grid(True)
plt.savefig('plots/pca_images/cubic_kernel.png')
plt.close()

# RBF Kernel

sigmas = [1, .5, 2, 4, 10]

for sigma in sigmas:
    kernel_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            dist = np.sum((data[i] - data[j])**2)
            kernel_matrix[i, j] = np.exp(-dist/(2*sigma**2))
            kernel_matrix[j, i] = kernel_matrix[i, j]
    centered_kernel_matrix = kernel_matrix - np.mean(kernel_matrix, axis=0) - np.mean(kernel_matrix, axis=1)[:, np.newaxis] + np.mean(kernel_matrix)
    # Calculate eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(centered_kernel_matrix)

    eigen_values[eigen_values<1e-12]=0

    alpha_1 = eigen_vectors[:,-1]/np.sqrt(data.shape[0]*eigen_values[-1])
    alpha_2 = eigen_vectors[:,-2]/np.sqrt(data.shape[0]*eigen_values[-2])
    x = np.dot(centered_kernel_matrix, alpha_1)
    y = np.dot(centered_kernel_matrix, alpha_2)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Radial Basis Function Kernel PCA (σ={sigma})')
    plt.grid(True)
    plt.savefig(f'plots/pca_images/rbf_kernel_sigma_{sigma}.png')
    plt.close()

# Part C
plt.figure(figsize=(8, 8))
plt.scatter(data[:500, 0], data[:500, 1], alpha=0.5, c='r')
plt.scatter(data[500:, 0], data[500:, 1], alpha=0.5, c='b')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Dataset with separate spirals')
plt.grid(True)
plt.savefig('plots/pca_images/dataset_separated.png')
plt.close()

sigmas = [0.2, 0.1, 100]
for sigma in sigmas:
    kernel_matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            dist = np.sum((data[i] - data[j])**2)
            kernel_matrix[i, j] = np.exp(-dist/(2*sigma**2))
            kernel_matrix[j, i] = kernel_matrix[i, j]
    centered_kernel_matrix = kernel_matrix - np.mean(kernel_matrix, axis=0) - np.mean(kernel_matrix, axis=1)[:, np.newaxis] + np.mean(kernel_matrix)
    # Calculate eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(centered_kernel_matrix)

    eigen_values[eigen_values<1e-12]=0

    alpha_1 = eigen_vectors[:,-1]/np.sqrt(data.shape[0]*eigen_values[-1])
    alpha_2 = eigen_vectors[:,-2]/np.sqrt(data.shape[0]*eigen_values[-2])
    x = np.dot(centered_kernel_matrix, alpha_1)
    y = np.dot(centered_kernel_matrix, alpha_2)
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:500], y[:500], alpha=0.5, c='r')
    plt.scatter(x[500:], y[500:], alpha=0.5, c='b')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Radial Basis Function Kernel PCA with separate spirals (σ={sigma})')
    plt.grid(True)
    plt.savefig(f'plots/pca_images/rbf_kernel_sigma_{sigma}_separated.png')
    plt.close()

