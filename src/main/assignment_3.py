import numpy as np

def gaussian_elimination(matrix):
    n = len(matrix)
    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1
        pivot = matrix[i][i]
        for j in range(i, n + 1):
            matrix[i][j] = matrix[i][j] / pivot
        
        # Eliminate column i from all rows below
        for k in range(i + 1, n):
            factor = matrix[k][i]
            for j in range(i, n + 1):
                matrix[k][j] = matrix[k][j] - factor * matrix[i][j]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = matrix[i][n]
        for j in range(i+1, n):
            x[i] = x[i] - matrix[i][j] * x[j]
    return x

def lu_factorization(matrix):
    n = len(matrix)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Create L and U matrices
    for i in range(n):
        # Upper triangular matrix
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum
            
        # Lower triangular matrix
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (matrix[k][i] - sum) / U[i][i]
                
    return L, U

def is_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal = abs(matrix[i][i])
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True

def is_positive_definite(matrix):
    # Check if matrix is symmetric
    if not np.allclose(matrix, matrix.T):
        return False
    
    # Check eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    return all(eigenval > 0 for eigenval in eigenvalues)

def main():
    # Question 1
    matrix1 = np.array([[2, -1, 1, 6],
                       [1, 3, 1, 0],
                       [-1, 5, 4, -3]], dtype=float)
    result1 = gaussian_elimination(matrix1.copy())
    print(result1)
    print()
    
    # Question 2
    matrix2 = np.array([[1, 1, 0, 3],
                       [2, 1, -1, 1],
                       [3, -1, -1, 2],
                       [-1, 2, 3, -1]], dtype=float)
    
    # Calculate determinant
    det = np.linalg.det(matrix2)
    print(f"{det}")
    
    # LU Factorization
    L, U = lu_factorization(matrix2)
    print(L)
    print(U)
    print()
    
    # Question 3
    matrix3 = np.array([[9, 0, 5, 2, 1],
                       [3, 9, 1, 2, 1],
                       [0, 1, 7, 2, 3],
                       [4, 2, 3, 12, 2],
                       [3, 2, 4, 0, 8]], dtype=float)
    print(is_diagonally_dominant(matrix3))
    print()
    
    # Question 4
    matrix4 = np.array([[2, 2, 1],
                       [2, 3, 0],
                       [1, 0, 2]], dtype=float)
    print(is_positive_definite(matrix4))

if __name__ == "__main__":
    main() 