import numpy as np

def create_convolution_matrix(input_shape, kernel, stride):
    input_height, input_width = input_shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    
    C_height = output_height * output_width
    C_width = input_height * input_width
    
    C = np.zeros((C_height, C_width)) # Create a zero matrix to store the convolution matrix
    
    idx = 0
    for i in range(0, output_height , stride):
        for j in range(0, output_width, stride):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    C[idx, (i+k)*input_width + (j+l)] = kernel[k, l]
            idx += 1
    
    return C

image = np.array([
    [3, 3, 2, 1],
    [0, 0, 1, 3],
    [3, 1, 2, 2],
    [2, 0, 0, 2]
])

kernel = np.array([
    [0, 1, 2],
    [2, 2, 0],
    [0, 1, 2]
])

stride = 1

I = image.reshape(-1, 1)

C = create_convolution_matrix(image.shape, kernel, stride)

result = np.dot(C, I) # Apply Matrix Convolution
result = result.reshape(image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1)


C_tranpose = C.T
result = result.reshape(4, 1)
res = np.dot(C_tranpose, result)
print(res.reshape(4, -1))