import numpy as np

H = np.array([
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0]
])

check_nbrs = { i: np.where(H[i,:] == 1)[0] for i in range(H.shape[0]) } # stores non-zero indexes of each ROW as a key,value pair
bit_nbrs = { j: np.where(H[:,j] == 1)[0] for j in range(H.shape[1]) } # stores non-zero indexes of each COLUMN as a key,value pair

print(check_nbrs)
print(bit_nbrs)