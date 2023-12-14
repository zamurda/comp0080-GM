import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def GF2_row_echelon(H: NDArray) -> NDArray:
    '''
    Returns the row echelon form of a binary matrix defined over GF(2).

    Follows a simple algorithm whereby pivots are found in each column, and used to 'knock out'
    1s in rows above or below using the XOR (addition modulo 2) operation.

    Pivots are found in each row recursively, so the function may break for matrices with >1000 rows.
    In this case, it is recommended to change the maximum recursion depth on your system.
    Params
    -----------------
    H:NDArray - the matrix to be put in REF

    Returns
    -----------------
    M:NDArray - the REF version of H
    '''

    # extract nrows and ncols
    m, n = H.shape

    # find leftmost nonzero column
    # this is our pivot column
    lm_nonzero = np.arange(n)[np.any(H, axis=0)].min()
    #print(lm_nonzero)

    # check if a 1 is in the topmost position
    if H[:,lm_nonzero][0] == 1:
        # check if there are other 1s
        # if there are, then add the first row to get rid
        for i, val in enumerate(H[:,lm_nonzero][1:]):
            if val == 1:
                #print(f'eliminating row {i}')
                H[i+1] = H[i+1] ^ H[0]

    else:
        # find the rows where there are 1s
        ones = np.where(H[:,lm_nonzero] == 1)[0]
        #print('swapping')
        H[[0, ones[0]]] = H[[ones[0], 0]] # swap with row containing a 1 to put it at the top

        # if there are other 1s, eliminate them
        if len(ones) > 1:
            #print('eliiminating')
            for i, val in enumerate(H[:,lm_nonzero][1:]):
                if val == 1:
                    H[i+1] = H[i+1] ^ H[0]

    # if there are no more nonzero rows below the pivot, we are finished
    # inner np.any() checks if rows are nonzero
    # outer np.any() is ONLY false if ALL rows are zero
    if not np.any(np.any(H[1:,:], axis=1)):
        return H
    else:
        # catch the reduced submatrix in our recursion
        #print('reducing')
        reduced_submat = GF2_row_echelon(H[1:,:])
        #print(reduced_submat)
        return np.vstack((H[0], reduced_submat))

          

def GF2_rref(H: NDArray) -> NDArray:
    '''
    Computes the row-reduced echelon form of a given binary matrix
    over the finite field GF(2)

    Params
    -----------------
    H:NDArray - the matrix to be row-reduces

    Returns
    ----------------
    M:NDArray - the RREF form of H
    '''

    H = GF2_row_echelon(H)

    # find all columns containing a leading 1
    leading_ones = np.array([np.where(row == 1)[0].min() for row in H \
                             if np.any(np.where(row == 1)[0])]) # Doesn't break for a zero row

    # starting from the rightmost column, eliminate all 1s above it
    for i in leading_ones[::-1]:
        ones = np.where(H[:,i] == 1)[0]

        if len(ones) > 1:
            for j in ones[:-1]: # the last value should be the 1 that we want to keep since we are only considering REF(H)
                H[j] = H[j] ^ H[ones[-1]]
        
        else:
            continue
    
    return H
        

def generate_encoder(H: NDArray) -> Tuple[NDArray, NDArray]:
    '''
    builds a systematic encoding matric given a parity check matrix H,
    by computing the echelon form, and permuting until the standard form is acheived

    the standard form of H is [P|I(n-k)] where n-k is the number of rows in H
    the systematic encoding metrix is thus defined as [I(k)|P].T
    
    parameters
    --------------
    H - the parity check matrix

    returns
    --------------
    H' - An in-place column-permuted version of H (the echelon form)
    G - the systematic encoder
    '''

    H = GF2_rref(H)
    
    n = H.shape[1]
    k = n - H.shape[0]
    
    # we need to permute the columns until we get the required form of H
    # do this by searching the columns of H for vectors [0..1..0].T which have 1s in the required positions
    for i in range(n-k):
        if np.array_equal(H[:, -(n-k):], np.eye(n-k)):
            break
        target = np.zeros(shape=(n-k,))
        target[i] = 1
        for j,vec in enumerate(H.T):
            if np.array_equal(vec, target):
               H[:, [k+i,j]] = H[:, [j,k+i]]

    assert(np.array_equal(H[:,-(n-k):], np.eye(n-k))) #making sure the last n-k cols form the identity

    P = H[:,:n-k]
    G = np.concatenate((np.eye(k), P), axis=0)
      
    return H, G


def _bit_to_check(col: NDArray, nbrs: NDArray) -> NDArray:
    '''
    Given a column of edges between a bit and neighbouring checks,
    returns the column vector of messages passed from the bit.
    
    For each edge (nonzero element in the column) the msg is the
    sum of all other nonzero elements apart from that one (- a constant)
    '''
    col[nbrs] = np.sum(col[nbrs]) - col[nbrs]

    return col

def _check_to_bit(row: NDArray, nbrs: NDArray) -> NDArray:
    '''
    Returns the row vector of messages passed from a check to neighouring bits

    '''
    tanh_prod = np.prod(np.tanh(0.5 * row[nbrs])) / np.tanh(0.5 * row[nbrs], dtype=np.float64)
    #print(tanh_prod)
    row[nbrs] = np.log(1 + tanh_prod) -  np.log(1 - tanh_prod)

    return row
    

def ldpc_decode(H:              NDArray,
                y:              NDArray,
                noise_ratio:    float,
                max_iter:       int = 20) -> Tuple[dict, NDArray]:
    
    '''
    LDPC decoder which uses the Loopy Belief Propagation (LBP) algorithm on a graph defined by the parity check matrix H
    to decode a recieved string of coded bits through a noisy channel with noise ratio p (binary {0,1} random variables)

    Params
    ----------------------
    H: NDArray          - parity check matrix in canonical form
    y: NDArray          - recieved string of bits
    noise_ratio: float  - probability of a recieved bit being 1
    max_iter: int       - the max number of steps LBP can run for


    Returns
    ---------------------
    DIAGNOSTIC_INFO:dict   - {'SUCCESS_CODE': success code (0 for successful convergence, -1 otherwise),
                                'NUM_ITER': current iteration}
    x:NDarray           - the decoded string on the last iteration
    '''

    # init info dict
    DIAGNOSTIC_dict = {
        'SUCCESS_CODE': -1,
        'NUM_ITER': 0
    }

    # init all data strutures to use in algorithm
    check_nbrs = { i: np.where(H[i,:] != 0)[0] for i in range(H.shape[0]) } # stores non-zero indexes of each ROW as a key,value pair
    bit_nbrs = { j: np.where(H[:,j] != 0)[0] for j in range(H.shape[1]) } # stores non-zero indexes of each COLUMN as a key,value pair
    z = y # store coding

    # Deep copy p-check matrix to avoid editing in-place
    M = np.copy(H).astype(np.float64)

    # init messages (v -> c)
    # if node is 0
    bits_0 = np.where(y == 0)[0]
    for i in bits_0:
        M[:,i][bit_nbrs[i]] = np.log(1-noise_ratio) - np.log(noise_ratio)

    # if node is 1
    bits_1 = np.where(y == 1)[0]
    for j in bits_1:
        M[:,j][bit_nbrs[j]] = np.log(noise_ratio) - np.log(1-noise_ratio)
    
    init_messages = np.copy(M).astype(np.float64) # need to add initial message to v -> c messages

    # DONT NEED ANYMORE
    # Add a layer of indices to the messages matrix M so that apply_along axes knows exactly which col or row it is in
    #M = np.c_[np.arange(M.shape[0]), M]
    #M = np.r_[np.arange(M.shape[1])[np.newaxis,:] - 1, M] # -1 offset is so that the first entry of every col is the TRUE COL IDX

    # set up loop
    curr_step = 0
    while curr_step <= max_iter:

        # check -> bit updates (rows), 
        for i in range(M.shape[0]):
            M[i,:] = _check_to_bit(M[i,:], check_nbrs[i])
        
        curr_step += 1
        DIAGNOSTIC_dict['NUM_ITER'] = curr_step
        
        # generate coding
        z = np.where(np.sum(M, axis=0) < 0, 1, 0)

        # break condition if we get the right code
        if np.all((H @ z) % 2 == 0):
            DIAGNOSTIC_dict['SUCCESS_CODE'] = 0
            break

        

        # bit -> check updates (columns), 
        for i in range(M.shape[1]):
            M[:,i] = _bit_to_check(M[:,i], bit_nbrs[i]) + init_messages[:,i]


    return DIAGNOSTIC_dict, z
