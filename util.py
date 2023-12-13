import numpy as np
from numpy.typing import NDArray

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
        

def generate_encoder(H: NDArray) -> tuple(NDArray, NDArray):
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


def _bit_to_check(col: NDArray, *args: dict) -> NDArray:
    '''
    Returns the column vector of messages passed from a bit to neighouring checks
    To be used with numpy.apply_along_axis
    '''
    pass

def _check_to_bit(row: NDArray, *args: dict) -> NDArray:
    '''
    Returns the row vector of messages passed from a check to neighouring bits
    To be used with numpy.apply_along_axis
    '''
    pass
    

def ldpc_decode(H:              NDArray,
                y:              NDArray,
                noise_ratio:    float,
                max_iter:       int = 20) -> tuple(int, NDArray):
    
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
        'NUM_ITER': 1
    }

    # TODO: init all data strutures to use in algorithm
    check_nbrs = {}
    bit_nbrs = {}
    z = None # store coding

    # TODO: init messages (v -> c) 

    # set up loop
    curr_step = 1
    while curr_step <= max_iter:
        DIAGNOSTIC_dict['NUM_ITER'] = curr_step

        # TODO: c -> v updates
        # TODO: v -> c updates


        # TODO: generate coding
        # break condition if we get the right code
        if (np.all(H @ z % 2 == 0)):
            DIAGNOSTIC_dict['SUCCESS_CODE'] = 0
            break
        
        curr_step += 1

    return DIAGNOSTIC_dict, None