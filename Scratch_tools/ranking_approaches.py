import numpy as np

def fill_lower_diag(a):
    n = int(np.sqrt(len(a)*2))+1
    mask = np.tri(n,dtype=bool, k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((n,n),dtype=float)
    out[mask] = a
    return out

def get_triangled_list(l, rows, typ='lower'):
    if type(l) is not list:

        return None

    if type(rows) is not int:

        return None

    if not(typ == 'lower' or typ == 'upper'):

        return None

    new_l = []
    length = len(l)
    num_items = ((rows-1) * rows)/ 2

    if length != num_items:

        return None

    if typ == 'upper':
        for i in range(rows):
            temp_l = [0]*(i+1) + [l.pop(0) for j in range(7-(i+1))]
            new_l.append(temp_l)
    elif typ=='lower':
        for i in range(rows):
            temp_l = [l.pop(0) for j in range(i)] + [0]*(rows-i)
            new_l.append(temp_l)

    return new_l

def dia_matrix2(mat,n):
    out = np.zeros((n, n))
    inds = np.triu_indices(len(out),1)
    out[inds] = mat
    return out.transpose()


def calc_rank(p_vals, medians, n):
    #n=n-1
    alpha=0.05
    score = np.zeros((n,n))
    #p_vals_mat = fill_lower_diag(p_vals)
    p_vals_mat = dia_matrix2(p_vals,n)
    for i in range(n):
        for j in range(i+1,n):
            if p_vals_mat[j,i]<alpha and medians[i]>medians[j]:
                score[j,i]=1
            elif p_vals_mat[j,i]<alpha and medians[i]<medians[j]:
                score[j,i]=-1

        for j in range(0,i):
            if p_vals_mat[i,j]<alpha and medians[i]>medians[j]:
                score[j+1,i]=1
            elif p_vals_mat[i,j]<alpha and medians[i]<medians[j]:
                score[j+1,i]=-1

    print("P_vals")
    print(p_vals_mat)
    print("Medians:")
    print(medians)
    print(score)
    score = np.sum(score, axis=0).reshape(-1)
    print(score)
    return score
