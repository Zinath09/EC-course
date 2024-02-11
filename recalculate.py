### recalculation
def git rec_inter_node(i_index, j_index, old_lista, dist_matrix):
    
    n = len(old_lista)


    _ = min([i_index, j_index])
    j_index = max([i_index, j_index])
    i_index = _

    if (j_index+1)%n == i_index:
        # print("(j_index+1)%n == i")
        return rec_edge(i_index, j_index -1 ,old_lista, dist_matrix)
    if j_index - i_index == 1:
        # print(" j_index - i_index == 1")
        return rec_edge(i_index-1, j_index, old_lista, dist_matrix)
    i = old_lista[i_index]
    j = old_lista[j_index]

    delta = 0
    delta -= dist_matrix[old_lista[i_index-1]][i] # zamiast: i-1 -> i
    delta += dist_matrix[old_lista[i_index-1]][j] # jest:    i-1 ->j 

    delta -= dist_matrix[old_lista[j_index-1]][j] # zamiast j-1 -> j
    delta += dist_matrix[old_lista[j_index-1]][i] # jest    j-1 -> i

    delta -= dist_matrix[i][old_lista[(i_index+1)%n]] # zamiast i->i+1
    delta += dist_matrix[i][old_lista[(j_index+1)%n]]# jest    i->j+1

    delta -= dist_matrix[j][old_lista[(j_index+1)%n]] # zamiast j->j+1
    delta += dist_matrix[j][old_lista[(i_index+1)%n]] # jest j -> i+1
    return delta


def rec_edge(i_index, j_index, old_lista, dist_matrix):
    n = len(old_lista)
    
    i = old_lista[i_index]
    j = old_lista[j_index]
    delta = 0

    delta -= dist_matrix[i][old_lista[(i_index+1)%n]] # zamiast i->i+1
    delta += dist_matrix[i][j]# 

    delta -= dist_matrix[j][old_lista[(j_index+1)%n]] # zamiast j->j+1
    delta += dist_matrix[old_lista[(i_index+1)%n]][old_lista[(j_index+1)%n]] 
    return delta

def rec_intra_node(internal_index, external_node, old_lista, dist_matrix, cost_list):
    n = len(old_lista)
    delta = 0
    przed_i = old_lista[(internal_index+1)%n]
    po_i = old_lista[(internal_index-1)%n]
    i = old_lista[internal_index]
    delta -= dist_matrix[i][przed_i]
    delta -= dist_matrix[i][po_i]
    delta -= cost_list[i]

    delta += dist_matrix[external_node][po_i]
    delta += dist_matrix[external_node][przed_i]
    delta += cost_list[external_node]
    return delta