""" DI """

def update_perfs_di(perfs, learning_stats) :

    #auc, bacc, di = perfs
    auc, uif, cons = perfs

    learning_stats["auc"].append(auc)
    #learning_stats["bacc"].append(bacc)
    #learning_stats["di"].append(di)
    ###### uif
    learning_stats["uif"].append(uif)
    learning_stats["consistency"].append(cons) # consistency

    return learning_stats


def write_perfs_di(result_path, file_name, mode, lmda, tau, learning_stats) :
    lines = ['Mode: {}\n'.format(mode), 'Lamda: {}\n'.format(lmda), 'Tau: {}\n'.format(tau),
             #'AUC: {}\n'.format(learning_stats["auc"][-1]), 'DI: {}'.format(learning_stats["di"][-1])]
             'AUC: {}\n'.format(learning_stats["auc"][-1]), 'DI: {}'.format(learning_stats["uif"][-1]),
             'Consistency: {}\n'.format(learning_stats['consistency'][-1])]
    file = open(result_path + file_name + '.txt', "w")
    for line in lines:
        file.write(line)
        file.write('\n')
    file.close()
