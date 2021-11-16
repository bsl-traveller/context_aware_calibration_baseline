import numpy as np
from constants import *

context_dic = {"sample": 0, "blur": 1, "detail": 2, "edge_enhance": 3, "smooth": 4, "sharp": 5}
def context_labels(labels, context, label_smooth=0):
    context_labels = []

    for label in labels:
        c = np.zeros(CONTEXTS)
        c[context_dic[context]] = 1

        l = np.full(NUM_CLASSES, label_smooth / (NUM_CLASSES-1))
        l[label] = 1-label_smooth

        context_labels.append(np.concatenate([l,c]))

    return np.array(context_labels)