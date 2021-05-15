from collections import Counter

import llr

def calculate_llr(X, y, label1, label2, n=25):
    x1 = [X[i] for i in range(0, len(X)) if y[i] == label1]
    x2 = [X[i] for i in range(0, len(X)) if y[i] == label2]
    x1_counter = Counter(' '.join(x1).split())
    x2_counter = Counter(' '.join(x2).split())

    cmp_results = llr.llr_compare(x1_counter, x2_counter)

    top_x1 = {k:v for k,v in sorted(cmp_results.items(), key=lambda x: (-x[1], x[0]))[:n]}
    top_x2 = {k:v for k,v in sorted(cmp_results.items(), key=lambda x: (x[1], x[0]))[:n]}

    return top_x1, top_x2