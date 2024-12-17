import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import problem_types.QR as QR
import sketching_methods.sketching as sketching
import matplotlib.pyplot as plt
import utilities.helpers as helpers
import problem_types.distortion as distortion

def draw_relative_spread(sketching_functions_dict):
    data = {}
    for key in sketching_functions_dict:
        print(key)
        results = compute_relative_spread(sketching_functions_dict[key])
        data[key] = results
    # draw_violin_plot(data)
    draw_box_plot(data)
    

def compute_relative_spread(sketching_function, loops = 1000):
    results = np.zeros(loops)
    m = 256
    n = 20
    k = 50
    for i in range(loops):
        A = np.random.standard_normal((m, n))
        As = sketching_function(k, A)
        x = np.random.standard_normal(n)
        #Ax_norm2 = np.linalg.norm(A @ x)**2
        Asx_norm2 = np.linalg.norm(As @ x)
        
        results[i] = Asx_norm2
    results /= np.mean(results)
    return results

def draw_box_plot(data):

    names, vals, xs = [], [] ,[]

    for i, key in enumerate(data):
        names.append(f"{key}")
        vals.append(data[key])
        xs.append(np.random.normal(i+1, 0.04, len((data[key]))))

    plt.boxplot(vals, labels=names, showfliers=False)
    for x, val in zip(xs, vals):
        plt.scatter(x, val, alpha=0.4, s=3)
    
    # plt.rcParams['text.usetex'] = True
    plt.xticks(rotation=45)
    plt.xlabel('Sketching Methods')
    plt.ylabel('Relative norm')
    plt.title('Change of Relative Norms for Different Sketching Methods')

    helpers.save_plot("relative_spread_box_plot")
    plt.show()


def draw_violin_plot(data):

    names, vals, xs = [], [] ,[]

    for i, key in enumerate(data):
        names.append(f"{key}")
        vals.append(data[key])
        xs.append(np.random.normal(i+1, 0.04, len((data[key]))))

    #plt.boxplot(vals, labels=names, showfliers=False)
    #for x, val in zip(xs, vals):
    #    plt.scatter(x, val, alpha=0.4, s=3)
    
    plt.violinplot(vals, showmeans=False, showmedians=True)
    plt.xticks(range(1, len(names) + 1), names, rotation=90)
    plt.xlabel('Sketching Methods')
    plt.ylabel('Relative norm')
    plt.title('Box Plot of Relative Variance for Different Sketching Methods')

    helpers.save_plot("relative_spread_violin_plot")
    plt.show()

if __name__ == "__main__":
    draw_relative_spread(sketching.sketching_functions_dict)



