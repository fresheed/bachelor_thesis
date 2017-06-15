import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
from log_parsing import parse_file
import os


def preprocess_name(name):
    replace_dict={"Extractor": "", 
                  "Classifier": "",
                  "LinearDiscriminantAnalysis": "LDA"}
    for key, value in replace_dict.items():
        name=name.replace(key, value)
    return name


def get_perf(input_path):
    print("trying", input_path)
    with open(input_path) as input_file:
        lines=input_file.read().splitlines()
    results=parse_file(lines)
    return results

def compute_combined(perfs):
    for index in range(len(perfs)):
        perfs[index]=sorted(perfs[index],
                            key=lambda arg: arg.experiment_name)
    names=[preprocess_name(res.experiment_name) for res in perfs[0]]

    group_values=list(zip(*[[item.accuracy for item in perf] 
                            for perf in perfs]))
    min_values=[min(group) for group in group_values]
    max_values=[max(group) for group in group_values]
    # pos=np.arange(len(group_values))
    # ax1.barh(pos, max_values, label="Accuracy",  color="red", edgecolor=None)
    # ax1.barh(pos, min_values, label="Accuracy",  color="blue", edgecolor=None)
    # ax1.set_yticks(pos)
    # ax1.set_yticklabels(names)    
    # ax1.legend(loc='upper right')
    # plt.show()
    pairs=zip(min_values, max_values)
    return names, pairs

def plot_pairs(names, *entries):
    plt.gcf().set_size_inches(6, 10, forward=True)

    ax1 = plt.subplot(111)
    #ax1.grid(axis="y")
    ax1.grid(axis="x")

    pos=np.arange(len(names))
    plot_width=1/5
    dark_value=0.5
    light_value=0.8
    max_colors = [(0, dark_value, 0), (0, dark_value, dark_value,),
                  (dark_value, dark_value, dark_value), "black" ]
    min_colors = [(0, light_value, 0), (0, light_value, light_value,),
                  (light_value, light_value, light_value), "black" ]
    most_min=1
    most_max=0
    for index, entry in enumerate(entries):
        mins, maxs=zip(*entry)
        most_min=min(most_min, min(mins))
        most_max=max(most_max, max(maxs))
        ax1.barh(pos+index*plot_width*1.1, maxs, 
                 label="Max acc\n (%d ppl.)" % (index+1),
                 color=max_colors[index], edgecolor=None,
                 height=plot_width)
        ax1.barh(pos+index*plot_width*1.1, mins, 
                 label="Min acc\n (%d ppl.)" % (index+1),
                 color=min_colors[index], edgecolor=None,
                 height=plot_width)
        ax1.legend(loc='upper left')
    
    ax1.set_yticks(pos)
    ax1.set_yticklabels(names)    
    ax1.set_xlim([most_min*9/10, most_max+0.05])
    plt.tight_layout()
    #plt.show()
    plt.savefig("progression.png", bbox_inches='tight')
    

def main():    
    perfs_1=[get_perf("1/"+path)
             for path in os.listdir("1")
             if "max" not in path and path.endswith(".txt")]
    names, pairs_1=compute_combined(perfs_1)
    perfs_2=[get_perf("2/"+path) for path in os.listdir("2")
             if path.endswith(".txt")]
    _, pairs_2=compute_combined(perfs_2)
    perfs_3=[get_perf("3/"+path) for path in os.listdir("3")
             if path.endswith(".txt")]
    _, pairs_3=compute_combined(perfs_3)
    perfs_4=[get_perf("4/"+path) for path in os.listdir("4")
             if path.endswith(".txt")]
    _, pairs_4=compute_combined(perfs_4)
    plot_pairs(names, pairs_1, pairs_2, pairs_3, pairs_4)

    #compute_combined(perfs_1)

if __name__=="__main__":
    main()
