import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
from log_parsing import parse_file


def preprocess_name(name):
    replace_dict={"Extractor": "", 
                  "Classifier": "",
                  "LinearDiscriminantAnalysis": "LDA"}
    for key, value in replace_dict.items():
        name=name.replace(key, value)
    return name


def plot_perf(input_path):
    with open(input_path) as input_file:
        lines=input_file.read().splitlines()
    results=parse_file(lines)

    pos=np.arange(len(results))
    
    names=[preprocess_name(res.experiment_name) for res in results]

    ax1 = plt.subplot(121)
    ax1.grid(axis="y")
    x1_values=[res.accuracy for res in results]
    ax1.barh(pos, x1_values, label="Accuracy")
    ax1.set_yticks(pos)
    ax1.set_yticklabels(names)    
    ax1.set_xlim([min(x1_values)*9/10, max(x1_values)])
    ax1.legend(loc='upper right')

    ax2=plt.subplot(122, sharey=ax1)
    ax2.grid(axis="y")
    ax2.barh(pos, [res.fit_time for res in results], 
             label="Fit time,\nsec")
    ax2.barh(pos, [res.score_time for res in results], color="red",
             label="Score time,\nsec")
    # ax2.set_yticks(pos)
    ax2.set_yticklabels(names, visible=False)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    #plt.show()
    plt.savefig("results.png")


def main():
    plot_perf("log_ALL_4.txt")


if __name__=="__main__":
    main()
