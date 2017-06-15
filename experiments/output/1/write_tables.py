import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
from log_parsing import parse_file




def main():
    with open("log_ALL_4.txt") as input_file:
        raw_data=input_file.read().splitlines()
    latex_data=parse_file(raw_data)
    with open("all_experiments.tex", "w") as output_file:
        output_file.write(latex_data)


if __name__=="__main__":
    main()
