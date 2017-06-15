import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
from log_parsing import parse_file, to_latex




def main():
    with open("log_ALL_4.txt") as input_file:
        raw_data=input_file.read().splitlines()
    results=parse_file(raw_data)
    latex_data=[to_latex(result, "full", "foo")
                for result in results]
    chunk_size=10
    chunks=[latex_data[i:i+chunk_size] for i in range(0, len(latex_data),
                                                      chunk_size)]
    for index, chunk in enumerate(chunks):
        with open("all_experiments_%d.tex" % index, "w") as output_file:
            output_file.write("\n\n".join(chunk))


if __name__=="__main__":
    main()
