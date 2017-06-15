import re
import pickle
import numpy as np 
from collections import namedtuple

entry_len=11


tranlsations={
    "pullups": "Подтягивания",
    "pushups": "Отжимания", 
    "sits": "Приседания",
    "walk": "Ходьба"
}

estimators_translations={
    "FFTCoeffsExtractor": "Выделение коэффициентов быстрого преобразования Фурье",
    "STFTCoeffsExtractor": "Выделение коэффициентов оконного преобразования Фурье",
    "HMMABOutExtractor": "Выделение параметров скрытой марковской модели",
    "HMMOutCovarsExtractor": "Выделение параметров распределений скрытой марковской модели, описывающих наблюдаемые состояния",
    "RawExtractor": "Использование значений ряда как признаков",
    "SignalInterpolator": "Использование коэффициентов аппроксимирующих сплайнов как признаков",
    "MultiARFeatureExtractor": "Вычисление параметров модели временного ряда",
    "SpectrumInterpolator": "Использование коэффициентов сплайнов, аппроксимирующих спектр, как признаков",
    "WaveletsFeaturesExtractor": "Выделение коэффициентов дискретного-вейвлет преобразования",
    "DTWTransformer": "Применение алгоритма динамического преобразования времени для определения расстояния между рядами",
    "KNeighborsClassifier": "применение метода k ближайших соседей",    
    "MLPClassifier": "применение нейронной сети прямого распространения",
    "GaussianNB": "применение наивного байесовского классификатора",
    "LinearDiscriminantAnalysis": "применение линейного дискриминантного анализа"
}


def parse_file(all_lines):
    lines=list(filter(None, all_lines))
    grouped = [lines[k:k+entry_len] for k in range(0, len(lines), entry_len)]
    parsed_results=list(map(parse, grouped))
    return parsed_results


def parse(entry_lines):
    lines=list(filter(None, entry_lines))
    if len(lines)!=entry_len:
        raise ValueError("Expected %d lines, got %d" % (entry_lines,
                                                        len(lines)))
    name=parse_name(lines[0])
    classes=parse_classes(lines[1])
    confmat=parse_confmat(lines[2:6], len(classes))
    accuracy=parse_score(lines[6])
    f1=parse_score(lines[7])
    fit_time=parse_time(lines[9])
    score_time=parse_time(lines[10])
    return PrintData(name, classes, confmat, accuracy, f1, fit_time, score_time)


def to_latex(parsed, prefix, _):
    table=get_content(parsed)
    # full_latex=wrap(,
    #                 "table", r"[\tableopts]")
    tabular=wrap(table, "tabular", r"{\tableformat}")
    caption=get_caption(parsed, prefix)
    full=wrap("\n".join([tabular, caption]), "table", r"[\tableopts]")
    return full


def get_latex_table(parsed):
    names=[tranlsations[eng] for eng in parsed.classes]
    to_line=lambda content: content+r" \\ \hline"
    header=r" \hline"+to_line(" & ".join([r"{}", ]+names))
    data=""
    def get_line(input):
        num, cls=input
        values=list(map(str, parsed.confmat[num, :]))
        line=to_line(" & ".join([cls,]+values))
        return line
    content="\n".join(map(get_line, enumerate(names)))
    return header+"\n"+content


def get_stats(num, name, value, postfix):
    tpl=r"\multicolumn{%d}{|c|}{%s: %f%s} \\ \hline"
    return tpl % (num, name, value, postfix)


def get_content(parsed):
    data=get_latex_table(parsed)
    stat1=get_stats(len(parsed.classes)+1, "Точность", parsed.accuracy, "")
    stat2=get_stats(len(parsed.classes)+1, "Значение F-метрики",
                    parsed.f1_score, "")
    stat3=get_stats(len(parsed.classes)+1, "Время обучения",
                    parsed.fit_time, " с")
    stat4=get_stats(len(parsed.classes)+1, "Время классификации",
                    parsed.score_time, " с")
    stats="\n".join([stat1, stat2, stat3, stat4])
    return data+"\n"+stats
    
  # {} & Подтягивания & Отжимания  & Приседания & Ходьба \\ \hline
  # Подтягивания & 21 & 7 & 1 & 3 \\ \hline
  # Отжимания & 6 & 20 & 2 & 1 \\ \hline
  # Приседания & 3 & 2 & 35 & 2 \\ \hline
  # Ходьба & 3 & 6 & 1 & 49 \\ \hline
  # \multicolumn{5}{c}{Точность: 0.771605} \\ \hline
  # \multicolumn{5}{c}{Значение F-метрики: 0.748750} \\ \hline
  # \multicolumn{5}{c}{Время обучения: 2.024183 с} \\ \hline
  # \multicolumn{5}{c}{Время классификации: 0.110922 с} \\ \hline



def wrap(text, env, params_str):
    first_line=r"\begin{%s}%s" % (env, params_str)
    last_line=r"\end{%s}" % (env)
    return "\n".join([first_line, text, last_line])


def get_caption(parsed, prefix):
    extractor, classifier=parsed.experiment_name.split("_")
    names=estimators_translations[extractor]+", "+estimators_translations[classifier]
    return r"\caption{\label{table:%s_%s} %s}" % (prefix, 
                                                  parsed.experiment_name,
                                                  names)


def parse_name(line):
    return "_".join(line.split("Experiment: ")[-1].split(" -> "))


def parse_classes(header):
    classes_array=re.search(r"\[(?P<classes>.*?)\]", header).group("classes")
    classes=[quoted.strip('\'') for quoted in
             classes_array.strip("[]").split(" ")]
    return classes


def parse_confmat(confmat_lines, num_classes):
    confmat_joined="".join(confmat_lines).replace("[", " ").replace("]", " ")
    confmat_items=np.fromiter(map(int, confmat_joined.strip(" ").split()), int)
    confmat=np.reshape(confmat_items, (num_classes, num_classes))
    return confmat


def parse_score(line):
    return float(line.split(": ")[-1])


def parse_time(line):
    return float(line.split(":")[-1].replace(" seconds", ""))
    
    
PrintData=namedtuple("PrintData", ["experiment_name",
                                   "classes", "confmat",
                                   "accuracy", "f1_score",
                                   "fit_time", "score_time"])
