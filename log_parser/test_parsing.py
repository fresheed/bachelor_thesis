import unittest
import log_parsing
import numpy as np



log_example="""
11: Experiment: FFTCoeffsExtractor -> MLPClassifier
Confusion for ['pullups' 'pushups' 'sits' 'walk']:
[[21  7  1  3]
 [ 6 20  2  1]
 [ 3  2 35  2]
 [ 3  6  1 49]]
Accuracy: 0.771605
F1 score: 0.748750
Best params: {"foo": "bar"}
Fit time: 2.024183 seconds
Score time: 0.110922 seconds
"""

output_example=r"""\begin{table}[\tableopts]
\begin{tabular}{\tableformat}
{} & Подтягивания & Отжимания & Приседания & Ходьба \\ \hline
Подтягивания & 21 & 7 & 1 & 3 \\ \hline
Отжимания & 6 & 20 & 2 & 1 \\ \hline
Приседания & 3 & 2 & 35 & 2 \\ \hline
Ходьба & 3 & 6 & 1 & 49 \\ \hline
\multicolumn{5}{c}{Точность: 0.771605} \\ \hline
\multicolumn{5}{c}{Значение F-метрики: 0.748750} \\ \hline
\multicolumn{5}{c}{Время обучения: 2.024183 с} \\ \hline
\multicolumn{5}{c}{Время классификации: 0.110922 с} \\ \hline
\end{tabular}  
\caption{\label{table:exp1_FFTCoeffsExtractor_MLPClassifier} experiment 1}
\end{table}
"""


class ParsingTestSuite(unittest.TestCase):
    
    def test_confmat(self):
        parsed=log_parsing.parse(log_example.splitlines())
        np.testing.assert_array_equal(np.asarray([[21, 7, 1,  3],
                                                  [ 6, 20, 2,  1],
                                                  [ 3, 2, 35,  2],
                                                  [ 3, 6,  1, 49]]),
                                      parsed.confmat)

    def test_classes(self):
        parsed=log_parsing.parse(log_example.splitlines())
        self.assertEqual(['pullups', 'pushups', 'sits', 'walk'],
                         parsed.classes)

    def test_scores(self):
        parsed=log_parsing.parse(log_example.splitlines())
        self.assertAlmostEqual(0.771605, parsed.accuracy, 0.001)
        self.assertAlmostEqual(0.748750, parsed.f1_score, 0.001)
                               
    def test_times(self):
        parsed=log_parsing.parse(log_example.splitlines())
        self.assertAlmostEqual(2.024183, parsed.fit_time, 0.001)
        self.assertAlmostEqual(0.110922, parsed.score_time, 0.001)

    def test_name(self):
        parsed=log_parsing.parse(log_example.splitlines())
        self.assertEqual("FFTCoeffsExtractor_MLPClassifier",
                         parsed.experiment_name)


class TransformTestSuite(unittest.TestCase):

    def get_out_lines(self):
        parsed=log_parsing.parse(log_example.splitlines())
        latex=log_parsing.to_latex(parsed, "exp1", "experiment")
        out_lines= latex.splitlines()
        return out_lines
    
    def test_transformed_len(self):
        out_lines=self.get_out_lines()
        self.assertEqual(14, len(out_lines))

    def test_environment(self):
        out_lines=self.get_out_lines()
        self.assertEqual(r"\begin{table}[\tableopts]", out_lines[0])
        self.assertEqual(r"\begin{tabular}{\tableformat}", out_lines[1])
        self.assertEqual(r"\end{table}", out_lines[-1])
        self.assertEqual(r"\end{tabular}", out_lines[-3])

    def test_caption(self):
        out_lines=self.get_out_lines()
        expected=r"\caption{\label{table:exp1_FFTCoeffsExtractor_MLPClassifier} experiment}"
        self.assertEqual(expected, out_lines[-2])
        
    def test_content(self):
        out_lines=self.get_out_lines()
        latex_table=out_lines[2:6]
        expected_table=output_example.splitlines()[2:6]
        for index, line in enumerate(latex_table):
            self.assertEqual(expected_table[index], line)
        
    def test_stats(self):
        out_lines=self.get_out_lines()
        latex_table=out_lines[7:11]
        expected_table=output_example.splitlines()[7:11]
        print("\n".join(out_lines))
        for index, line in enumerate(latex_table):
            self.assertEqual(expected_table[index], line)
        
