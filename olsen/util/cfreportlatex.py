"""
Code to parse sklearn classification_report
Original: https://gist.github.com/julienr/6b9b9a03bd8224db7b4f

Modified to work with Python 3 and classification report averages
"""

import collections
import sys


def parse_and_convert_to_latex(clfreport):
    return report_to_latex_table(parse_classification_report(clfreport))


def parse_classification_report(clfreport):
    """
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class
    """
    lines = clfreport.split('\n')
    # Remove empty lines
    lines = list(filter(lambda l: not len(l.strip()) == 0, lines))

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']

    def parse_line(l):
        """Parse a line of classification_report"""
        frag = l.split()
        if len(frag) == 3:
            cls_name = frag[0].strip()
            assert cls_name == 'accuracy'
            fscore, support = frag[1:]
            precision = " "
            recall = " "
            fscore = str(fscore)
            support = str(support)
        else:
            cls_name = " ".join(frag[:-4]).strip()
            precision, recall, fscore, support = frag[-4:]
            precision = str(precision)
            recall = str(recall)
            fscore = str(fscore)
            support = str(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    return data


def report_to_latex_table(data):
    out = ""
    out += "\\begin{table}\n"
    out += "\\centering\n"
    out += "\\begin{tabular}{r | c c c r}\n"
    out += "Class & Precision & Recall & F-score & Support\\\\\n"
    out += "\hline\n"
    out += "\hline\n"
    for cls, scores in data.items():
        if 'accuracy' in cls:
            out += "\hline\n"
        out += cls + " & " + " & ".join([value for value in scores])
        out += "\\\\\n"
    out += "\\end{tabular}\n"
    out += "\\caption{Latex Table from Classification Report}\n"
    out += "\\label{table:classification:report}\n"
    out += "\\end{table}"
    return out


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        data = parse_classification_report(f.read())
    print(report_to_latex_table(data))
