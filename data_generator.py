import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
import xlsxwriter


def load_data():
    original_data = pd.read_csv('', header=None)
    return np.array(original_data)


def generate_data():
    or_d = load_data()

    smo = BorderlineSMOTE(kind="borderline-2")

    x = [x[:-1] for x in or_d]
    y = [y[-1] for y in or_d]
    x_smo, y_smo = smo.fit_resample(x, y)

    workbook = xlsxwriter.Workbook('filename.csv')
    worksheet = workbook.add_worksheet('blsmote_data')
    for i in range(len(x_smo)):
        j = 0
        for x in x_smo[i]:
            worksheet.write(i, j, x)
            j += 1
        worksheet.write(i, j, y_smo[i])
    workbook.close()


if __name__ == "__main__":
    generate_data()
