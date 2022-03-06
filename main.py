import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report


import xlrd
from xlutils.copy import copy



# Предобработка геологических данных
Shifts_R = pd.read_excel("data/data.xls", sheet_name=1, header=1, index_col=0, usecols=[1, 4], )
Shifts_R = Shifts_R.round({"Az1": 0, }).astype('int32')
X = Shifts_R.values
plt.figure(figsize=(9, 6))

len(Shifts_R.values)

# Расчет качества кластеризации для разого количества подмножеств n
classQuality = []

print('Enter max number of classes ')
maxClasses = int(input())

for n in range(2, maxClasses + 1):
    cluster = KMeans(n_clusters=n, random_state=0).fit(X)
    result = cluster.predict(X)
    k = sklearn.metrics.davies_bouldin_score(X, result)
    classQuality.append(k)

# Построение графика
plt.plot(range(2, maxClasses + 1), classQuality, color='red', marker='o')
plt.grid()
plt.show()

# Кластеризация по оптимальному числу кластеров


best = min(classQuality)

bestIndex = classQuality.index(best)

print('Best number of classes = ', bestIndex + 2, ' with value = ', best)

print('How many classes do you want? ')

numberOfClasses = int(input())

cluster = KMeans(n_clusters=numberOfClasses, random_state=0).fit(X)
result = cluster.predict(X)

# Запись вывода в Exel файл

workSheetIndex = 1

rb = xlrd.open_workbook("data/data.xls", formatting_info=True)
r_sheet = rb.sheet_by_index(workSheetIndex)
r = r_sheet.nrows
wb = copy(rb)
sheet = wb.get_sheet(workSheetIndex)

# The data
classColIndex = 11
xColIndex = 2
yColIndex = 3
azColIndex = 4
headerHeight = 2

i = 0

row = sheet.row(0)
row.write(classColIndex, 'Classes')

row = sheet.row(1)
row.write(classColIndex, '№')

for num in result:
    value = int(num)
    row = sheet.row(i + headerHeight)
    i = i + 1
    row.write(classColIndex, value)

# Save the result
wb.save('data/data.xls')

# Визуализация сдвигов по кластерам

book = xlrd.open_workbook('data/data.xls')

sh = book.sheet_by_index(workSheetIndex)

# print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))

x, y, k, Az = [], [], [], []

for rx in range(headerHeight, sh.nrows):
    x.append(sh.cell(rx, xColIndex).value)
    y.append(sh.cell(rx, yColIndex).value)
    k.append(sh.cell(rx, classColIndex).value)
    Az.append(sh.cell(rx, azColIndex).value)

k_color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

medianAz = [0] * numberOfClasses
for i in range(len(k)):
    medianAz[int(k[i])] = medianAz[int(k[i])] + Az[i]

for j in range(numberOfClasses):
    medianAz[j] = medianAz[j] / int(k.count(j))

for j in range(numberOfClasses):
    print((j + 1), ') elements count ', int(k.count(j) / 2), ', mean Az ', str(round(medianAz[j],2)))

print('Press any key to vis all classes or \'C\' to choose')

ans = input()[0]

strToShow = 'All'

if ans == 'c' or ans == 'C':
    print('Input class numbers to show ')
    strToShow = input()
    classesToShow = [int(s) for s in strToShow.split()]
else:
    classesToShow = list(range(1, numberOfClasses + 1))

colors = []
for j in range(numberOfClasses):
    if j < len(k_color):
        colors.append(k_color[j])
    else:
        tmp = '#' + str(random.randint(0, 9)) + \
              str(random.randint(0, 9)) + \
              str(random.randint(0, 9)) + \
              str(random.randint(0, 9)) + \
              str(random.randint(0, 9)) + \
              str(random.randint(0, 9))
        colors.append(tmp)

plt.figure(figsize=(12, 12), dpi=300.0)
for i in range(0, len(x), 2):
    if (int(k[i]) + 1) in classesToShow:
        plt.plot(x[i:i + 2], y[i:i + 2], colors[int(k[i])], linewidth=1)

existsColors = []
markerLabels = []

for i in range(len(classesToShow)):
    existsColors.append(colors[classesToShow[i] - 1])
    markerLabels.append(str(classesToShow[i]) + '] Az = ' + str(round(medianAz[int(classesToShow[i] - 1)], 2)))

markers = [plt.Line2D([0, 0], [0, 0], color=color, linestyle='-') for color in existsColors]
plt.legend(markers, markerLabels, numpoints=1, loc=3, title="Цвета класса и средний азимут")

filename = 'Cl_' + str(maxClasses) + ' Auto_' + str((bestIndex + 2)) + ' Usr_' + str(
    numberOfClasses) + ' Vis_' + strToShow + '.png'

plt.savefig('data/'+filename)

# plt.show()
