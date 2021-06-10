import numpy as np
from sklearn.metrics import confusion_matrix

predictions = np.array([3, 3, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 3, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0, 2, 0,
 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 3, 0, 3, 0, 3, 3, 3, 0, 0, 0, 3, 3, 0, 3, 3, 3, 3, 0,
 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0,
 0, 0, 3, 3, 0, 3, 3, 0, 3, 0, 0, 0, 0, 3, 0, 0, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0,
 3, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 3, 3, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0,
 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 3, 2, 0, 0, 3, 1, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 1, 0, 3, 2, 0, 0, 0, 0, 0, 0,
 0, 3, 3, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 0, 0,
 3, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 3, 2, 3, 0, 0, 3, 0, 3, 0, 3, 3, 3, 0, 3, 3, 0, 3, 0, 3, 3, 3, 1, 0, 0,
 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0,
 2, 0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 1, 2, 0, 3, 3])

labels = np.array([3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 3])

for i in range(len(labels)):
    if labels[i] == 2:
        labels[i] = 1
    if labels[i] == 3:
        labels[i] = 2

for i in range(len(predictions)):
    if predictions[i] == 2:
        predictions[i] = 1
    if predictions[i] == 3:
        predictions[i] = 2

# print(confusion_matrix(labels, predictions))
# print(confusion_matrix(labels, predictions).ravel())
# tn, fp, fp1, fn, tp, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
# print(tn, fp, fn, tp)
#
# print("tn percentage", (tn * 100) / (tn + fp + fn + tp))
# print("fp percentage", (fp * 100) / (tn + fp + fn + tp))
# print("fn percentage", (fn * 100) / (tn + fp + fn + tp))
# print("tp percentage", (tp * 100) / (tn + fp + fn + tp))

# import scikitplot as skplt
# import matplotlib.pyplot as plt
#
# labels.reshape(len(labels),1)
# predictions.reshape(len(predictions),1)
#
#
# skplt.metrics.plot_roc_curve(labels, predictions)
# plt.show()

# import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    p = predictions.copy()
    l = labels.copy()

    if i == 0:
        for j in range(len(l)):
            if l[j] == 0:
                l[j] = 1
            elif l[j] == 1:
                l[j] = 0
            elif l[j] == 2:
                l[j] = 0

        for j in range(len(p)):
            if p[j] == 0:
                p[j] = 1
            elif p[j] == 1:
                p[j] = 0
            elif p[j] == 2:
                p[j] = 0

    if i == 1:
        for j in range(len(l)):
            if l[j] == 0:
                l[j] = 0
            elif l[j] == 1:
                l[j] = 1
            elif l[j] == 2:
                l[j] = 0

        for j in range(len(p)):
            if p[j] == 0:
                p[j] = 0
            elif p[j] == 1:
                p[j] = 1
            elif p[j] == 2:
                p[j] = 0

    if i == 2:
        for j in range(len(l)):
            if l[j] == 0:
                l[j] = 0
            elif l[j] == 1:
                l[j] = 0
            elif l[j] == 2:
                l[j] = 1

        for j in range(len(p)):
            if p[j] == 0:
                p[j] = 0
            elif p[j] == 1:
                p[j] = 0
            elif p[j] == 2:
                p[j] = 1

    # print(l)
    # print(p)
    fpr[i], tpr[i], _ = roc_curve(l, p)
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc)
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(3):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= 3

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
copd_classes = {
    0: "Healthy",
    1: "Both",
    2: "COPD"
}
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(copd_classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('COPD Prediction System - ROC Graph')
plt.legend(loc="lower right")
plt.show()
