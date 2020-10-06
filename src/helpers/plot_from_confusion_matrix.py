import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(confusion_matrix, labels, cmap: str = 'Blues'):
    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, cmap=cmap)
