import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true, y_pred, type="normal", onv_vs_all_class = None, normalize=None, classes=None, return_confusion_matrix=False, width=10, height=8, cmap="Blues", fmt="d"):
    """
    y_true: array-like of shape (n_samples,)
    y_pred: array-like of shape (n_samples,)
    normalize: str, one of 'true', 'pred', or 'all'; normalize along rows ('true'), columns ('pred'), or all values ('all') 
    classes: array-like of shape (n_classes,), default=None; list of classes to plot
    return_confusion_matrix: return the confusion matrix
    """    
    from sklearn.metrics import confusion_matrix

    if normalize:
        assert normalize in ['true', 'pred', 'all'], "normalize must be one of 'true', 'pred', or 'all'"
    
    if classes is None:
        classes = np.unique(y_true)

    if type == "one_vs_all":
        assert onv_vs_all_class is not None, "to plot a one_vs_all matrix you need to specify a class with one_vs_all_class"
        assert onv_vs_all_class in classes, "onv_vs_all_class must be one of the classes"
        y_true_one_vs_all = np.where(y_true == onv_vs_all_class, 1, 0)
        y_pred_one_vs_all = np.where(y_pred == onv_vs_all_class, 1, 0)
        conf_matrix = confusion_matrix(y_true_one_vs_all, y_pred_one_vs_all, normalize=normalize)
        classes = ["Not " + str(onv_vs_all_class), str(onv_vs_all_class)]
    elif type == "normal":
        import warnings
        if onv_vs_all_class is not None: warnings.warn("Generating One-vs-All confusion matrices for each class.")
        conf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize, labels=classes)
    else:
        raise ValueError("type must be one of 'normal' or 'one_vs_all'")
    
    plt.figure(figsize=(width, height)) 
    plot = sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap,
                       xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
        
    if return_confusion_matrix:
        return conf_matrix