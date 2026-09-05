import numpy as np
import matplotlib.pyplot as plt

def custom_roc_auc_score(y_true, y_pred_probas, plot=False, ax=None):
    """
    Calculates AUC score and draw ROC curve
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred_probas).flatten()

    # Sorting y_true by the y_pred
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
     
    # Defining the grid size
    n_true_pos = np.sum(y_true == 1)
    n_true_neg = np.sum(y_true == 0)
    
    # Fraction of True Positive objects
    tpr = 0
	# Area Under the Curve value
    auc = 0

    if not plot:
    # Rang approach
        for label in y_true_sorted:
            if label == 1:
                tpr += 1 / n_true_pos
            else:
                auc += tpr / n_true_neg
    
    # If with plot drawing
    else:
		# False Positive count
        fp = 0
	    # Points for graph (start with (0, 0))
        fpr_points = [0.0]
        tpr_points = [0.0]
	    
	    # Rang approach
        for label in y_true_sorted:
            if label == 1:
                tpr += 1 / n_true_pos
            else:
                fp += 1
                fpr = fp / n_true_neg
                auc += tpr / n_true_neg
	            # Save point (FPR, TPR) for graph
                fpr_points.append(fpr)
                tpr_points.append(tpr)
    
	    # Adding (1, 1) point
        if fpr_points[-1] != 1.0 or tpr_points[-1] != 1.0:
            fpr_points.append(1.0)
            tpr_points.append(1.0)
	    
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()
	    
        # ROC-curve
        ax.plot(fpr_points, tpr_points, 
                 label=f'ROC Curve (AUC = {auc:.4f})')
        
        # Diag line (random classifier)
        ax.plot([0, 1], [0, 1], 
                 label='Random Classifier (AUC = 0.5)')
        
        # Graph settings
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])  
	
    return auc

def custom_gini_coefficient(y_true, y_pred_probas):
    """
    Calculates Gini coefficient = 2 * AUC - 1
    """
    return abs(2 * custom_roc_auc_score(y_true, y_pred_probas) - 1)


def custom_recall_score(y_true, y_pred):
    """
    Calculates Recall = TP / (TP + FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def custom_precision_score(y_true, y_pred):
    """
    Calculates Recall = TP / (TP + FP)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def custom_f1_score(y_true, y_pred):
    """
    Calculates F1-score = 2*(Precision*Recall)/(Precision+Recall)
    """
    num = 2 * custom_precision_score(y_true, y_pred) * custom_recall_score(y_true, y_pred)
    den = custom_precision_score(y_true, y_pred) + custom_recall_score(y_true, y_pred)
    return num / den

def custom_pr_auc_score(y_true, y_pred_probas, plot=False, ax=None):
    """
    Calculates PR AUC (Area Under Precision-Recall Curve) score and draw PR curve
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred_probas).flatten()
    
    # Sorting y_true by the y_pred (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Defining the grid size
    n_true_pos = np.sum(y_true == 1)
    
    # Counters
    tp = 0  # True Positives count
    fp = 0  # False Positives count
    
    # Precision value
    precision = 0
    
    # Area Under the Curve value
    auc = 0
    
    if not plot:
        # Rang approach
        for label in y_true_sorted:
            if label == 1:
                tp += 1
                precision = tp / (tp + fp)
                # Calculate areas only in Recall changing cases (label == 1)
                auc += precision / n_true_pos
            else:
                fp += 1
                precision = tp / (tp + fp)
    
    else:
        # Points for graph
        precision_points = [1.0]
        recall_points = [0.0]
        
        tp = 0
        fp = 0
        
        # Rang approach
        for label in y_true_sorted:
            if label == 1:
                tp += 1
                precision = tp / (tp + fp)
                recall = tp / n_true_pos
                
                # Calculate area only in case label == 1
                auc += precision / n_true_pos
                
                precision_points.append(precision)
                recall_points.append(recall)
            else:
                fp += 1
                precision = tp / (tp + fp)
                recall = tp / n_true_pos # Recall doesn't change but save it for the list
                
                precision_points.append(precision)
                recall_points.append(recall)
        
        # Adding (1, 0) point for completeness
        if precision_points[-1] != 0.0 or recall_points[-1] != 1.0:
            precision_points.append(0.0)
            recall_points.append(1.0)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()
            
        ax.plot(recall_points, precision_points, label=f'PR Curve (AUC = {auc:.4f})')
        
        baseline = n_true_pos / len(y_true)
        ax.plot([0, 1], [baseline, baseline], label=f'Baseline (Prevalence = {baseline:.4f})')
        
        ax.set_xlabel('Recall (TPR)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
    
    return auc