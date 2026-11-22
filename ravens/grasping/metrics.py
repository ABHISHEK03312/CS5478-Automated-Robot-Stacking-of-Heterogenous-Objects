"""Metrics for binary grasp-success prediction."""

import numpy as np


def binary_grasp_metrics(labels, probabilities, threshold=0.5):
  labels = np.asarray(labels, dtype=np.int32)
  probabilities = np.asarray(probabilities, dtype=np.float64)
  if labels.ndim != 1 or probabilities.shape != labels.shape or not len(labels):
    raise ValueError('labels and probabilities must be non-empty 1D arrays')
  if not np.all(np.isin(labels, (0, 1))):
    raise ValueError('labels must be binary')
  probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
  predictions = probabilities >= threshold
  positives = labels == 1
  negatives = ~positives
  true_positive = int(np.sum(predictions & positives))
  true_negative = int(np.sum(~predictions & negatives))
  false_positive = int(np.sum(predictions & negatives))
  false_negative = int(np.sum(~predictions & positives))

  def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')

  recall = ratio(true_positive, true_positive + false_negative)
  specificity = ratio(true_negative, true_negative + false_positive)
  balanced = (float(np.nanmean([recall, specificity]))
              if not (np.isnan(recall) and np.isnan(specificity)) else float('nan'))
  log_loss = -np.mean(labels * np.log(probabilities) +
                      (1 - labels) * np.log(1 - probabilities))

  # Mann-Whitney rank formulation of ROC AUC. Ties receive average ranks.
  order = np.argsort(probabilities, kind='mergesort')
  ranks = np.empty(len(order), dtype=np.float64)
  sorted_probabilities = probabilities[order]
  start = 0
  while start < len(order):
    end = start + 1
    while (end < len(order) and
           sorted_probabilities[end] == sorted_probabilities[start]):
      end += 1
    ranks[order[start:end]] = (start + 1 + end) / 2.0
    start = end
  n_positive = int(np.sum(positives))
  n_negative = len(labels) - n_positive
  if n_positive and n_negative:
    auc = ((np.sum(ranks[positives]) - n_positive * (n_positive + 1) / 2) /
           (n_positive * n_negative))
  else:
    auc = float('nan')

  return {
      'samples': int(len(labels)),
      'positives': n_positive,
      'negatives': n_negative,
      'accuracy': ratio(true_positive + true_negative, len(labels)),
      'balanced_accuracy': balanced,
      'precision': ratio(true_positive, true_positive + false_positive),
      'recall': recall,
      'specificity': specificity,
      'roc_auc': float(auc),
      'log_loss': float(log_loss),
      'true_positive': true_positive,
      'true_negative': true_negative,
      'false_positive': false_positive,
      'false_negative': false_negative,
  }
