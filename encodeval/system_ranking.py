from dataclasses import dataclass
from itertools import product
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

# Aliases for data types used for clarity
SegmentScores = pd.DataFrame
ResamplingScores = pd.DataFrame
HeadToHeads = pd.DataFrame


def rank_systems(
    seg_scores: SegmentScores,
    threshold_p_value: float = 0.05,
    bootstrap_resampling: bool = False,
    sample_size: int = 100,
    num_splits: int = 100,
    tie_epsilon: float = 0.01,
) -> pd.Series:
    """
    Ranks systems by comparing their segment-level scores using statistical tests.

    Args:
        seg_scores (SegmentScores): DataFrame of segment-level scores.
        threshold_p_value (float): Significance threshold for statistical tests.
        bootstrap_resampling (bool): Whether to use bootstrap resampling.
        sample_size (int): Size of each bootstrap sample.
        num_splits (int): Number of bootstrap resampling splits.
        tie_epsilon (float): Threshold for treating two scores as a tie.

    Returns:
        pd.Series: System rankings (1 is best).
    """
    scores = seg_scores
    if bootstrap_resampling:
        scores = bootstrap_resampling(seg_scores, sample_size, num_splits)

    head_to_heads = build_head_to_heads(scores, tie_epsilon)
    clusters = build_clusters(seg_scores, head_to_heads, threshold_p_value)

    result = pd.Series(index=scores.columns)
    for i, cluster in enumerate(clusters):
        for system in cluster:
            result.loc[system] = i + 1

    return result


def bootstrap_resampling(
    seg_scores: SegmentScores, sample_size: int, num_splits: int
) -> ResamplingScores:
    """
    Performs bootstrap resampling of segment scores.

    Args:
        seg_scores (SegmentScores): DataFrame of segment-level scores [N x M].
        sample_size (int): Number of examples to sample per split.
        num_splits (int): Number of bootstrap splits.

    Returns:
        ResamplingScores: Resampled scores averaged across splits [num_splits x M].
    """
    seg_scores_values = seg_scores.values.T  # M x N
    population_size = seg_scores_values.shape[1]

    # Sample indices with replacement
    subsample_ids = np.random.choice(population_size, size=(sample_size, num_splits), replace=True)
    subsamples = np.take(seg_scores_values, subsample_ids, axis=1)  # M x sample_size x num_splits
    resample_scores_values = np.mean(subsamples, axis=1)  # M x num_splits

    return pd.DataFrame(resample_scores_values.T, columns=seg_scores.columns)


@dataclass
class HeadToHead:
    """Stores head-to-head comparison between two systems."""
    ties: int
    wins: int
    losses: int
    p_value: float

    def __str__(self) -> str:
        return f"p-value: {self.p_value:.3f} ties: {self.ties} wins: {self.wins} losses: {self.losses}"


def build_head_to_heads(scores, eps=0.01) -> HeadToHeads:
    """
    Constructs a pairwise head-to-head comparison table between all systems.

    Args:
        scores (pd.DataFrame): DataFrame of segment-level scores (e.g., from resampling).
        eps (float): Tolerance value to treat score differences as ties.

    Returns:
        HeadToHeads: DataFrame of HeadToHead objects.
    """
    systems = scores.columns
    comparisons = pd.DataFrame(index=systems, columns=systems, dtype=object)

    for system1, system2 in product(systems, repeat=2):
        if system1 == system2:
            comparisons.loc[system1, system1] = None
            continue
        delta = scores[system1] - scores[system2]
        ties = (delta.abs() < eps).sum()
        wins = (delta >= eps).sum()
        losses = (delta <= -eps).sum()
        ttest_result = stats.ttest_rel(
            scores[system1], scores[system2], alternative="greater"
        )
        comparisons.loc[system1, system2] = HeadToHead(
            ties, wins, losses, ttest_result.pvalue
        )

    return comparisons


def build_clusters(
    seg_scores: SegmentScores,
    head_to_heads: HeadToHeads,
    threshold_p_value: float = 0.05,
):
    """
    Groups systems into clusters of indistinguishable performance using statistical tests.

    Args:
        seg_scores (SegmentScores): Original scores used to determine average order.
        head_to_heads (HeadToHeads): Pairwise system comparisons.
        threshold_p_value (float): Significance threshold to determine difference.

    Returns:
        list: List of clusters (each a list of system names).
    """
    avg_scores = seg_scores.mean()
    p_values = head_to_heads.map(lambda x: x.p_value if x else np.nan)
    diffs = p_values.map(lambda x: x < threshold_p_value)

    clusters = [[]]
    descending_systems = avg_scores.sort_values(ascending=False).index
    for system in descending_systems:
        curr_cluster = clusters[-1]
        if any(diffs.loc[curr_cluster, system]):
            clusters.append([system])
        else:
            curr_cluster.append(system)

    return clusters


def average_score_across_subsets_accuracy_ndcg(results, metric):
    """
    Compute the average accuracy or NDCG score across subsets.

    Args:
        results (dict): Dictionary containing:
            - metric (str): List of per-instance scores.
            - subset (list): List of corresponding subset identifiers.
        metric (str): Metric name, either 'accuracy' or 'ndcg'.

    Returns:
        float: Mean score across all subsets.
    """
    scores = results.get(metric, None)
    if scores is None:
        raise ValueError(f"Results do not contain scores for metric {metric}.")

    subsets = results.get("subset", None)
    if subsets is None:
        raise ValueError("Results do not contain subsets.")

    unique_subsets = list(set(subsets))

    scores_by_subset = []
    for subset in unique_subsets:
        subset_indices = [i for i, s in enumerate(subsets) if s == subset]
        subset_scores = [scores[i] for i in subset_indices]
        scores_by_subset.append(np.mean(subset_scores))

    return np.mean(scores_by_subset)


def average_score_across_subsets_spearman(results):
    """
    Compute the average Spearman correlation across subsets.

    Args:
        results (dict): Dictionary containing:
            - prediction (list): Predicted values.
            - label (list): Ground truth values.
            - subset (list): List of subset identifiers.

    Returns:
        float: Mean Spearman correlation across subsets.
    """
    predictions = results.get("prediction", None)
    if predictions is None:
        raise ValueError("Results do not contain predictions.")

    labels = results.get("label", None)
    if labels is None:
        raise ValueError("Results do not contain labels.")

    subsets = results.get("subset", None)
    if subsets is None:
        raise ValueError("Results do not contain subset.")

    unique_subsets = list(set(subsets))

    scores_by_subset = []
    for subset in unique_subsets:
        subset_indices = [i for i, s in enumerate(subsets) if s == subset]
        subset_predictions = [predictions[i] for i in subset_indices]
        subset_labels = [labels[i] for i in subset_indices]
        scores_by_subset.append(stats.spearmanr(subset_predictions, subset_labels).correlation)

    return np.mean(scores_by_subset)


def average_score_across_subsets_f1_score(results):
    """
    Compute the average micro F1 score across subsets.

    Args:
        results (dict): Dictionary containing:
            - prediction (list): List of token-level prediction sequences.
            - labels (list): List of token-level ground truth sequences.
            - subset (list): List of subset identifiers.

    Returns:
        float: Mean micro F1 score across subsets.
    """
    prediction_token = results.get("prediction", None)
    if prediction_token is None:
        raise ValueError("Results do not contain prediction.")

    labels_token = results.get("labels", None)
    if labels_token is None:
        raise ValueError("Results do not contain labels.")

    subsets = results.get("subset", None)
    if subsets is None:
        raise ValueError("Results do not contain subset.")

    unique_labels = sorted(list(set(pred for preds in prediction_token for pred in preds)))
    unique_subsets = list(set(subsets))

    scores_by_subset = []
    for subset in unique_subsets:
        subset_indices = [i for i, s in enumerate(subsets) if s == subset]
        subset_predictions = [prediction_token[i] for i in subset_indices]
        subset_labels = [labels_token[i] for i in subset_indices]
        subset_flat_predictions = [pred for preds in subset_predictions for pred in preds]
        subset_flat_labels = [label for labels in subset_labels for label in labels]
        scores_by_subset.append(
            f1_score(subset_flat_labels, subset_flat_predictions, average="micro", labels=unique_labels)
        )

    return np.mean(scores_by_subset)


def get_best_hyperparam(base_path, model, task_type, dataset, metric):
    """
    Find the best-performing hyperparameter based on validation metric.

    Args:
        base_path (str): Root path to results.
        model (str): Model name.
        task_type (str): Task type (e.g., SC, IR, SR, TC).
        dataset (str): Dataset name.
        metric (str): Evaluation metric.

    Returns:
        str or None: Name of best hyperparameter folder, or None if not found.
    """
    dataset_path = os.path.join(base_path, model, task_type, dataset)
    if not os.path.isdir(dataset_path):
        print(f"Dataset directory {dataset_path} not found.")
        return None

    best_hyperparam = None
    best_score = float("-inf")

    for hyperparam in os.listdir(dataset_path):
        results_file = os.path.join(dataset_path, hyperparam, "results.json")
        if os.path.isfile(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                results = data.get("validation", None)
                if results is None:
                    raise ValueError("Results do not contain validation data.")
                if metric in ("accuracy", "ndcg"):
                    score = average_score_across_subsets_accuracy_ndcg(results, metric)
                elif metric == "spearman":
                    score = average_score_across_subsets_spearman(results)
                elif metric == "f1_score":
                    score = average_score_across_subsets_f1_score(results)
                else:
                    raise ValueError(f"Unsupported metric {metric}.")
                if score > best_score:
                    best_score = score
                    best_hyperparam = hyperparam
            except Exception as e:
                print(f"Error reading {results_file}: {e}")

    return best_hyperparam


def get_test_segment_results_by_subset(base_path, model, task_type, dataset, metric):
    """
    Retrieve segment-level test scores grouped by subset for the best hyperparameter.

    Args:
        base_path (str): Base results directory.
        model (str): Model name.
        task_type (str): Task type (e.g., SC, IR, etc.).
        dataset (str): Dataset name.
        metric (str): Evaluation metric.

    Returns:
        dict or None: Dictionary mapping each subset to a list of scores.
    """
    if "results.json" not in os.listdir(os.path.join(base_path, model, task_type, dataset)):
        best_hyperparam = get_best_hyperparam(base_path, model, task_type, dataset, metric)
        if best_hyperparam is None:
            print("No best hyperparameter found.")
            return None
        results_file = os.path.join(base_path, model, task_type, dataset, best_hyperparam, "results.json")

    else:
        results_file = os.path.join(base_path, model, task_type, dataset, "results.json")
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return None

    test_data = data.get("test", {})
    accuracies = test_data.get(metric, [])
    subsets = test_data.get("subset", [])

    if len(accuracies) != len(subsets):
        print("Warning: The number of accuracies and subsets do not match in the test split.")

    grouped_results = {}
    for acc, subset in zip(accuracies, subsets):
        grouped_results.setdefault(subset, []).append(acc)

    return grouped_results


def get_system_average_scores_subsets(base_path, models, task_type, dataset, metric):
    """
    Compute average scores for each model and subset.

    Args:
        base_path (str): Results root directory.
        models (list): List of model names.
        task_type (str): Task type (SC, IR, etc.).
        dataset (str): Dataset name.
        metric (str): Metric name (e.g., "accuracy", "ndcg").

    Returns:
        dict: Nested dictionary mapping subset -> model -> average score.
    """
    segments_by_model = {}
    for model in models:
        results = get_test_segment_results_by_subset(base_path, model, task_type, dataset, metric)
        segments_by_model[model] = results

    segments_by_subset = {}
    subsets = next(iter(segments_by_model.values())).keys()
    for subset in subsets:
        subset_segments = {}
        for model in models:
            if segments_by_model[model] is not None:
                subset_segments[model] = segments_by_model[model][subset]
        segments_by_subset[subset] = pd.DataFrame(subset_segments)

    average_scores = {}
    for subset, df in segments_by_subset.items():
        average_scores[subset] = df.mean(axis=0).to_dict()

    return average_scores


def get_system_rankings_subsets(base_path, models, task_type, dataset, metric):
    """
    Compute model rankings per subset based on average scores.

    Args:
        base_path (str): Results directory.
        models (list): List of model names.
        task_type (str): Task type.
        dataset (str): Dataset name.
        metric (str): Evaluation metric.

    Returns:
        dict: Dictionary mapping subset to ranked systems.
    """
    segments_by_model = {}
    for model in models:
        segments_by_model[model] = get_test_segment_results_by_subset(base_path, model, task_type, dataset, metric)

    segments_by_subset = {}
    subsets = next(iter(segments_by_model.values())).keys()
    for subset in subsets:
        subset_segments = {}
        for model in models:
            if segments_by_model[model] is not None:
                subset_segments[model] = segments_by_model[model][subset]
        segments_by_subset[subset] = pd.DataFrame(subset_segments)

    rankings = {}
    for subset in subsets:
        rankings[subset] = rank_systems(segments_by_subset[subset])

    return rankings


def get_system_subsets(base_path, model, task_type, dataset, metric):
    """
    Retrieve system predictions and labels grouped by subset.

    Args:
        base_path (str): Root results directory.
        model (str): Model name.
        task_type (str): Task type.
        dataset (str): Dataset name.
        metric (str): Metric name.

    Returns:
        tuple: (predictions_by_subset, labels_by_subset)
    """
    if "results.json" not in os.listdir(os.path.join(base_path, model, task_type, dataset)):    
        best_hyperparam = get_best_hyperparam(base_path, model, task_type, dataset, metric)
        if best_hyperparam is None:
            return None
        results_file = os.path.join(base_path, model, task_type, dataset, best_hyperparam, "results.json")
        
    else:
        results_file = os.path.join(base_path, model, task_type, dataset, "results.json")

    results = json.load(open(results_file))
    test_data = results.get("test", {})
    predictions = test_data.get("prediction", [])
    subsets = test_data.get("subset", [])
    labels = test_data.get("label", [])

    assert len(predictions) == len(labels) == len(subsets), "The number of predictions, labels, and subsets do not match."

    predictions_by_subset = {}
    labels_by_subset = {}

    for subset in set(subsets):
        predictions_by_subset[subset] = [pred for pred, sub in zip(predictions, subsets) if sub == subset]
        labels_by_subset[subset] = [label for label, sub in zip(labels, subsets) if sub == subset]

    return predictions_by_subset, labels_by_subset


def get_systems_averages_subsets_spearman(base_path, models, task_type, dataset, metric):
    """
    Compute average Spearman correlation per model and subset.

    Args:
        base_path (str): Root path.
        models (list): List of models.
        task_type (str): Task type (e.g., SR).
        dataset (str): Dataset name.
        metric (str): Metric name (should be 'spearman').

    Returns:
        tuple: (data, found_models) where data is subset -> model -> correlation.
    """
    model_predictions, model_labels, found_models = {}, {}, []

    for model in models:
        results = get_system_subsets(base_path, model, task_type, dataset, metric)
        if results is None:
            continue
        preds, labels = results
        model_predictions[model] = preds
        model_labels[model] = labels
        found_models.append(model)

    subsets = next(iter(model_predictions.values())).keys()
    data = {}
    for subset in subsets:
        subset_data = {}
        for model in found_models:
            subset_data[model] = stats.spearmanr(model_predictions[model][subset], model_labels[model][subset]).correlation
        data[subset] = subset_data

    return data, found_models


def get_systems_bootstrapped_spearman_r(base_path, models, task_type, dataset, metric):
    """
    Compute bootstrapped Spearman correlations for each model and subset.

    Args:
        base_path (str): Root path.
        models (list): List of model names.
        task_type (str): Task type.
        dataset (str): Dataset name.
        metric (str): Metric name.

    Returns:
        dict: Dictionary of subset -> DataFrame of bootstrapped results.
    """
    model_predictions = {}
    model_labels = {}

    for model in models:
        preds, labels = get_system_subsets(base_path, model, task_type, dataset, metric)
        model_predictions[model] = preds
        model_labels[model] = labels

    subsets_data = {}
    subsets = next(iter(model_predictions.values())).keys()

    for subset in subsets:
        subset_predictions = {m: model_predictions[m][subset] for m in models}
        subset_labels = model_labels[models[0]][subset]

        assert all(labels == subset_labels for labels in [model_labels[m][subset] for m in models])

        predictions_df = pd.DataFrame(subset_predictions)
        full_spearman_data = []

        for _ in range(100):
            idxs = np.random.choice(len(subset_labels), len(subset_labels), replace=True)
            sample_labels = [subset_labels[i] for i in idxs]
            step_spearman = {
                model: stats.spearmanr(predictions_df[model].iloc[idxs], sample_labels).correlation
                for model in models
            }
            full_spearman_data.append(step_spearman)

        subsets_data[subset] = pd.DataFrame(full_spearman_data)

    return subsets_data


def get_averages_and_rankings_bootstrapped_spearman_r(base_path, models, task_type, dataset, metric):
    """
    Compute average Spearman scores and bootstrapped rankings for each subset.

    Args:
        base_path (str): Results directory.
        models (list): List of model names.
        task_type (str): Task type (e.g., SR).
        dataset (str): Dataset name.
        metric (str): Metric name (should be 'spearman').

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    average_scores, found_models = get_systems_averages_subsets_spearman(base_path, models, task_type, dataset, metric)
    spearman_by_subset = get_systems_bootstrapped_spearman_r(base_path, found_models, task_type, dataset, metric)

    rankings = {subset: rank_systems(spearman_by_subset[subset]) for subset in spearman_by_subset}
    return pd.DataFrame(average_scores), pd.DataFrame(rankings)


def get_system_subsets_token(base_path, model, task_type, dataset, metric):
    """
    Retrieve token-level predictions and labels grouped by subset.

    Args:
        base_path (str): Results directory.
        model (str): Model name.
        task_type (str): Task type.
        dataset (str): Dataset name.
        metric (str): Metric name.

    Returns:
        tuple: (predictions_by_subset, labels_by_subset)
    """
    if "results.json" not in os.listdir(os.path.join(base_path, model, task_type, dataset)):    
        best_hyperparam = get_best_hyperparam(base_path, model, task_type, dataset, metric)
        if best_hyperparam is None:
            return None
        results_file = os.path.join(base_path, model, task_type, dataset, best_hyperparam, "results.json")
        
    else:
        results_file = os.path.join(base_path, model, task_type, dataset, "results.json")
    
    results = json.load(open(results_file))
    test_data = results.get("test", {})
    predictions = test_data.get("prediction", [])
    subsets = test_data.get("subset", [])
    labels = test_data.get("labels", [])

    assert len(predictions) == len(labels) == len(subsets), "Mismatched prediction/label/subset lengths."

    predictions_by_subset = {}
    labels_by_subset = {}

    for subset in set(subsets):
        predictions_by_subset[subset] = [pred for pred, sub in zip(predictions, subsets) if sub == subset]
        labels_by_subset[subset] = [label for label, sub in zip(labels, subsets) if sub == subset]

    return predictions_by_subset, labels_by_subset


def get_systems_averages_subsets_f1_score(base_path, models, task_type, dataset, metric):
    """
    Compute average micro F1 scores for each model and subset.

    Args:
        base_path (str): Results directory.
        models (list): List of models.
        task_type (str): Task type.
        dataset (str): Dataset name.
        metric (str): Metric name (should be 'f1_score').

    Returns:
        tuple: (data_dict, found_models)
    """
    model_predictions = {}
    model_labels = {}
    found_models = []

    for model in models:
        results = get_system_subsets_token(base_path, model, task_type, dataset, metric)
        if results is None:
            continue
        preds, labels = results
        model_predictions[model] = preds
        model_labels[model] = labels
        found_models.append(model)

    subsets = next(iter(model_predictions.values())).keys()
    data = {}

    for subset in subsets:
        subset_data = {}
        for model in found_models:
            predictions = model_predictions[model][subset]
            labels = model_labels[model][subset]
            flat_predictions = [pred for preds in predictions for pred in preds]
            flat_labels = [label for labels_ in labels for label in labels_]
            unique_labels = sorted(list(set(flat_predictions)))
            subset_data[model] = f1_score(flat_labels, flat_predictions, average="micro", labels=unique_labels)
        data[subset] = subset_data

    return data, found_models


def get_systems_bootstrapped_f1_score(base_path, models, task_type, dataset, metric):
    """
    Compute bootstrapped micro F1 scores for each model and subset.

    Args:
        base_path (str): Path to results.
        models (list): List of model names.
        task_type (str): Task type (e.g., TC).
        dataset (str): Dataset name.
        metric (str): Metric name.

    Returns:
        dict: Mapping from subset to DataFrame of bootstrapped F1 scores.
    """
    model_predictions = {}
    model_labels = {}

    for model in models:
        preds, labels = get_system_subsets_token(base_path, model, task_type, dataset, metric)
        model_predictions[model] = preds
        model_labels[model] = labels

    subsets = next(iter(model_predictions.values())).keys()
    subsets_data = {}

    for subset in subsets:
        subset_predictions = {model: model_predictions[model][subset] for model in models}
        subset_labels = model_labels[models[0]][subset]

        assert all(model_labels[model][subset] == subset_labels for model in models)

        predictions_df = pd.DataFrame(subset_predictions)
        full_f1_data = []

        for _ in range(100):
            idxs = np.random.choice(len(subset_labels), len(subset_labels), replace=True)
            sample_labels = [subset_labels[i] for i in idxs]
            flat_sample_labels = [label for labels_ in sample_labels for label in labels_]
            step_f1 = {}
            for model in models:
                sample_predictions = predictions_df[model].iloc[idxs]
                flat_sample_predictions = [pred for preds in sample_predictions for pred in preds]
                unique_labels = sorted(list(set(flat_sample_predictions)))
                step_f1[model] = f1_score(flat_sample_labels, flat_sample_predictions, average="micro", labels=unique_labels)
            full_f1_data.append(step_f1)

        subsets_data[subset] = pd.DataFrame(full_f1_data)

    return subsets_data


def get_averages_and_rankings_bootstrapped_f1_score(base_path, models, task_type, dataset, metric):
    """
    Compute average and bootstrapped rankings for micro F1 score.

    Args:
        base_path (str): Results path.
        models (list): List of models.
        task_type (str): Task type.
        dataset (str): Dataset name.
        metric (str): Metric name.

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    average_scores, found_models = get_systems_averages_subsets_f1_score(base_path, models, task_type, dataset, metric)
    f1_by_subset = get_systems_bootstrapped_f1_score(base_path, found_models, task_type, dataset, metric)

    rankings = {subset: rank_systems(f1_by_subset[subset]) for subset in f1_by_subset}
    return pd.DataFrame(average_scores), pd.DataFrame(rankings)


def get_results_sc(base_path, models, dataset, valid_langs):
    """
    Compute final SC task results: average scores and rankings.

    Args:
        base_path (str): Path to results.
        models (list): List of model names.
        dataset (str): Dataset name.
        valid_langs (list): List of valid subsets to include.

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    average_scores = get_system_average_scores_subsets(base_path, models, "SC", dataset, "accuracy")
    langs = [lang for lang in valid_langs if lang in average_scores]
    average_scores = pd.DataFrame(average_scores)[langs]
    average_scores["average"] = average_scores.mean(axis=1)
    system_rankings = get_system_rankings_subsets(base_path, models, "SC", dataset, "accuracy")
    system_rankings = pd.DataFrame(system_rankings)[langs]
    system_rankings["borda"] = system_rankings.mean(axis=1)
    return average_scores, system_rankings


def get_results_ir(base_path, models, dataset, valid_langs):
    """
    Compute final IR task results: average scores and rankings.

    Args:
        base_path (str): Path to results.
        models (list): List of model names.
        dataset (str): Dataset name.
        valid_langs (list): List of valid subsets to include.

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    average_scores = get_system_average_scores_subsets(base_path, models, "IR", dataset, "ndcg")
    langs = [lang for lang in valid_langs if lang in average_scores]
    average_scores = pd.DataFrame(average_scores)[langs]
    average_scores["average"] = average_scores.mean(axis=1)
    system_rankings = get_system_rankings_subsets(base_path, models, "IR", dataset, "ndcg")
    system_rankings = pd.DataFrame(system_rankings)[langs]
    system_rankings["borda"] = system_rankings.mean(axis=1)
    return average_scores, system_rankings


def get_results_sr(base_path, models, dataset, valid_langs):
    """
    Compute final SR task results: average scores and rankings.

    Args:
        base_path (str): Results path.
        models (list): List of model names.
        dataset (str): Dataset name.
        valid_langs (list): List of valid subsets.

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    average_scores, rankings = get_averages_and_rankings_bootstrapped_spearman_r(base_path, models, "SR", dataset, "spearman")
    langs = [lang for lang in valid_langs if lang in average_scores]
    average_scores, rankings = average_scores[langs], rankings[langs]
    average_scores["average"] = average_scores.mean(axis=1)
    rankings["borda"] = rankings.mean(axis=1)
    return average_scores, rankings


def get_results_tc(base_path, models, dataset, valid_langs):
    """
    Compute final TC task results: average scores and rankings.

    Args:
        base_path (str): Results path.
        models (list): List of model names.
        dataset (str): Dataset name.
        valid_langs (list): List of valid subsets.

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    average_scores, rankings = get_averages_and_rankings_bootstrapped_f1_score(base_path, models, "TC", dataset, "f1_score")
    langs = [lang for lang in valid_langs if lang in average_scores]
    average_scores, rankings = average_scores[langs], rankings[langs]
    average_scores["average"] = average_scores.mean(axis=1)
    rankings["borda"] = rankings.mean(axis=1)
    return average_scores, rankings


def get_results(base_path, models, task_type, dataset, valid_langs):
    """
    General dispatcher for getting results for any task type.

    Args:
        base_path (str): Base directory path.
        models (list): List of models.
        task_type (str): Task type ("SC", "IR", "SR", or "TC").
        dataset (str): Dataset name.
        valid_langs (list): List of valid subsets to include.

    Returns:
        tuple: (average_scores_df, rankings_df)
    """
    if task_type == "IR":
        return get_results_ir(base_path, models, dataset, valid_langs)
    elif task_type == "SC":
        return get_results_sc(base_path, models, dataset, valid_langs)
    elif task_type == "SR":
        return get_results_sr(base_path, models, dataset, valid_langs)
    elif task_type == "TC":
        return get_results_tc(base_path, models, dataset, valid_langs)
    else:
        raise NotImplementedError(f"Unknown task type: {task_type}")
