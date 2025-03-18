import random
import math
from collections import Counter, defaultdict
import sys
from math import log2
import csv
import numpy as np
from numpy.random import choice
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score
from scipy.stats import chi2_contingency

class DecisionNode:
    def __init__(self, feature=None, threshold=None, branches=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.branches = branches
        self.value = value

def calculate_f_statistic(data, feature, target):
    classes = list(set(data[target]))
    groups = []
    for cls in classes:
        group = [float(data[feature][i]) for i in range(len(data[target])) if data[target][i] == cls]
        groups.append(group)
    
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        return 0.0
    
    all_values = [float(x) for x in data[feature]]
    overall_mean = np.mean(all_values)
    
    ss_between = 0.0
    for group in groups:
        group_mean = np.mean(group)
        ss_between += len(group) * (group_mean - overall_mean) ** 2
    
    ss_within = 0.0
    for group in groups:
        group_mean = np.mean(group)
        ss_within += sum((x - group_mean) ** 2 for x in group)
    
    k = len(groups)
    df_between = k - 1
    df_within = len(all_values) - k
    
    if df_within == 0 or df_between == 0:
        return 0.0
    
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    return ms_between / ms_within if ms_within != 0 else 0.0

def calculate_cramers_v(data, feature, target):
    target_values = list(set(data[target]))
    feature_values = list(set(data[feature]))
    
    contingency = {f_val: {t_val: 0 for t_val in target_values} for f_val in feature_values}
    
    for i in range(len(data[target])):
        f_val = data[feature][i]
        t_val = data[target][i]
        contingency[f_val][t_val] += 1
    
    observed = np.array([[contingency[f][t] for t in target_values] for f in feature_values])
    
    if observed.size == 0:
        return 0.0
    
    chi2, _, _, _ = chi2_contingency(observed)
    n = np.sum(observed)
    
    if n == 0:
        return 0.0
    
    phi2 = chi2 / n
    rows, cols = observed.shape
    min_dim = min(rows - 1, cols - 1)
    
    return np.sqrt(phi2 / min_dim) if min_dim != 0 else 0.0

def get_entropy(labels):
    label_counts = Counter(labels)
    total = len(labels)
    entropy = 0.0
    for count in label_counts.values():
        probability = count / total
        entropy -= probability * log2(probability)
    return entropy

def information_gain_numerical(data, feature, target):
    total_entropy = get_entropy(data[target])
    values = [float(x) for x in data[feature]]
    if len(values) == 0:
        return 0, None

    sorted_values = sorted(values)
    unique_values = sorted(list(set(sorted_values)))
    if len(unique_values) <= 1:
        return 0, None

    max_ig = -1
    best_threshold = None

    thresholds = [(unique_values[i] + unique_values[i+1])/2 for i in range(len(unique_values)-1)]

    for threshold in thresholds:
        left_labels = []
        right_labels = []
        for i in range(len(data[feature])):
            value = float(data[feature][i])
            if value <= threshold:
                left_labels.append(data[target][i])
            else:
                right_labels.append(data[target][i])

        if not left_labels or not right_labels:
            continue

        left_entropy = get_entropy(left_labels)
        right_entropy = get_entropy(right_labels)
        prob_left = len(left_labels) / len(data[target])
        prob_right = len(right_labels) / len(data[target])
        current_ig = total_entropy - (prob_left * left_entropy + prob_right * right_entropy)

        if current_ig > max_ig:
            max_ig = current_ig
            best_threshold = threshold

    return max_ig, best_threshold

def gain_ratio(data, feature, target, numerical_features):
    if feature in numerical_features:
        ig, threshold = information_gain_numerical(data, feature, target)
        if ig == 0 or threshold is None:
            return 0, None

        left_count = sum(1 for val in data[feature] if float(val) <= threshold)
        right_count = len(data[feature]) - left_count
        total = len(data[feature])
        prob_left = left_count / total
        prob_right = right_count / total

        split_info = 0.0
        if prob_left > 0:
            split_info -= prob_left * log2(prob_left)
        if prob_right > 0:
            split_info -= prob_right * log2(prob_right)

        if split_info == 0:
            return 0, None
        return ig / split_info, threshold
    else:
        total_entropy = get_entropy(data[target])
        unique_values = set(data[feature])
        expected_entropy = 0.0
        split_info = 0.0

        for value in unique_values:
            subset_labels = [data[target][i] for i in range(len(data[target])) if data[feature][i] == value]
            subset_entropy = get_entropy(subset_labels)
            prob = len(subset_labels) / len(data[target])
            expected_entropy += prob * subset_entropy
            split_info -= prob * log2(prob) if prob > 0 else 0

        ig = total_entropy - expected_entropy
        if split_info == 0:
            return 0, None
        return ig / split_info, None

def build_tree(data, target, features, numerical_features, depth=0, max_depth=10):
    if depth >= max_depth or len(set(data[target])) == 1:
        return DecisionNode(value=Counter(data[target]).most_common(1)[0][0])
    
    if not features:
        return DecisionNode(value=Counter(data[target]).most_common(1)[0][0])

    best_feature = None
    best_threshold = None
    max_gr = -1

    available_features = [f for f in features if f not in numerical_features] + \
                        [f for f in features if f in numerical_features]
    
    for feature in available_features:
        gr, threshold = gain_ratio(data, feature, target, numerical_features)
        if feature in numerical_features and threshold is None:
            continue
        if gr > max_gr:
            max_gr, best_feature, best_threshold = gr, feature, threshold

    if best_feature is None:
        return DecisionNode(value=Counter(data[target]).most_common(1)[0][0])

    remaining_features = [f for f in features if f != best_feature]
    
    if best_feature in numerical_features:
        left_data = {k: [] for k in data}
        right_data = {k: [] for k in data}
        
        for i in range(len(data[target])):
            val = float(data[best_feature][i])
            target_dict = left_data if val <= best_threshold else right_data
            for k in data:
                target_dict[k].append(data[k][i])
        
        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            branches={
                'left': build_tree(left_data, target, remaining_features, numerical_features, depth+1, max_depth),
                'right': build_tree(right_data, target, remaining_features, numerical_features, depth+1, max_depth)
            }
        )
    else:
        branches = {}
        for value in set(data[best_feature]):
            subset = {k: [v for i, v in enumerate(data[k]) if data[best_feature][i] == value] 
                     for k in data}
            branches[value] = build_tree(subset, target, remaining_features, numerical_features, depth+1, max_depth)
        
        return DecisionNode(
            feature=best_feature,
            branches=branches
        )

def classify(node, sample):
    if node.value is not None:
        return node.value
    
    if node.threshold is not None:
        #added numerical split thingies
        sample_value = float(sample[node.feature])
        if sample_value <= node.threshold:
            return classify(node.branches['left'], sample)
        else:
            return classify(node.branches['right'], sample)
    else:
        #old categorical handling
        value = sample.get(node.feature)
        if value in node.branches:
            return classify(node.branches[value], sample)
        else:
            return Counter([classify(child, sample) for child in node.branches.values()]).most_common(1)[0][0]

def load_csv_to_dict(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    features = data.pop(0)
    target = features.pop()
    data_dict = {k: [] for k in features + [target]}
    for row in data:
        for i, k in enumerate(features + [target]):
            data_dict[k].append(row[i])

    numerical_features = []
    for feature in features:
        try:
            [float(x) for x in data_dict[feature]]
            numerical_features.append(feature)
        except ValueError:
            continue

    return data_dict, features, target, numerical_features

K_NEIGHBORS = 100
FOREST_SIZE = 75
MAX_DEPTH = 10 
SAMPLE_SIZE = 0.1

train_data, train_features, target, numerical_features = load_csv_to_dict('train.csv')
test_data, _, _, _ = load_csv_to_dict('test.csv')

train_vectors = []
for i in range(len(train_data[target])):
    train_vectors.append([float(train_data[feature][i]) for feature in numerical_features])

knn = NearestNeighbors(n_neighbors=K_NEIGHBORS)
knn.fit(train_vectors)

num_attrs = int(math.log2(len(train_features)))
forest = []
for _ in range(FOREST_SIZE):
    print(_)
    sample_size = int(SAMPLE_SIZE * len(train_data[target]))
    sample_indices = np.random.choice(len(train_data[target]), sample_size, replace=True)
    sampled_data = {k: [train_data[k][i] for i in sample_indices] for k in train_data}
    
    sampled_features = list(np.random.choice(train_features, num_attrs, replace=False))
    
    tree = build_tree(sampled_data, target, sampled_features, numerical_features, max_depth=MAX_DEPTH)
    forest.append((tree, sampled_features))

y_true = []
y_pred = []
confusion = defaultdict(lambda: defaultdict(int))

for i in range(len(test_data[target])):
    print(i)
    test_sample = {k: test_data[k][i] for k in test_data}
    true_label = test_data[target][i]
    y_true.append(true_label)
    
    test_vec = [float(test_sample[f]) for f in numerical_features]
    _, neighbor_indices = knn.kneighbors([test_vec])
    neighbor_indices = neighbor_indices[0]
    
    local_data = {k: [] for k in train_data}
    for idx in neighbor_indices:
        for k in train_data:
            local_data[k].append(train_data[k][idx])
    
    feature_weights = {}
    for feature in train_features:
        #gr, _ = gain_ratio(local_data, feature, target, numerical_features)
        gr, _ = information_gain_numerical(local_data, feature, target)
        feature_weights[feature] = gr if gr is not None else 0.0
    
    total_weight = sum(feature_weights.values())
    if total_weight == 0:
        normalized = {f: 1/len(train_features) for f in train_features}
    else:
        normalized = {f: w/total_weight for f, w in feature_weights.items()}
    
    tree_weights = []
    for tree, features in forest:
        tree_weight = sum(normalized[f] for f in features)
        tree_weights.append(tree_weight)
    
    votes = defaultdict(float)
    for (tree, _), weight in zip(forest, tree_weights):
        pred = classify(tree, test_sample)
        votes[pred] += weight
    
    pred_label = max(votes, key=votes.get) if votes else '0'
    y_pred.append(pred_label)
    confusion[true_label][pred_label] += 1

accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')


print("\nConfusion Matrix:")
classes = sorted(set(y_true))
print(" " * 12 + "".join([f"{c:>8}" for c in classes]))
for true in classes:
    row = [f"{confusion[true].get(pred, 0):8}" for pred in classes]
    print(f"{true:<12}{''.join(row)}")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
if '0' in classes and '1' in classes:
    tn = confusion['0'].get('0', 0)
    fp = confusion['0'].get('1', 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f'Specificity: {specificity:.4f}')