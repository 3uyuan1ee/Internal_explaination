"""
Intervention experiments 1-4 for Concept Bottleneck Model.

Experiments:
  1. Error attribution — decompose errors into concept vs label predictor
  2. Strategy comparison — 6 intervention strategies across k values
  3. Minimal intervention — greedy search for minimum concepts to correct
  4. Noisy experts — robustness to noise types and budget constraints

Usage:
    python scripts/run_interventions.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cbm.config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR, TORCH_LOAD_KWARGS,
    INTERVENTION_K_VALUES, RANDOM_TRIALS, NOISE_LEVELS, NOISE_BUDGETS,
    NOISE_TYPES,
)
from cbm.dataset import CUBDataset, get_transforms, select_50_classes
from cbm.models.concept_predictor import ConceptPredictor
from cbm.models.label_predictor import LabelPredictor
from cbm.utils import set_seed


# ---------------------------------------------------------------------------
# Helper: load trained models
# ---------------------------------------------------------------------------
def load_models():
    """Load concept and label predictors from checkpoints.

    Returns:
        concept_model: ConceptPredictor on DEVICE (eval mode)
        label_model: LabelPredictor on DEVICE (eval mode)
        ckpt_concept: dict with saved concept predictor state
        ckpt_label: dict with saved label predictor state
    """
    ckpt_concept = torch.load(
        CHECKPOINT_DIR / "concept_predictor_best.pth",
        map_location=DEVICE, **TORCH_LOAD_KWARGS,
    )
    ckpt_label = torch.load(
        CHECKPOINT_DIR / "label_predictor_best.pth",
        map_location=DEVICE, **TORCH_LOAD_KWARGS,
    )

    num_concepts = ckpt_concept["num_concepts"]
    num_classes = ckpt_label["num_classes"]

    concept_model = ConceptPredictor(num_concepts).to(DEVICE)
    concept_model.load_state_dict(ckpt_concept["model_state_dict"])
    concept_model.eval()

    label_model = LabelPredictor(num_concepts, num_classes).to(DEVICE)
    label_model.load_state_dict(ckpt_label["model_state_dict"])
    label_model.eval()

    print(f"[Models] ConceptPredictor loaded (acc={ckpt_concept['concept_acc']:.2f}%)")
    print(f"[Models] LabelPredictor loaded (acc={ckpt_label['label_acc']:.2f}%)")
    print(f"[Models] {num_concepts} concepts, {num_classes} classes")

    return concept_model, label_model, ckpt_concept, ckpt_label


# ---------------------------------------------------------------------------
# Helper: collect all predictions on the test set
# ---------------------------------------------------------------------------
def collect_predictions(concept_model, label_model, dataset):
    """Run both models on the test set and collect predictions.

    Args:
        concept_model: ConceptPredictor in eval mode
        label_model: LabelPredictor in eval mode
        dataset: CUBDataset (test split, no augmentation)

    Returns:
        c_hat:  [N, n_concepts] predicted concept probabilities
        c_gt:   [N, n_concepts] ground-truth binary concepts
        y:      [N] true class labels
        yhat_cbm: [N] CBM predicted classes (using c_hat)
        yhat_oracle: [N] oracle predicted classes (using c_gt)
        W:      [n_classes, n_concepts] label predictor weight matrix
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2,
                        pin_memory=torch.cuda.is_available())
    n_samples = len(dataset)
    n_concepts = dataset.num_concepts
    n_classes = dataset.num_classes

    c_hat_list, c_gt_list, y_list = [], [], []

    with torch.no_grad():
        for images, concepts, labels, _ in tqdm(loader, desc="Collecting predictions"):
            images = images.to(DEVICE)
            probs, _ = concept_model(images)
            c_hat_list.append(probs.cpu())
            c_gt_list.append(concepts)
            y_list.append(labels)

    c_hat = torch.cat(c_hat_list)   # [N, n_concepts]
    c_gt = torch.cat(c_gt_list)     # [N, n_concepts]
    y = torch.cat(y_list)           # [N]

    # CBM predictions: c_hat -> label predictor
    with torch.no_grad():
        logits_cbm = label_model(c_hat.to(DEVICE))
        yhat_cbm = logits_cbm.argmax(dim=1).cpu()

    # Oracle predictions: c_gt -> label predictor
    with torch.no_grad():
        logits_oracle = label_model(c_gt.to(DEVICE))
        yhat_oracle = logits_oracle.argmax(dim=1).cpu()

    # Weight matrix
    W = label_model.weight_matrix.cpu()  # [n_classes, n_concepts]

    acc_cbm = (yhat_cbm == y).float().mean().item() * 100
    acc_oracle = (yhat_oracle == y).float().mean().item() * 100
    concept_err_rate = (c_hat.round() != c_gt).float().mean().item() * 100

    print(f"[Predictions] N={n_samples}, CBM acc={acc_cbm:.2f}%, "
          f"Oracle acc={acc_oracle:.2f}%, Concept error rate={concept_err_rate:.2f}%")

    return c_hat, c_gt, y, yhat_cbm, yhat_oracle, W


# ---------------------------------------------------------------------------
# Helper: predict with intervention
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_with_intervention(c_hat_single, c_gt_single, indices, label_model):
    """Replace selected concept predictions with ground truth and classify.

    Args:
        c_hat_single: [n_concepts] predicted concept probs
        c_gt_single: [n_concepts] ground truth concepts
        indices: list/tensor of concept indices to correct
        label_model: LabelPredictor on DEVICE

    Returns:
        logits: [n_classes] class logits after intervention
        pred: int, predicted class after intervention
    """
    c_int = c_hat_single.clone()
    if len(indices) > 0:
        idx = torch.tensor(indices, dtype=torch.long)
        c_int[idx] = c_gt_single[idx]
    logits = label_model(c_int.unsqueeze(0).to(DEVICE))
    return logits.squeeze(0), logits.argmax(dim=1).item()


# ---------------------------------------------------------------------------
# Experiment 1: Error Attribution
# ---------------------------------------------------------------------------
def experiment1_error_attribution(c_hat, c_gt, y, yhat_cbm, yhat_oracle, W):
    """Decompose classification errors into concept-attributable vs label predictor.

    For each misclassified sample:
    - If oracle (GT concepts) also wrong: label predictor error
    - If oracle correct but CBM wrong: concept predictor error (attributable)

    Also computes per-concept error contribution: for concept-attributable errors,
    which wrong concepts contributed most (ranked by |W[true_class, i]|).
    """
    n_samples, n_concepts = c_hat.shape
    n_classes = W.shape[0]

    wrong_mask = yhat_cbm != y
    oracle_correct_mask = yhat_oracle == y
    n_wrong = wrong_mask.sum().item()

    # Error types
    concept_error_mask = wrong_mask & oracle_correct_mask  # concept predictor to blame
    label_error_mask = wrong_mask & ~oracle_correct_mask    # label predictor to blame

    n_concept_err = concept_error_mask.sum().item()
    n_label_err = label_error_mask.sum().item()

    # Per-concept contribution to concept-attributable errors
    c_hat_binary = c_hat.round()
    wrong_concept_mask = c_hat_binary != c_gt  # [N, n_concepts]

    # For each concept-attributable error, compute importance of wrong concepts
    # Impact = W[y_true, ci] * (c_hat[ci] - c_gt[ci]): negative means harmful
    concept_importance = torch.zeros(n_concepts)
    concept_error_count = torch.zeros(n_concepts)

    for i in range(n_samples):
        if not concept_error_mask[i]:
            continue
        true_class = y[i].item()
        wrong_cpts = wrong_concept_mask[i].nonzero(as_tuple=True)[0]
        for ci in wrong_cpts:
            ci_val = ci.item()
            impact = W[true_class, ci_val].item() * (
                c_hat_binary[i, ci_val].item() - c_gt[i, ci_val].item()
            )
            if impact < 0:
                concept_importance[ci_val] += abs(impact)
            concept_error_count[ci_val] += 1

    # Normalize
    total_importance = concept_importance.sum()
    if total_importance > 0:
        concept_importance = concept_importance / total_importance

    results = {
        "n_wrong": n_wrong,
        "n_concept_error": n_concept_err,
        "n_label_error": n_label_err,
        "frac_concept_error": n_concept_err / max(n_wrong, 1),
        "frac_label_error": n_label_err / max(n_wrong, 1),
        "concept_importance": concept_importance.numpy(),
        "concept_error_count": concept_error_count.numpy(),
    }

    print(f"\n[Exp1] Error Attribution:")
    print(f"  Total errors: {n_wrong}/{n_samples} "
          f"({n_wrong/n_samples*100:.1f}%)")
    print(f"  Concept-attributable: {n_concept_err} "
          f"({n_concept_err/max(n_wrong,1)*100:.1f}% of errors)")
    print(f"  Label predictor: {n_label_err} "
          f"({n_label_err/max(n_wrong,1)*100:.1f}% of errors)")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Strategy Comparison
# ---------------------------------------------------------------------------
def experiment2_strategies(c_hat, c_gt, y, W, label_model):
    """Compare 6 intervention strategies across k values.

    Strategies:
      1. Random: random.choice k concepts, averaged over RANDOM_TRIALS
      2. Uncertainty: top-k concepts with smallest |p - 0.5|
      3. Importance-Weighted: top-k concepts by |W[predicted_class, i] * p[i]|
      4. Greedy Oracle: iterative greedy correction maximizing P(y_true)
      5. Oracle-Targeted: correct only wrong concepts, ranked by |W[true_class, i]|
         (oracle: uses GT error detection + true class label)
      6. Error-Targeted: correct only wrong concepts, ranked by |W[predicted_class, i]|
         (practical: uses GT error detection + predicted class label)
    """
    n_samples, n_concepts = c_hat.shape
    n_classes = W.shape[0]

    strategies = ["random", "uncertainty", "importance", "greedy_oracle",
                  "oracle_targeted", "error_targeted"]
    k_values = INTERVENTION_K_VALUES
    results = {s: {k: [] for k in k_values} for s in strategies}

    # Pre-compute per-sample auxiliary info
    uncertainty_scores = (c_hat - 0.5).abs()  # [N, n_concepts] — lower = more uncertain

    with torch.no_grad():
        all_logits = label_model(c_hat.to(DEVICE))
        yhat = all_logits.argmax(dim=1).cpu()  # [N]

    # Importance weights: |W[predicted_class, i] * p[i]|
    importance_weights = torch.zeros_like(c_hat)
    for i in range(n_samples):
        pc = yhat[i].item()
        importance_weights[i] = W[pc].abs() * c_hat[i]

    # Error mask
    c_hat_binary = c_hat.round()
    wrong_concept_mask = c_hat_binary != c_gt  # [N, n_concepts]

    print(f"\n[Exp2] Strategy comparison over {len(k_values)} k-values, {n_samples} samples")

    for sample_idx in tqdm(range(n_samples), desc="Exp2 samples"):
        c_h = c_hat[sample_idx]   # [n_concepts]
        c_g = c_gt[sample_idx]    # [n_concepts]
        true_y = y[sample_idx].item()

        # Precompute sorted indices for each strategy
        # Uncertainty: ascending |p-0.5| (most uncertain first)
        unc_order = uncertainty_scores[sample_idx].argsort()

        # Importance: descending |W * p|
        imp_order = importance_weights[sample_idx].argsort(descending=True)

        # Oracle-Targeted: wrong concepts ranked by |W[true_class, i]| descending
        wrong_cpts = wrong_concept_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        oracle_err_weights = W[true_y].abs()
        oracle_err_order = sorted(wrong_cpts,
                                  key=lambda ci: oracle_err_weights[ci].item(),
                                  reverse=True)

        # Error-Targeted (practical): wrong concepts ranked by |W[predicted_class, i]| descending
        practical_err_weights = W[yhat[sample_idx].item()].abs()
        practical_err_order = sorted(wrong_cpts,
                                     key=lambda ci: practical_err_weights[ci].item(),
                                     reverse=True)

        # Greedy oracle: will be built iteratively
        greedy_order = []

        for k in k_values:
            effective_k = n_concepts if k == -1 else k

            # --- Random (average over RANDOM_TRIALS) ---
            random_accs = []
            for _ in range(RANDOM_TRIALS):
                if effective_k >= n_concepts:
                    indices = list(range(n_concepts))
                else:
                    indices = random.sample(range(n_concepts), effective_k)
                _, pred = predict_with_intervention(c_h, c_g, indices, label_model)
                random_accs.append(1.0 if pred == true_y else 0.0)
            results["random"][k].append(np.mean(random_accs))

            # --- Uncertainty ---
            unc_indices = unc_order[:effective_k].tolist()
            _, pred = predict_with_intervention(c_h, c_g, unc_indices, label_model)
            results["uncertainty"][k].append(1.0 if pred == true_y else 0.0)

            # --- Importance-Weighted ---
            imp_indices = imp_order[:effective_k].tolist()
            _, pred = predict_with_intervention(c_h, c_g, imp_indices, label_model)
            results["importance"][k].append(1.0 if pred == true_y else 0.0)

            # --- Greedy Oracle (iterative) ---
            # Build greedy ordering incrementally up to effective_k
            if len(greedy_order) < effective_k:
                _extend_greedy_order(
                    greedy_order, c_h, c_g, true_y, effective_k,
                    label_model, n_concepts,
                )
            greedy_indices = greedy_order[:effective_k]
            _, pred = predict_with_intervention(c_h, c_g, greedy_indices, label_model)
            results["greedy_oracle"][k].append(1.0 if pred == true_y else 0.0)

            # --- Oracle-Targeted (uses true_y for ranking) ---
            if len(oracle_err_order) == 0:
                results["oracle_targeted"][k].append(
                    1.0 if yhat[sample_idx].item() == true_y else 0.0)
            else:
                err_indices = oracle_err_order[:effective_k]
                _, pred = predict_with_intervention(c_h, c_g, err_indices, label_model)
                results["oracle_targeted"][k].append(1.0 if pred == true_y else 0.0)

            # --- Error-Targeted (practical: uses predicted class for ranking) ---
            if len(practical_err_order) == 0:
                results["error_targeted"][k].append(
                    1.0 if yhat[sample_idx].item() == true_y else 0.0)
            else:
                err_indices = practical_err_order[:effective_k]
                _, pred = predict_with_intervention(c_h, c_g, err_indices, label_model)
                results["error_targeted"][k].append(1.0 if pred == true_y else 0.0)

    # Aggregate: compute accuracy per strategy per k
    summary = {}
    for s in strategies:
        summary[s] = {}
        for k in k_values:
            acc = np.mean(results[s][k]) * 100
            summary[s][k] = acc
            # Store raw per-sample arrays as numpy
            results[s][k] = np.array(results[s][k])

    print(f"\n[Exp2] Accuracy by strategy and k:")
    header = f"{'k':>6s}"
    for s in strategies:
        header += f"  {s:>16s}"
    print(header)
    for k in k_values:
        row = f"{k:>6d}" if k != -1 else f"{'all':>6s}"
        for s in strategies:
            row += f"  {summary[s][k]:>15.2f}%"
        print(row)

    return {"raw": results, "summary": summary}


def _extend_greedy_order(current_order, c_h, c_g, true_y, target_k,
                         label_model, n_concepts):
    """Extend greedy oracle ordering up to target_k concepts.

    At each step, batch-evaluate all remaining candidates and pick the one
    that maximally increases P(y_true).
    """
    corrected = set(current_order)

    with torch.no_grad():
        while len(corrected) < target_k:
            c_int = c_h.clone()
            for ci in corrected:
                c_int[ci] = c_g[ci]

            logits_curr = label_model(c_int.unsqueeze(0).to(DEVICE))
            prob_true_curr = F.softmax(logits_curr, dim=1)[0, true_y].item()

            # Batch-evaluate all remaining candidates in one forward pass
            remaining = [ci for ci in range(n_concepts) if ci not in corrected]
            n_rem = len(remaining)
            if n_rem == 0:
                break
            batch = c_int.unsqueeze(0).repeat(n_rem, 1)
            rem_idx = torch.tensor(remaining, dtype=torch.long)
            batch[range(n_rem), rem_idx] = c_g[rem_idx]
            logits_batch = label_model(batch.to(DEVICE))
            probs_batch = F.softmax(logits_batch, dim=1)[:, true_y]
            gains = probs_batch - prob_true_curr

            best_local = gains.argmax().item()
            best_ci = remaining[best_local]
            corrected.add(best_ci)
            current_order.append(best_ci)


# ---------------------------------------------------------------------------
# Experiment 3: Minimal Intervention
# ---------------------------------------------------------------------------
def experiment3_minimal(c_hat, c_gt, y, yhat_cbm, label_model):
    """Find minimum number of concepts to correct per sample via greedy search.

    For each currently-wrong sample, greedily correct concepts one by one,
    stopping as soon as the prediction matches y_true.
    """
    n_samples, n_concepts = c_hat.shape
    wrong_mask = yhat_cbm != y
    wrong_indices = wrong_mask.nonzero(as_tuple=True)[0].tolist()

    min_concepts_needed = []
    per_sample = {}  # {sample_idx: (min_k, corrected_indices)}

    print(f"\n[Exp3] Minimal intervention on {len(wrong_indices)} wrong samples")

    for sample_idx in tqdm(wrong_indices, desc="Exp3 greedy"):
        c_h = c_hat[sample_idx]
        c_g = c_gt[sample_idx]
        true_y = y[sample_idx].item()

        corrected = set()
        found = False

        with torch.no_grad():
            for step in range(n_concepts):
                c_int = c_h.clone()
                for ci in corrected:
                    c_int[ci] = c_g[ci]

                logits_curr = label_model(c_int.unsqueeze(0).to(DEVICE))
                pred_curr = logits_curr.argmax(dim=1).item()

                if pred_curr == true_y:
                    min_concepts_needed.append(len(corrected))
                    per_sample[sample_idx] = (len(corrected), list(corrected))
                    found = True
                    break

                # Batch-evaluate all remaining candidates
                prob_true_curr = F.softmax(logits_curr, dim=1)[0, true_y].item()
                remaining = [ci for ci in range(n_concepts) if ci not in corrected]
                n_rem = len(remaining)
                if n_rem == 0:
                    break
                batch = c_int.unsqueeze(0).repeat(n_rem, 1)
                rem_idx = torch.tensor(remaining, dtype=torch.long)
                batch[range(n_rem), rem_idx] = c_g[rem_idx]
                logits_batch = label_model(batch.to(DEVICE))
                probs_batch = F.softmax(logits_batch, dim=1)[:, true_y]
                gains = probs_batch - prob_true_curr

                best_local = gains.argmax().item()
                corrected.add(remaining[best_local])

            if not found:
                min_concepts_needed.append(n_concepts)
                per_sample[sample_idx] = (n_concepts, list(corrected))

    arr = np.array(min_concepts_needed)
    results = {
        "per_sample_k": arr,
        "per_sample_detail": {str(k): v for k, v in per_sample.items()},
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "histogram": np.bincount(arr).tolist(),
    }

    print(f"  Mean concepts needed: {arr.mean():.2f}")
    print(f"  Median: {np.median(arr):.1f}, Min: {arr.min()}, Max: {arr.max()}")
    print(f"  Histogram (k -> count): {dict(enumerate(results['histogram']))}")

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Noisy Experts
# ---------------------------------------------------------------------------
def experiment4_noisy(c_hat, c_gt, y, W, label_model, exp2_results=None):
    """Test noisy experts with two noise models and budget constraints.

    Method A: For each noise level × noise type, use budget=10 with
              Random/Uncertainty/Importance.
              - Random noise: with prob=noise_level, give random {0,1} value.
              - Adversarial noise: with prob=noise_level, flip to opposite value.
    Method B: Noise level × budget grid for Uncertainty strategy.
              Tests interaction between noise and expert effort.
    """
    n_samples, n_concepts = c_hat.shape
    n_classes = W.shape[0]

    # Pre-compute auxiliary info
    uncertainty_scores = (c_hat - 0.5).abs()
    with torch.no_grad():
        all_logits = label_model(c_hat.to(DEVICE))
        yhat = all_logits.argmax(dim=1).cpu()
    importance_weights = torch.zeros_like(c_hat)
    for i in range(n_samples):
        pc = yhat[i].item()
        importance_weights[i] = W[pc].abs() * c_hat[i]

    strategies_a = ["random", "uncertainty", "importance"]

    # --- Method A: Noise levels × noise types ---
    print(f"\n[Exp4-A] Noise robustness (budget=10, {len(NOISE_LEVELS)} noise levels, "
          f"{len(NOISE_TYPES)} noise types)")
    results_a = {nt: {s: {nl: [] for nl in NOISE_LEVELS}
                      for s in strategies_a}
                 for nt in NOISE_TYPES}

    for noise_type in NOISE_TYPES:
        for sample_idx in tqdm(range(n_samples),
                               desc=f"Exp4-A ({noise_type})"):
            c_h = c_hat[sample_idx]
            c_g = c_gt[sample_idx]
            true_y = y[sample_idx].item()

            unc_order = uncertainty_scores[sample_idx].argsort().tolist()
            imp_order = importance_weights[sample_idx].argsort(descending=True).tolist()

            for noise_level in NOISE_LEVELS:
                budget = 10

                for strategy in strategies_a:
                    if strategy == "random":
                        trial_accs = []
                        for _ in range(RANDOM_TRIALS):
                            indices = random.sample(range(n_concepts),
                                                    min(budget, n_concepts))
                            acc = _noisy_intervention(
                                c_h, c_g, indices, true_y, noise_level,
                                label_model, noise_type=noise_type,
                            )
                            trial_accs.append(acc)
                        results_a[noise_type][strategy][noise_level].append(
                            np.mean(trial_accs))
                    else:
                        order = unc_order if strategy == "uncertainty" else imp_order
                        indices = order[:budget]
                        acc = _noisy_intervention(
                            c_h, c_g, indices, true_y, noise_level,
                            label_model, noise_type=noise_type,
                        )
                        results_a[noise_type][strategy][noise_level].append(acc)

    # Aggregate A
    summary_a = {}
    for nt in NOISE_TYPES:
        summary_a[nt] = {}
        for s in strategies_a:
            summary_a[nt][s] = {}
            for nl in NOISE_LEVELS:
                acc = np.mean(results_a[nt][s][nl]) * 100
                summary_a[nt][s][nl] = acc
                results_a[nt][s][nl] = np.array(results_a[nt][s][nl])

        print(f"\n[Exp4-A] {nt} noise — Accuracy by noise level (budget=10):")
        header = f"{'noise':>8s}"
        for s in strategies_a:
            header += f"  {s:>16s}"
        print(header)
        for nl in NOISE_LEVELS:
            row = f"{nl:>8.2f}"
            for s in strategies_a:
                row += f"  {summary_a[nt][s][nl]:>15.2f}%"
            print(row)

    # --- Method B: Noise × Budget grid (Uncertainty strategy) ---
    print(f"\n[Exp4-B] Noise × Budget grid (Uncertainty strategy)")
    results_b = {nt: {nl: {b: [] for b in NOISE_BUDGETS}
                      for nl in NOISE_LEVELS}
                 for nt in NOISE_TYPES}

    for noise_type in NOISE_TYPES:
        for sample_idx in tqdm(range(n_samples),
                               desc=f"Exp4-B ({noise_type})"):
            c_h = c_hat[sample_idx]
            c_g = c_gt[sample_idx]
            true_y = y[sample_idx].item()
            unc_order = uncertainty_scores[sample_idx].argsort().tolist()

            for noise_level in NOISE_LEVELS:
                for budget in NOISE_BUDGETS:
                    indices = unc_order[:budget]
                    acc = _noisy_intervention(
                        c_h, c_g, indices, true_y, noise_level,
                        label_model, noise_type=noise_type,
                    )
                    results_b[noise_type][noise_level][budget].append(acc)

    # Aggregate B
    summary_b = {}
    for nt in NOISE_TYPES:
        summary_b[nt] = {}
        for nl in NOISE_LEVELS:
            summary_b[nt][nl] = {}
            for b in NOISE_BUDGETS:
                acc = np.mean(results_b[nt][nl][b]) * 100
                summary_b[nt][nl][b] = acc
                results_b[nt][nl][b] = np.array(results_b[nt][nl][b])

        print(f"\n[Exp4-B] {nt} noise — Accuracy by noise_level × budget:")
        header = f"{'nl/bud':>8s}"
        for b in NOISE_BUDGETS:
            header += f"  k={b:>13d}"
        print(header)
        for nl in NOISE_LEVELS:
            row = f"{nl:>8.2f}"
            for b in NOISE_BUDGETS:
                row += f"  {summary_b[nt][nl][b]:>15.2f}%"
            print(row)

    return {
        "noise": {"raw": results_a, "summary": summary_a},
        "budget": {"raw": results_b, "summary": summary_b},
    }


def _noisy_intervention(c_h, c_g, indices, true_y, noise_level, label_model,
                         noise_type="random"):
    """Apply noisy intervention: with prob=noise_level, introduce noise.

    Args:
        noise_type: "random" gives random {0,1} value;
                    "adversarial" flips to the opposite value (1 - GT).
    """
    c_int = c_h.clone()
    with torch.no_grad():
        for ci in indices:
            if random.random() < noise_level:
                if noise_type == "adversarial":
                    c_int[ci] = 1.0 - c_g[ci]  # adversarial flip
                else:
                    c_int[ci] = random.choice([0.0, 1.0])  # random noise
            else:
                c_int[ci] = c_g[ci]  # correct
        logits = label_model(c_int.unsqueeze(0).to(DEVICE))
        pred = logits.argmax(dim=1).item()
    return 1.0 if pred == true_y else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(42)

    print("=" * 70)
    print("Intervention Experiments 1-4")
    print("=" * 70)

    # 1. Load models
    concept_model, label_model, ckpt_concept, ckpt_label = load_models()

    # 2. Rebuild test dataset using saved metadata from concept checkpoint
    selected_classes = ckpt_concept["selected_classes"]
    valid_attr_indices = ckpt_concept["valid_attr_indices"]
    attr_names = ckpt_concept["attr_names"]

    dataset = CUBDataset(
        train=False,
        selected_classes=selected_classes,
        valid_attr_indices=valid_attr_indices,
        filtered_attr_names=attr_names,
    )
    n_concepts = dataset.num_concepts
    print(f"[Dataset] Test set: {len(dataset)} samples, {n_concepts} concepts")

    # 3. Collect predictions
    c_hat, c_gt, y, yhat_cbm, yhat_oracle, W = collect_predictions(
        concept_model, label_model, dataset,
    )

    # 4. Run experiments
    exp1 = experiment1_error_attribution(c_hat, c_gt, y, yhat_cbm, yhat_oracle, W)
    exp2 = experiment2_strategies(c_hat, c_gt, y, W, label_model)
    exp3 = experiment3_minimal(c_hat, c_gt, y, yhat_cbm, label_model)
    exp4 = experiment4_noisy(c_hat, c_gt, y, W, label_model, exp2_results=exp2)

    # 5. Save all results
    all_results = {
        "c_hat": c_hat.numpy(),
        "c_gt": c_gt.numpy(),
        "y": y.numpy(),
        "yhat_cbm": yhat_cbm.numpy(),
        "yhat_oracle": yhat_oracle.numpy(),
        "W": W.numpy(),
        "attr_names": dataset.attr_names,
        "class_names": dataset.class_names,
        "n_concepts": n_concepts,
        "exp1": exp1,
        "exp2": exp2,
        "exp3": exp3,
        "exp4": exp4,
    }

    save_path = RESULTS_DIR / "all_intervention_results.pth"
    torch.save(all_results, save_path)
    print(f"\n[Save] All results saved to {save_path}")

    # Print summary
    cbm_acc = (yhat_cbm == y).float().mean().item() * 100
    oracle_acc = (yhat_oracle == y).float().mean().item() * 100
    print(f"\n{'=' * 70}")
    print(f"Summary:")
    print(f"  CBM accuracy:     {cbm_acc:.2f}%")
    print(f"  Oracle accuracy:  {oracle_acc:.2f}%")
    print(f"  Room for improvement via intervention: {oracle_acc - cbm_acc:.2f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
