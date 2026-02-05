import json
import numpy as np
from rouge_score import rouge_scorer
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

###############################################################
# Load JSON
###############################################################
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


###############################################################
# Normalize contradictions: extract evidence_texts and intensity
###############################################################
def extract_contradictions(data):
    out = {}
    for pid, pdata in data.items():
        out[pid] = []
        contradictions = pdata.get("analysis", [])
        for idx, c in enumerate(contradictions):
            ev = c.get("evidence", [])
            evidence_texts = []

            if isinstance(ev, list):
                evidence_texts = [x.strip() for x in ev if str(x).strip()]
            elif isinstance(ev, dict):
                evidence_texts = [v.strip() for v in ev.values() if v.strip()]

            # Extract intensity - handle different formats
            intensity = None
            if "intensity" in c:
                intensity_val = c["intensity"]
                if isinstance(intensity_val, dict) and "score" in intensity_val:
                    intensity = intensity_val["score"]
                elif isinstance(intensity_val, (int, float)):
                    intensity = intensity_val
            
            out[pid].append({
                "original_id": idx,
                "evidence_texts": evidence_texts,
                "intensity": intensity
            })

    return out


###############################################################
# Compute ROUGE scores for one contradiction pair
###############################################################
def rouge_pair(gt_texts, ex_texts, scorer):
    if not gt_texts or not ex_texts:
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}

    cost = np.zeros((len(gt_texts), len(ex_texts)))
    cache = {}

    for i, g in enumerate(gt_texts):
        for j, e in enumerate(ex_texts):
            s = scorer.score(g, e)
            cache[(i, j)] = {
                "rouge1": s["rouge1"].fmeasure,
                "rouge2": s["rouge2"].fmeasure,
                "rougeL": s["rougeL"].fmeasure,
            }
            cost[i, j] = -s["rougeL"].fmeasure

    # Hungarian matching on sentence level
    ri, ci = linear_sum_assignment(cost)

    r1 = []; r2 = []; rL = []
    for a, b in zip(ri, ci):
        sc = cache[(a, b)]
        r1.append(sc["rouge1"])
        r2.append(sc["rouge2"])
        rL.append(sc["rougeL"])

    return {
        "rouge1": np.mean(r1) if r1 else 0,
        "rouge2": np.mean(r2) if r2 else 0,
        "rougeL": np.mean(rL) if rL else 0,
    }


###############################################################
# GLOBAL HUNGARIAN MATCHING (aspect-independent)
###############################################################
def global_match(gt_list, ex_list, scorer):
    if len(gt_list) == 0:
        return [], [], ex_list
    if len(ex_list) == 0:
        return [], gt_list, []

    n_gt = len(gt_list)
    n_ex = len(ex_list)

    cost = np.zeros((n_gt, n_ex))
    score_cache = {}

    # compute pairwise scores
    for i, g in enumerate(gt_list):
        for j, e in enumerate(ex_list):
            s = rouge_pair(g["evidence_texts"], e["evidence_texts"], scorer)
            score_cache[(i, j)] = s
            cost[i, j] = -s["rougeL"]

    # Hungarian algorithm on full grid
    ri, ci = linear_sum_assignment(cost)

    matches = []
    matched_gt_idx = set()
    matched_ex_idx = set()

    for a, b in zip(ri, ci):
        matches.append({
            "gt": gt_list[a],
            "ex": ex_list[b],
            "scores": score_cache[(a, b)]
        })
        matched_gt_idx.add(a)
        matched_ex_idx.add(b)

    # unmatched
    unmatched_gt = [gt_list[i] for i in range(n_gt) if i not in matched_gt_idx]
    unmatched_ex = [ex_list[j] for j in range(n_ex) if j not in matched_ex_idx]

    return matches, unmatched_gt, unmatched_ex


###############################################################
# FN / FP
###############################################################
def fn_fp(unmatched_gt, unmatched_ex, total_gt, total_ex):
    FN = len(unmatched_gt)
    FP = len(unmatched_ex)
    return {
        "FN": FN,
        "FP": FP,
        "FN_rate": FN / total_gt if total_gt else 0,
        "FP_rate": FP / total_ex if total_ex else 0,
    }


###############################################################
# Calculate Intensity Metrics
###############################################################
def calculate_intensity_metrics(matches):
    """
    Calculate Cohen's Kappa, Spearman correlation, and Kendall's Tau
    for intensity scores of matched pairs
    """
    gt_intensities = []
    pred_intensities = []
    
    for m in matches:
        gt_int = m["gt"].get("intensity")
        ex_int = m["ex"].get("intensity")
        
        # Only include pairs where both have intensity values
        if gt_int is not None and ex_int is not None:
            gt_intensities.append(gt_int)
            pred_intensities.append(ex_int)
    
    if len(gt_intensities) < 2:
        return {
            "n_pairs_with_intensity": len(gt_intensities),
            "cohen_kappa": None,
            "spearman_correlation": None,
            "spearman_pvalue": None,
            "kendall_tau": None,
            "kendall_pvalue": None,
            "note": "Insufficient pairs with intensity values (need at least 2)"
        }
    
    # Calculate metrics
    try:
        kappa = cohen_kappa_score(gt_intensities, pred_intensities)
    except Exception as e:
        kappa = None
        print(f"Warning: Could not calculate Cohen's Kappa: {e}")
    
    try:
        spearman_corr, spearman_p = spearmanr(gt_intensities, pred_intensities)
    except Exception as e:
        spearman_corr, spearman_p = None, None
        print(f"Warning: Could not calculate Spearman correlation: {e}")
    
    try:
        kendall_corr, kendall_p = kendalltau(gt_intensities, pred_intensities)
    except Exception as e:
        kendall_corr, kendall_p = None, None
        print(f"Warning: Could not calculate Kendall's Tau: {e}")
    
    return {
        "n_pairs_with_intensity": len(gt_intensities),
        "cohen_kappa": kappa,
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_p,
        "kendall_tau": kendall_corr,
        "kendall_pvalue": kendall_p,
        "ground_truth_intensities": gt_intensities,
        "predicted_intensities": pred_intensities
    }


###############################################################
# MAIN evaluation
###############################################################
def evaluate(gt_file, pred_file, rouge_threshold=None):

    gt_raw = load_json(gt_file)
    pred_raw = load_json(pred_file)

    gt = extract_contradictions(gt_raw)
    pred = extract_contradictions(pred_raw)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    all_matches = []
    all_unmatched_gt = []
    all_unmatched_ex = []

    for pid in gt:
        if pid not in pred:
            print(f"WARNING: No predictions for {pid}")
            continue

        gt_list = gt[pid]
        ex_list = pred[pid]

        # global aspect-independent 1-1 matching
        matches, un_gt, un_ex = global_match(gt_list, ex_list, scorer)

        passed = []
        failed_gt = []
        failed_ex = []

        # thresholding if any
        for m in matches:
            if rouge_threshold is not None and m["scores"]["rougeL"] < rouge_threshold:
                failed_gt.append(m["gt"])
                failed_ex.append(m["ex"])
            else:
                passed.append(m)

        all_matches.extend(passed)
        all_unmatched_gt.extend(un_gt + failed_gt)
        all_unmatched_ex.extend(un_ex + failed_ex)

    total_gt = sum(len(v) for v in gt.values())
    total_ex = sum(len(v) for v in pred.values())

    metrics = fn_fp(all_unmatched_gt, all_unmatched_ex, total_gt, total_ex)

    avg_rouge = {
        "rouge1": np.mean([m["scores"]["rouge1"] for m in all_matches]) if all_matches else 0,
        "rouge2": np.mean([m["scores"]["rouge2"] for m in all_matches]) if all_matches else 0,
        "rougeL": np.mean([m["scores"]["rougeL"] for m in all_matches]) if all_matches else 0,
    }

    # Calculate intensity metrics for matched pairs that passed threshold
    intensity_metrics = calculate_intensity_metrics(all_matches)

    return {
        "avg_rouge": avg_rouge,
        "FN_FP_metrics": metrics,
        "intensity_metrics": intensity_metrics,
        "total_matches": len(all_matches),
        "total_gt": total_gt,
        "total_pred": total_ex
    }


###############################################################
# Run
###############################################################
if __name__ == "__main__":
    GT = "ground_data.json"
    PRED = "qwen_trained_reasoning.json"

    results = evaluate(GT, PRED, 0.30)

    print("\n=== FINAL SCORES (Aspect-independent Hungarian Matching) ===")
    print("Average ROUGE =", results["avg_rouge"])
    print("FN/FP Metrics =", results["FN_FP_metrics"])
    print("\n=== INTENSITY METRICS ===")
    int_metrics = results["intensity_metrics"]
    print(f"Pairs with intensity values: {int_metrics['n_pairs_with_intensity']}")
    if int_metrics['cohen_kappa'] is not None:
        print(f"Cohen's Kappa: {int_metrics['cohen_kappa']:.4f}")
    if int_metrics['spearman_correlation'] is not None:
        print(f"Spearman Correlation: {int_metrics['spearman_correlation']:.4f} (p={int_metrics['spearman_pvalue']:.4f})")
    if int_metrics['kendall_tau'] is not None:
        print(f"Kendall's Tau: {int_metrics['kendall_tau']:.4f} (p={int_metrics['kendall_pvalue']:.4f})")
    
    print(f"\nMatches = {results['total_matches']}")
    print(f"GT contradictions = {results['total_gt']}")
    print(f"Pred contradictions = {results['total_pred']}")