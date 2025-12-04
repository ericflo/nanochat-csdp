#!/usr/bin/env python3
"""
Process comprehensive evaluation results and merge with extracted data.
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

def load_comprehensive_eval():
    """Load the comprehensive evaluation results."""
    eval_files = list(DATA_DIR.glob("comprehensive_eval_*.json"))
    if not eval_files:
        print("No comprehensive evaluation files found")
        return None

    # Use the most recent one
    eval_file = sorted(eval_files)[-1]
    print(f"Loading: {eval_file}")

    with open(eval_file) as f:
        return json.load(f)

def process_results(data):
    """Process results into a structured format."""
    results = {
        "extended_ood": {},
        "category_breakdown": {},
        "stage_progression": {},
        "csdp_metrics_new": {},
    }

    for r in data["results"]:
        curr = r["curriculum"]
        stage = r["stage"]

        # Initialize curriculum dict if needed
        if curr not in results["extended_ood"]:
            results["extended_ood"][curr] = {}
            results["category_breakdown"][curr] = {}
            results["stage_progression"][curr] = {}
            results["csdp_metrics_new"][curr] = {}

        # Store extended OOD accuracy by stage
        results["extended_ood"][curr][stage] = r["ExtendedOOD"]["accuracy"]

        # Store category breakdown (SFT only)
        if stage == "sft":
            results["category_breakdown"][curr] = r["ExtendedOOD"].get("category_scores", {})
            results["csdp_metrics_new"][curr] = {
                "ExtendedOOD": r["ExtendedOOD"]["accuracy"],
                "SelfKnowledge": r["SelfKnowledge"]["accuracy"],
                "Calibration": r["Calibration"]["accuracy"],
                "OODSelfKnowledge": r["OODSelfKnowledge"]["accuracy"],
                "SocialEngineering": r["SocialEngineering"]["accuracy"],
                "ToneLeakage": r["ToneLeakage"]["accuracy"],
            }

        # Store stage progression
        results["stage_progression"][curr][stage] = {
            "ExtendedOOD": r["ExtendedOOD"]["accuracy"],
            "SelfKnowledge": r["SelfKnowledge"]["accuracy"],
            "Calibration": r["Calibration"]["accuracy"],
        }

    return results

def merge_with_extracted_data(new_results):
    """Merge new results with existing extracted data."""
    extracted_path = SCRIPT_DIR / "extracted_data.json"

    with open(extracted_path) as f:
        extracted = json.load(f)

    # Add new data
    extracted["comprehensive_eval"] = new_results

    # Save merged data
    output_path = SCRIPT_DIR / "extracted_data_with_comprehensive.json"
    with open(output_path, "w") as f:
        json.dump(extracted, f, indent=2)

    print(f"Saved merged data to: {output_path}")
    return extracted

def print_summary(results):
    """Print a summary of key findings."""
    print("\n" + "="*60)
    print("KEY FINDINGS FROM EXTENDED OOD EVALUATION (128 probes)")
    print("="*60)

    print("\n--- SFT Stage Results ---")
    curricula = ["aria", "sage", "nova", "heart", "bare"]

    # Extended OOD
    print("\nExtended OOD Accuracy:")
    for curr in curricula:
        if curr in results["extended_ood"]:
            score = results["extended_ood"][curr].get("sft", 0)
            print(f"  {curr.upper()}: {score:.3f}")

    # Category breakdown
    print("\nCategory Leaders (SFT):")
    categories = ["self_model", "calibration", "adversarial", "metacognition", "philosophical"]
    for cat in categories:
        best_curr = None
        best_score = 0
        for curr in curricula:
            if curr in results["category_breakdown"]:
                score = results["category_breakdown"][curr].get(cat, 0)
                if score > best_score:
                    best_score = score
                    best_curr = curr
        if best_curr:
            print(f"  {cat}: {best_curr.upper()} ({best_score:.3f})")

    # Key insights
    print("\n--- Key Insights ---")

    # SAGE's calibration advantage
    sage_cal = results["category_breakdown"].get("sage", {}).get("calibration", 0)
    aria_cal = results["category_breakdown"].get("aria", {}).get("calibration", 0)
    print(f"1. SAGE leads calibration: {sage_cal:.3f} vs ARIA {aria_cal:.3f}")

    # ARIA's self-model advantage
    aria_self = results["category_breakdown"].get("aria", {}).get("self_model", 0)
    sage_self = results["category_breakdown"].get("sage", {}).get("self_model", 0)
    print(f"2. ARIA leads self-model: {aria_self:.3f} vs SAGE {sage_self:.3f}")

    # SAGE's adversarial vulnerability
    sage_adv = results["category_breakdown"].get("sage", {}).get("adversarial", 0)
    aria_adv = results["category_breakdown"].get("aria", {}).get("adversarial", 0)
    print(f"3. SAGE vulnerable to adversarial: {sage_adv:.3f} vs ARIA {aria_adv:.3f}")

    # HEART underperformance
    heart_ext = results["extended_ood"].get("heart", {}).get("sft", 0)
    bare_ext = results["extended_ood"].get("bare", {}).get("sft", 0)
    print(f"4. HEART underperforms on ExtendedOOD: {heart_ext:.3f} vs BARE {bare_ext:.3f}")

def main():
    # Load comprehensive eval
    data = load_comprehensive_eval()
    if data is None:
        return

    # Process results
    results = process_results(data)

    # Print summary
    print_summary(results)

    # Merge with extracted data
    merged = merge_with_extracted_data(results)

    print("\nDone! Run generate_figures.py to regenerate figures with new data.")

if __name__ == "__main__":
    main()
