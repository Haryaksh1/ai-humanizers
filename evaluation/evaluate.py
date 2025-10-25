import os
import sys
import csv
import textstat
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

# Dynamically determine the project root directory (the parent of the 'evaluation' folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, import the classes directly from the converted .py files
try:
    from humanizer_balanced import AdvancedAITextHumanizer
    from humanizer_aggressive import UltraAggressiveHumanizer
except ImportError as e:
    print(f"Error importing from .py files: {e}")
    print("Please ensure you have converted the notebooks to .py files using 'jupyter nbconvert --to script <notebook_name>.ipynb'")
    sys.exit(1)


# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_metrics(original_text, humanized_text):
    """Calculates readability, similarity, and other metrics."""
    original_tokens = word_tokenize(original_text.lower())
    humanized_tokens = word_tokenize(humanized_text.lower())

    # 1. Readability (Flesch Reading Ease)
    readability_score = textstat.flesch_reading_ease(humanized_text)

    # 2. Similarity (BLEU Score)
    bleu_score = sentence_bleu([original_tokens], humanized_tokens, weights=(0.5, 0.5))

    # 3. Word Count Change
    word_count_diff = len(humanized_tokens) - len(original_tokens)

    return {
        "readability_flesch": readability_score,
        "similarity_bleu": bleu_score,
        "word_count_change": word_count_diff,
    }

def run_evaluation():
    """Runs both humanizers on sample inputs and computes metrics."""
    print("Starting evaluation...")

    # --- Configuration ---
    INPUT_DIR = os.path.join(project_root, "example_inputs")
    OUTPUT_DIR = os.path.join(project_root, "evaluation")
    SAMPLE_FILE = os.path.join(INPUT_DIR, "sample_texts.txt")
    CSV_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    MD_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "evaluation_results.md")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Sample Texts ---
    try:
        with open(SAMPLE_FILE, 'r', encoding='utf-8') as f:
            full_text = f.read()
            # Split based on the structure of your sample_texts.txt
            sample_texts = [text.strip() for text in full_text.split('\n\nAnother example:\n') if text.strip()]
        print(f"Loaded {len(sample_texts)} sample texts from {SAMPLE_FILE}")
    except FileNotFoundError:
        print(f"Error: Sample input file not found at {SAMPLE_FILE}")
        return

    # --- Initialize Humanizers ---
    # Set load_datasets=False to speed up initialization for evaluation
    print("\nInitializing humanizers (datasets loading is disabled for speed)...")
    balanced_humanizer = AdvancedAITextHumanizer(load_datasets=False)
    aggressive_humanizer = UltraAggressiveHumanizer(load_datasets=False)
    
    humanizers = {
        "balanced": balanced_humanizer,
        "aggressive": aggressive_humanizer
    }

    # --- Run Humanization and Collect Results ---
    results = []
    for i, text in enumerate(sample_texts):
        print(f"\n--- Processing Sample {i+1} ---")
        print(f"Original: {text[:80]}...")
        
        for name, model in humanizers.items():
            print(f"Running {name} humanizer...")
            if name == "balanced":
                humanized_text, stats = model.humanize(text, intensity='maximum')
            else: # aggressive
                humanized_text, stats = model.humanize(text, intensity='ultra')

            metrics = calculate_metrics(text, humanized_text)
            
            results.append({
                "sample_id": i + 1,
                "humanizer_type": name,
                "original_text": text,
                "humanized_text": humanized_text,
                **metrics,
                "ai_score_before": stats.get('initial_ai_score', 0),
                "ai_score_after": stats.get('final_ai_score', 0),
            })
            print(f"Finished running {name} humanizer.")

    # --- Save Results to CSV ---
    if results:
        with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Evaluation results saved to {CSV_OUTPUT_FILE}")

    # --- Generate Markdown Summary ---
    summary = "# Humanizer Evaluation Results\n\n"
    summary += "This report compares the `balanced` and `aggressive` humanizers based on several metrics.\n\n"
    summary += "| Sample ID | Humanizer | Readability (Higher=Easier) | Similarity (Lower=More Changed) | AI Score Before | AI Score After |\n"
    summary += "|---|---|---|---|---|---|\n"

    for res in results:
        summary += (
            f"| {res['sample_id']} | {res['humanizer_type']} | "
            f"{res['readability_flesch']:.2f} | {res['similarity_bleu']:.4f} | "
            f"{res['ai_score_before']:.2f} | {res['ai_score_after']:.2f} |\n"
        )
    
    with open(MD_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"✅ Markdown summary saved to {MD_OUTPUT_FILE}")


if __name__ == "__main__":
    run_evaluation()
import sys
from pathlib import Path

# Ensure project root on path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import score_text  # noqa: E402


def demo():
    balanced_text = (
        "This tool converts formal, robotic writing into clear, natural prose. "
        "It keeps the original meaning while improving flow and readability."
    )

    aggressive_text = (
        "Imagine your paragraphs suddenly breathing—phrases loosen up, tone warms, "
        "and stiff constructs crack into lively, varied sentences!"
    )

    print("== Balanced output ==")
    print(balanced_text)
    print(score_text(balanced_text))
    print()

    print("== Aggressive output ==")
    print(aggressive_text)
    print(score_text(aggressive_text))


if __name__ == "__main__":
    demo()
