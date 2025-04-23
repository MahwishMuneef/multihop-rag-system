import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from multihop import answer_multi_hop_query

# Determine the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate_predictions(json_filename: str = 'pregnancy_qna.json') -> list:
    """
    Load ground truth questions and answers from a JSON file located in the same folder as this script,
    run multi-hop QA to get predictions, and compute BLEU and ROUGE metrics.
    JSON entries should have keys 'question' and 'answer'.
    Returns a list of result dicts.
    """
    json_path = os.path.join(SCRIPT_DIR, json_filename)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Ground-truth file '{json_path}' not found.")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize scorers
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    smooth_fn = SmoothingFunction().method1

    results = []
    for item in data:
        question = item.get('question', '').strip()
        reference = item.get('answer', '').strip()
        # Generate prediction from multi-hop QA
        response = answer_multi_hop_query(question)
        prediction = getattr(response, 'content', str(response)).strip()

        # Compute BLEU score
        bleu = sentence_bleu([
            reference.split()
        ], prediction.split(), smoothing_function=smooth_fn)

        # Compute ROUGE scores
        rouge_scores = rouge_scorer_obj.score(reference, prediction)
        r1 = rouge_scores['rouge1'].fmeasure
        r2 = rouge_scores['rouge2'].fmeasure
        rl = rouge_scores['rougeL'].fmeasure

        results.append({
            'question': question,
            'reference': reference,
            'prediction': prediction,
            'bleu': bleu,
            'rouge_1_f1': r1,
            'rouge_2_f1': r2,
            'rouge_l_f1': rl
        })

    return results


if __name__ == '__main__':
    try:
        scores = evaluate_predictions()
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Print per-sample results
    for idx, sc in enumerate(scores, start=1):
        print(f"Sample {idx}: Q: {sc['question']}")
        print(f"  Reference:  {sc['reference']}")
        print(f"  Prediction: {sc['prediction']}")
        print(f"  BLEU:       {sc['bleu']:.4f}")
        print(f"  ROUGE-1 F1: {sc['rouge_1_f1']:.4f}")
        print(f"  ROUGE-2 F1: {sc['rouge_2_f1']:.4f}")
        print(f"  ROUGE-L F1: {sc['rouge_l_f1']:.4f}\n")

    # Compute and print averages
    avg_bleu = sum(s['bleu'] for s in scores) / len(scores)
    avg_r1 = sum(s['rouge_1_f1'] for s in scores) / len(scores)
    avg_r2 = sum(s['rouge_2_f1'] for s in scores) / len(scores)
    avg_rl = sum(s['rouge_l_f1'] for s in scores) / len(scores)

    print("Average Scores:")
    print(f"  BLEU:       {avg_bleu:.4f}")
    print(f"  ROUGE-1 F1: {avg_r1:.4f}")
    print(f"  ROUGE-2 F1: {avg_r2:.4f}")
    print(f"  ROUGE-L F1: {avg_rl:.4f}")
