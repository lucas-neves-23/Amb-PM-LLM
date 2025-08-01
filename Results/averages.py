import ast
import re
import numpy as np
from collections import defaultdict

# Specify the file path
file_path = "/Users/lucasneves/Desktop/Amb-PM-LLM/Results/one_shot_Query_PM_LLM_no_context.txt" 


# Read the file content
with open(file_path, "r") as file:
    content = file.read()

# Remove np.float64() wrappers using regex
content = re.sub(r'np\.float64\((.*?)\)', r'\1', content)

# Parse the cleaned content with ast.literal_eval
document = ast.literal_eval(content)

# Initialize lists for overall metrics
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []
bleu_scores = []
bertscore_scores = []

# Initialize dictionary to store metrics per category
category_metrics = defaultdict(lambda: {
    'rouge-1': [],
    'rouge-2': [],
    'rouge-l': [],
    'bleu': [],
    'bertscore': []
})

# Extract metrics from each entry and group by category
for entry in document:
    category = entry.get("category")
    metrics = entry.get("metrics", {})
    rouge_1 = float(metrics.get("rouge-1", 0.0))
    rouge_2 = float(metrics.get("rouge-2", 0.0))
    rouge_l = float(metrics.get("rouge-l", 0.0))
    bleu = float(metrics.get("bleu", 0.0))
    bertscore = float(metrics.get("bertscore", 0.0))
    
    # Append to category-specific lists
    category_metrics[category]['rouge-1'].append(rouge_1)
    category_metrics[category]['rouge-2'].append(rouge_2)
    category_metrics[category]['rouge-l'].append(rouge_l)
    category_metrics[category]['bleu'].append(bleu)
    category_metrics[category]['bertscore'].append(bertscore)
    
    # Append to overall lists
    rouge_1_scores.append(rouge_1)
    rouge_2_scores.append(rouge_2)
    rouge_l_scores.append(rouge_l)
    bleu_scores.append(bleu)
    bertscore_scores.append(bertscore)

# Calculate and print averages per category
categories = sorted(category_metrics.keys())
for category in categories:
    metrics = category_metrics[category]
    n = len(metrics['rouge-1'])
    if n > 0:
        avg_rouge_1 = np.mean(metrics['rouge-1'])
        avg_rouge_2 = np.mean(metrics['rouge-2'])
        avg_rouge_l = np.mean(metrics['rouge-l'])
        avg_bleu = np.mean(metrics['bleu'])
        avg_bertscore = np.mean(metrics['bertscore'])
    else:
        avg_rouge_1 = 0.0
        avg_rouge_2 = 0.0
        avg_rouge_l = 0.0
        avg_bleu = 0.0
        avg_bertscore = 0.0
    
    print(f"Category {category} (n={n}):")
    print(f"  Average ROUGE-1: {avg_rouge_1:.4f}")
    print(f"  Average ROUGE-2: {avg_rouge_2:.4f}")
    print(f"  Average ROUGE-L: {avg_rouge_l:.4f}")
    print(f"  Average BLEU: {avg_bleu:.4f}")
    print(f"  Average BERTScore: {avg_bertscore:.4f}")

# Calculate and print overall averages
n_total = len(rouge_1_scores)
print(f"\nOverall (n={n_total}):")
avg_rouge_1 = np.mean(rouge_1_scores) if n_total > 0 else 0.0
avg_rouge_2 = np.mean(rouge_2_scores) if n_total > 0 else 0.0
avg_rouge_l = np.mean(rouge_l_scores) if n_total > 0 else 0.0
avg_bleu = np.mean(bleu_scores) if n_total > 0 else 0.0
avg_bertscore = np.mean(bertscore_scores) if n_total > 0 else 0.0

print(f"Average ROUGE-1: {avg_rouge_1:.4f}")
print(f"Average ROUGE-2: {avg_rouge_2:.4f}")
print(f"Average ROUGE-L: {avg_rouge_l:.4f}")
print(f"Average BLEU: {avg_bleu:.4f}")
print(f"Average BERTScore: {avg_bertscore:.4f}")