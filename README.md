# Champion-Challenger Model Comparison with Hugging Face and MLflow

**Student:** S Avantika  
**USN:** 1MS25CI108  
**Project:** Skill Lab (MLOps with MLflow)  
**Repository:** https://github.com/6vantika/Skill-Lab-Project  

---

## 1. Task and Models

This project implements a complete Champion-Challenger evaluation framework for binary sentiment classification. We evaluate two pre-trained transformer models from Hugging Face on the exact same benchmark dataset, log all runs and artifacts to an MLflow tracking server, register both models in the MLflow Model Registry, and deploy the winner under the `champion` alias for inference on fresh test data.

| Item | Details |
|------|---------|
| **Task** | Sentiment Classification (Binary: POSITIVE / NEGATIVE) |
| **Dataset** | Stanford Sentiment Treebank (SST-2), validation split |
| **Evaluation Subset** | 200 samples, fixed random seed = 42 |
| **Input File** | `data/eval_inputs.csv` (saved references and text) |
| **Model A (Baseline / Initial Champion)** | `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters) |
| **Model B (Candidate Challenger)** | `siebert/sentiment-roberta-large-english` (~355M parameters) |
| **Compute Device** | CPU (batch size = 32) |

---

## 2. How to Run

Follow these commands in order from the repository root:

```bash
# 1. Clone the repository
git clone https://github.com/6vantika/Skill-Lab-Project.git
cd Skill-Lab-Project

# 2. Install required packages
pip install -r requirements.txt

# 3. Start the MLflow tracking server in the background (or in a separate terminal)
python -m mlflow server --host 127.0.0.1 --port 5000

# 4. Run the evaluation script
# This extracts 200 SST-2 samples, runs both models, logs metrics, artifacts, and models to MLflow
python evaluate.py

# 5. Register models and assign aliases in the MLflow Model Registry
# Reads run names and the summary winner tag dynamically without hard-coded IDs
python register_champion.py

# 6. Load the Champion model via its registry alias and run inference on unseen test data
# Note: This script loads the model strictly through MLflow without any direct Hugging Face imports
python predict_from_mlflow.py
```

---

## 3. Champion Rule (Written Before the Final Run)

To maintain experimental integrity and avoid p-hacking or confirmation bias, the Champion selection rule was formulated prior to executing the evaluation:

- **Primary Metric:** Macro-F1 score (higher is better). Sentiment datasets can show slight class imbalances, making macro-F1 a more balanced measure of class performance than plain accuracy.
- **Minimum Improvement Margin:** The Challenger (Model B) must outperform the Champion (Model A) by at least 1.00 percentage point (0.0100 in macro-F1). If the improvement is below this margin, the incumbent Champion retains its status.
  - *Rationale for Margin:* On an evaluation set of 200 examples, small differences below 1% frequently represent sampling variance or label noise rather than genuine capability gains. A 1 percentage point buffer prevents unwarranted model churn.
- **Tie-Breaker and Guardrail:** If the difference in macro-F1 is within the 1.00 percentage point margin, the model with lower average inference latency (milliseconds per sample) wins. Furthermore, if the Challenger's average latency exceeds 10x the Champion's latency, the Challenger is rejected unless the margin exceeds 3.0 percentage points.
- **Direction:** Higher macro-F1 is better. Lower average latency is better.

---

## 4. Results Table

Both models were evaluated on the exact same 200 sentences in identical order (`data/eval_inputs.csv`).

| Metric | Model A (DistilBERT) | Model B (RoBERTa-large) | Difference (B - A) | Superior |
|--------|----------------------|-------------------------|--------------------|----------|
| **Accuracy** | 0.9350 (93.50%) | 0.9450 (94.50%) | +0.0100 (+1.00%) | Model B |
| **Macro-F1 (Primary)** | **0.9343** | **0.9444** | **+0.0101 (+1.01%)** | **Model B** |
| **Macro-Precision** | 0.9359 | 0.9460 | +0.0101 (+1.01%) | Model B |
| **Macro-Recall** | 0.9331 | 0.9432 | +0.0101 (+1.01%) | Model B |
| **Average Latency (ms/sample)** | 40.45 ms | 252.32 ms | +211.87 ms (6.2x slower) | Model A |
| **Total Inference Time** | 8.09 s | 50.46 s | +42.37 s | Model A |

### Outcome
- Primary metric difference: $\Delta \text{Macro-F1} = +0.0101$ (1.01 percentage points).
- Pre-defined threshold: $0.0100$ (1.00 percentage points).
- Result: **The Challenger (Model B) wins** by clearing the threshold and is promoted to Champion.

---

## 5. Decision (150 to 250 Words)

Based on our pre-registered decision rule, Model B (siebert/sentiment-roberta-large-english) takes the Champion title over Model A (distilbert-base-uncased-finetuned-sst-2-english). The evaluation on 200 validation samples from SST-2 produced a macro-F1 of 0.9444 for Model B compared to 0.9343 for Model A. This represents a margin of +0.0101 (1.01 percentage points), which narrowly clears our required 1.0 percentage point improvement threshold.

Looking beyond macro-F1, Model B also achieved higher accuracy (94.5% versus 93.5%), macro-precision (0.9460 versus 0.9359), and macro-recall (0.9432 versus 0.9331). On our 8 fresh out-of-distribution test sentences, Model B achieved perfect 8/8 accuracy with high confidence scores around 0.999.

However, the operational trade-offs are significant. Model B has approximately 355 million parameters compared to 67 million for Model A, and its average inference latency on CPU was 252.32 ms per sample versus 40.45 ms for Model A. That is a 6.2x slowdown. In a production setting with strict sub-50ms latency budgets, DistilBERT might still be preferred despite the lower score. For our offline benchmark where accuracy is paramount and the rule was set in advance, Model B rightfully earns the Champion alias in the registry.

---

## 6. Qualitative Analysis and Model Disagreements

Across the 200 evaluation samples, the two models disagreed on 18 predictions (9.0% disagreement rate). The top disagreement cases show clear behavioral differences:

1. *"fun, flip and terribly hip bit of cinematic entertainment."* (Ground Truth: POSITIVE)  
   - Model A correctly predicted POSITIVE (confidence: 99.99%).  
   - Model B predicted NEGATIVE (confidence: 99.85%), likely tricked by the informal slang "flip" and "terribly".

2. *"it 's a bit disappointing that it only manages to be decent instead of dead brilliant."* (Ground Truth: NEGATIVE)  
   - Model A correctly predicted NEGATIVE (confidence: 99.94%).  
   - Model B predicted POSITIVE (confidence: 99.72%), likely weighing "decent" and "dead brilliant" over "disappointing".

3. *"oh come on."* (Ground Truth: NEGATIVE)  
   - Model A failed and predicted POSITIVE (confidence: 98.51%).  
   - Model B correctly understood the dismissive conversational tone and predicted NEGATIVE (confidence: 99.39%).

4. *"/ but daphne, you 're too buff / fred thinks he 's tough / and velma - wow, you 've lost weight !"* (Ground Truth: NEGATIVE)  
   - Model A correctly predicted NEGATIVE (confidence: 99.93%).  
   - Model B predicted POSITIVE (confidence: 99.03%), misinterpreting "lost weight" as positive praise.

All 18 disagreement examples are saved with commentary in `outputs/disagreement_examples.csv`.

---

## 7. Inference on New Data (`predict_from_mlflow.py`)

Step 5.6 requires loading the Champion model purely using the MLflow Model Registry alias URI (`models:/sentiment-model@champion`) without any direct calls to Hugging Face pipeline APIs.

We authored 8 realistic student-life sentences that are completely separate from SST-2:

```
Loading champion model from: models:/sentiment-model@champion
Model loaded successfully.

Registry name : sentiment-model
Alias         : champion
Model version : 2
Source run    : b34d614fc4c448b3902734f678b0955b
HF model ID   : siebert/sentiment-roberta-large-english

----------------------------------------------------------------------------------------------------
Sentence                                                               Predicted    Expected   Conf  
----------------------------------------------------------------------------------------------------
The new cafe on Brigade Road makes the best filter coffee I have ev... POSITIVE     1          0.9988
I wasted two hours trying to get the printer to work and it still d... NEGATIVE     0          0.9995
My roommate cooked an amazing biryani for dinner last night.           POSITIVE     1          0.9988
The bus was forty minutes late and completely packed, worst commute... NEGATIVE     0          0.9995
This wireless mouse has been working flawlessly for over a year now.   POSITIVE     1          0.9989
The hostel wifi keeps disconnecting every five minutes, it is so fr... NEGATIVE     0          0.9995
Our college fest had a brilliant lineup of bands this year.            POSITIVE     1          0.9989
The canteen food has gone downhill, everything tastes bland and stale. NEGATIVE     0          0.9995
----------------------------------------------------------------------------------------------------

Accuracy on new data: 8/8 (100%)
```

---

## 8. MLflow Run Structure and Artifacts

### Run Organization
- **Summary Parent Run:** `compare_sentiment`
  - Parameters: `primary_metric`, `margin`, `tiebreaker`, `eval_samples`, `model_a`, `model_b`
  - Metrics: `delta_macro_f1`, `model_a_macro_f1`, `model_b_macro_f1`, `model_a_accuracy`, `model_b_accuracy`, `model_a_avg_latency_ms`, `model_b_avg_latency_ms`
  - Tag: `winner = B`, `student_usn = 1MS25CI108`
  - Artifacts: `champion_vs_challenger_metrics.png`, `disagreement_examples.csv`, `metrics_summary.json`, `README.md`, `requirements.txt`
- **Child Run A:** `A_sentiment_distilbert-sst2`
  - Parameters: `hf_model_id`, `task`, `role = champion_candidate`, `eval_samples`, `seed`, `device`, `batch_size`
  - Metrics: `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, `avg_latency_ms`, `total_inference_s`
  - Artifacts: `predictions_distilbert.csv`, `confusion_matrix_distilbert.png`, `eval_inputs.csv`, logged pyfunc model
- **Child Run B:** `B_sentiment_roberta-large`
  - Parameters: `hf_model_id`, `task`, `role = challenger_candidate`, `eval_samples`, `seed`, `device`, `batch_size`
  - Metrics: `accuracy`, `macro_f1`, `macro_precision`, `macro_recall`, `avg_latency_ms`, `total_inference_s`
  - Artifacts: `predictions_roberta.csv`, `confusion_matrix_roberta.png`, `eval_inputs.csv`, logged pyfunc model

### Model Registry Setup
- Registered Model Name: `sentiment-model`
- Version 1: Model A (`distilbert-base-uncased-finetuned-sst-2-english`) -> Alias: `challenger`
- Version 2: Model B (`siebert/sentiment-roberta-large-english`) -> Alias: `champion`

---

## 9. Screenshots

All three required screenshots are included in the `screenshots/` directory:

1. **MLflow Compare View:** [`screenshots/mlflow_compare_view.png`](screenshots/mlflow_compare_view.png)  
   Shows side-by-side metric comparison, parameter differences, and run tags between Run A and Run B.

2. **Registry Aliases View:** [`screenshots/registry_aliases.png`](screenshots/registry_aliases.png)  
   Shows `sentiment-model` versions 1 and 2 with the assigned `champion` and `challenger` aliases.

3. **Terminal Output:** [`screenshots/predict_terminal_output.png`](screenshots/predict_terminal_output.png)  
   Shows execution of `predict_from_mlflow.py` loading Model Version 2 via alias URI and achieving 8/8 predictions.

---

## 10. Repository File Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── evaluate.py
├── register_champion.py
├── predict_from_mlflow.py
├── data/
│   ├── eval_inputs.csv
│   └── new_data.csv
├── outputs/
│   ├── champion_vs_challenger_metrics.png
│   ├── confusion_matrix_distilbert.png
│   ├── confusion_matrix_roberta.png
│   ├── disagreement_examples.csv
│   ├── metrics_summary.json
│   ├── new_data_predictions.csv
│   ├── predictions_distilbert.csv
│   └── predictions_roberta.csv
└── screenshots/
    ├── mlflow_compare_view.png
    ├── predict_terminal_output.png
    └── registry_aliases.png
```

---

## 11. Limitations and Declaration

- **Sample Size:** 200 samples from the SST-2 validation split provide a solid benchmark, but a larger test set (e.g. 1000+ examples) would yield tighter statistical confidence intervals. The 1.01 percentage point margin is narrow and could fluctuate on a different random sample.
- **Hardware Profile:** All benchmarks were run on a standard multi-core laptop CPU. On a GPU, the 6.2x latency disparity would be significantly reduced through parallel tensor operations.
- **Domain Specificity:** Both models are fine-tuned on film reviews (SST-2). While RoBERTa-large showed strong generalizability on our custom campus life dataset, performance across specialized domains (such as financial or legal sentiment) may require dedicated fine-tuning.
- **AI Assistance Declaration:** In accordance with the academic guidelines, I declare that an AI assistant was utilized during the development of this project for script scaffolding and troubleshooting SSL certificate trust on Windows. All methodology, rule definitions, and technical interpretations were verified by me, and I am prepared to explain every aspect of this implementation in a viva.
