## Second Prompt — Revised Version
We will proceed with this probe idea:
{
"probe_type": "Region-Attribute Causal Sensitivity",
"probe_name": "Mouth Occlusion for Lipstick and Smiling",
"explanation": "Systematically occlude or blur only the mouth/lip bounding box for CelebA crops and measure Δ-logit for Wearing_Lipstick, Smiling, and Mouth_Slightly_Open; sharp confidence drops only under mouth occlusion verify the model is using semantically correct evidence rather than global hue cues.",
"reference": "Fong & Vedaldi, 2017, Interpretable Explanations via Meaningful Perturbations",
"confidence": 0.88
},
### Role Setup
Imagine you are a professional Machine Learning Engineer (MLE). Read the following file relationships carefully:
- You are allowed to read `./` for information.
- `.wrapper_plot.py` — contains plotting functions you must use to display your probe conclusion (see head comment for details).
## Probe Intention
The probe.py should reflect the quality of the current training process.
For example, you may extract statistics such as true/false rate, prediction error, etc., then:
- Construct probe logic based on these data.
- Generate a conclusion.
- Present the conclusion using plots (via `wrapper_plot.py`).
## Core Implementation Principle
Your probe must explicitly define:
### 1. Metric
A numerical measurement that quantifies model performance.
- The metric **must be computed from data accessible during training**.
- It must be clearly defined with a precise computation formula.
- Example: covariance between model quality and race label in a bias probe.
---
### 2. Expectation (a string)
A description of what the metric should look like for a well-trained model.
- Define what “good performance” means numerically.
- Clearly interpret what values indicate strong vs weak training.
- Example: A well-trained model should produce balanced subgroup distributions.
---
### 3. Threshold (a numerical value)
A numerical decision boundary used to judge performance.
- Explicitly define the threshold.
- Compare metric value against threshold.
- Example: If metric > 0.5 and threshold = 0.5 → considered good.
You must clearly define:
- Metric
- Expectation
- Threshold
- Their logical relationship
### 4. visualization
you must read and properly call wrapper_plot.py to display your conclusion as figure/plot
### 5. other Required
When designing the probe:
- Explicitly state metric, expectation, and threshold.
- Clearly describe:
  - Data source
  - Data processing steps
  - Computation formula
  - Interpretation logic
- Use functions from `wrapper_plot.py` for visualization. read what plot is supported in wrapper_plot and consider how to design your implementation
- stdout your probe conclusions in a JSON file in the following format:

```json
{
  "metric_data1": "numerical value",
  "threshold": "numerical value",
  "expectation": "textual interpretation of how to evaluate the metric"
}
```
## Your Task
Generate **three different implementation (`dev_docs`)**.
### Requirements for Each `dev_doc`
Each `dev_doc` must:
- Follow a **consistent and structured format**
- Clearly define:
  - **Metric**
  - **Expectation**
  - **Threshold**
- Describe:
  - **Data collection method**
  - **Computation formula**
  - **Result interpretation logic**
  - **Visualization plan**
  - **JSON output structure**
- Be written **clearly, logically, and concisely**
- Only use **data accessible during training**
### Output Requirements
return a json in following schema to me in the conversation
## JSON Schema Requirement
Your output must follow this exact structure:
```json
{
  "dev_docs": [
    {
      "rank": 1,
      "title": "string",
      "confidence": 0.0,
      "metric": {
        "name": "string",
        "definition": "string",
        "formula": "string"
      },
      "expectation": "string",
      "threshold": {
        "value": 0.0,
        "comparison_logic": "string"
      },
      "data_source": "string",
      "data_processing_steps": ["step1", "step2"],
      "interpretation_logic": "string",
      "visualization_plan": "string (must reference wrapper_plot functions)",
      "json_output_example": {
        "metric_name": 0.0,
        "threshold": 0.0,
        "expectation": "string",
        "interpretation": "string"
      }
    }
  ]
}