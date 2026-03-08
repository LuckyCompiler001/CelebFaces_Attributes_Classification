## Model Task / I/O

- **Input:**  
  A batch of RGB face images `x` with shape `[B, 3, H, W]`  
  (typically resized/cropped to `224×224` for `timm` backbones).

- **Output:**  
  For each image, the model outputs **40 real-valued logits** `z`  
  with shape `[B, 40]`, one logit per semantic attribute.

- **Post-processing:**  
  - Apply sigmoid to logits:  
    $p = \sigma(z)$.
  - Threshold probabilities (e.g., `p > 0.5`) to produce  
    **40 binary attribute predictions per image**.

- **Training Objective:**  
  Multi-label classification (each image may have multiple attributes true).  
  Use:
  $$\text{BCEWithLogitsLoss}$$
  between the 40 logits and the 40-dimensional binary target vector.

---

## Dataset / Data Loading

- **Dataset:**  
  CelebA (~200k celebrity face images) with **40 binary attribute annotations** per image.  
  Use the official `train / val / test` split provided by CelebA.

- **Each Sample Provides:**
  - `image`: RGB face photo
  - `target`: 40-dimensional binary vector $y \in \{0,1\}^{40}$
- **The 40 Attributes:**:

- [
  ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs"],
  ["Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair"],
  ["Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair"],
  ["Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes"],
  ["No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks"],
  ["Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat"],
  ["Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
]

## Probe Inspirations (CelebA Multi-Label Attribute Classification Context)
### 1. Region–Attribute Causal Sensitivity Probe
**Description:**  
Evaluate whether an attribute prediction is causally driven by its semantically relevant facial region rather than peripheral cues.
**Example:**  
Mask or occlude the mouth region when predicting *Wearing_Lipstick* or *Smiling*. Compute Δ-logit or Δ-probability before and after masking.  
- Large change when masking the correct region → expected causal dependence.  
- Large change when masking irrelevant regions (e.g., hair, background) → spurious feature reliance.  
This probe directly tests whether the model localizes evidence appropriately.
### 2. Contextual Shortcut Dependency Probe
**Description:**  
Assess whether predictions depend on background, clothing, or illumination rather than intrinsic facial features.
**Example:**  
Apply controlled background blur or replacement while preserving facial pixels. Measure prediction shift for attributes like *Male*, *Young*, or *Smiling*.  
Significant Δ-probability under background-only edits suggests shortcut learning from contextual co-occurrence patterns.
### 3. Cross-Attribute Logical Consistency Probe
**Description:**  
Examine whether the model encodes coherent relationships across attributes instead of predicting them independently.
**Example:**  
Quantify violation rate of logically incompatible pairs (e.g., *Bald* ∧ *Bangs*, *No_Beard* ∧ *Mustache*).  
High joint probability for inconsistent attribute pairs indicates weak structural modeling and potential reliance on local texture statistics rather than global facial semantics.
### 4. Demographic Proxy Sensitivity Probe
**Description:**  
Test whether non-visual demographic attributes disproportionately influence cosmetic or appearance-based predictions.
**Example:**  
Analyze conditional probabilities such as  
P(Wearing_Lipstick | Male = high) vs P(Wearing_Lipstick | Male = low).  
If lipstick prediction systematically shifts with predicted *Male* status even when visual evidence is constant, the model may encode demographic bias instead of visual causation.
### 5. Controlled Perturbation Stability Probe
**Description:**  
Measure robustness under non-semantic perturbations and responsiveness under semantic edits.
**Example:**  
Apply mild brightness, color jitter, or Gaussian noise.  
- Excessive prediction variance under minor pixel changes → over-sensitivity to low-level statistics.  
- Minimal response to meaningful edits (e.g., synthetic hair recoloring for *Blond_Hair*) → failure to encode true semantic feature.  