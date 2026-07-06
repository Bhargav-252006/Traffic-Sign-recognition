# Proposed Algorithm: Two-Stage Traffic Sign Recognition via IEEE Ensemble CNN and LLM Semantic Verification

---

## 1. Problem Statement

The goal is to correctly classify a traffic sign image into one of 43 categories from the German Traffic Sign Recognition Benchmark (GTSRB). Beyond simple classification, the system must also:

1. Identify when a prediction is uncertain and should not be blindly trusted.
2. Verify whether a prediction is semantically consistent with known visual properties of traffic signs.
3. Provide a human-readable explanation for every decision made.

A single CNN classifier handles task (1) poorly and completely ignores tasks (2) and (3). This proposed system solves all three through a two-stage pipeline: a CNN ensemble for initial prediction followed by an LLM for semantic verification.

---

## 2. Motivation

Prior work in traffic sign recognition has progressed through five phases:

- **2020–2021:** Standard CNNs achieved near-human accuracy (~98–99%) on GTSRB but operated as black boxes with no uncertainty awareness.
- **2022:** Lightweight and efficient CNNs were developed for deployment on embedded hardware, but the black-box problem remained.
- **2023:** Researchers addressed adversarial robustness and introduced hybrid CNN + Transformer models, but predictions still lacked semantic verification.
- **2024:** Vision Transformers and multi-task learning improved feature quality, yet no system reasoned about the meaning of its predictions.
- **2025:** Architectures began moving toward a vision encoder paired with a higher-level reasoning module, pointing directly toward the design used in this work.

This system is the natural realisation of that direction: a CNN ensemble producing predictions and an LLM reasoning about whether those predictions make semantic sense — using only text, never the raw image.

---

## 3. System Architecture

The system is divided into three stages executed in sequence:

**Stage 1 — IEEE Ensemble CNN:**
Three individual CNN models each independently classify the input image. Their probability outputs are averaged to produce a single ensemble prediction.

**Stage 2 — Confidence Gate:**
The ensemble output is checked against confidence and uncertainty thresholds. If the prediction is strong and certain, it is accepted immediately without involving the LLM.

**Stage 3 — LLM Semantic Reviewer:**
If the gate detects uncertainty or low confidence, the prediction details (class names, confidence scores, entropy, and margin) are sent as text to a Large Language Model. The LLM reasons about whether the prediction is semantically consistent with traffic sign visual categories and returns a structured verdict.

---

## 4. Proposed Algorithm — Step by Step

### Phase A: Image Preprocessing

**Step 1.** Receive the input image of a traffic sign (RGB colour image, any resolution).

**Step 2.** Resize the image to 32 × 32 pixels to match the training input size of all three CNN models.

**Step 3.** Convert pixel values to floating-point tensors and normalise them using the GTSRB dataset statistics (mean and standard deviation per colour channel). This ensures the input distribution matches what the models were trained on.

---

### Phase B: Stage 1 — IEEE Ensemble CNN Prediction

**Step 4.** Pass the preprocessed image through the first CNN model, **ConvNeXt-Tiny**. This model uses a modern hierarchical convolutional architecture with its first layer modified to use stride-1 instead of stride-4, which preserves finer spatial detail for small 32×32 inputs. Record the output probability score for each of the 43 traffic sign classes.

**Step 5.** Pass the same preprocessed image through the second CNN model, **ResNet18 (Enhanced)**. This is a residual network with its first convolution changed to stride-1 and its max-pooling layer removed, again to avoid over-reducing spatial resolution at small input sizes. Record the output probability score for each of the 43 classes.

**Step 6.** Pass the same preprocessed image through the third CNN model, **EfficientNet-B0**. This is a compound-scaled network with its stem convolution changed to stride-1 for the same reason. Record the output probability score for each of the 43 classes.

**Step 7.** Compute the **ensemble prediction** by taking the simple average of the three sets of probability scores across all 43 classes. This method is called soft voting — it combines the opinion of all three models rather than taking a majority vote, which results in a more calibrated and accurate final distribution.

**Step 8.** From the averaged probability distribution, identify the **Top-3 predicted classes** along with their confidence scores. Label them:
- Top-1: the most likely class (highest score)
- Top-2: the second most likely class
- Top-3: the third most likely class

**Step 9.** Calculate the **Shannon Entropy** of the full 43-class probability distribution. Entropy measures how spread out the model's belief is across all classes. A low entropy value means the model is very focused on one class. A high entropy value means the probability is spread across many classes, indicating confusion.

**Step 10.** Calculate the **Confidence Margin**, defined as the difference between the Top-1 confidence score and the Top-2 confidence score. A large margin means the top-1 class clearly dominates. A small margin means two classes are nearly tied, indicating genuine ambiguity.

---

### Phase C: Stage 2 — Confidence Gate

**Step 11.** Check whether the Top-1 confidence score is at or above **85%**.

**Step 12.** Check whether the Entropy value is at or below **0.50**.

**Step 13.** If both conditions in Step 11 and Step 12 are satisfied, the prediction is considered reliable. Accept the Top-1 class as the final prediction without any further verification. Record the decision path as **CNN_DIRECT** and proceed to output. Skip to Step 22.

**Step 14.** If either condition fails (confidence too low or entropy too high), the prediction is considered uncertain. Proceed to Stage 3.

---

### Phase D: Stage 3 — LLM Semantic Verification

**Step 15.** Construct a structured text description containing the following information:
- The name and confidence score of the Top-1, Top-2, and Top-3 predicted classes.
- The entropy value and whether it is Low, Medium, or High.
- The confidence margin value.
- A description of all five semantic groups of GTSRB traffic signs and their visual characteristics (shape, colour, border type).
- A set of reasoning rules that explain which combinations of top predictions are semantically plausible and which are suspicious.

**Step 16.** Send only this text to the LLM via the API. No image data is transmitted at any point. The LLM never sees pixel values — it only reads the structured description of the CNN's numerical output.

**Step 17.** The LLM reads the description and applies the following semantic reasoning rules to decide whether the CNN's prediction is trustworthy:

- **Rule 1 — Dominant Prediction:** If the confidence margin is greater than 50%, the Top-2 class score is considered negligible noise. The prediction is accepted regardless of which category Top-2 belongs to, because the margin is so large that the alternative is not meaningful.

- **Rule 2 — Fine-Grained Confusion:** If both Top-1 and Top-2 predictions belong to the same semantic group (for example, both are speed limit signs), the confusion is expected and visually plausible since signs within the same group look very similar to each other. The prediction is accepted.

- **Rule 3 — Structural Conflict:** If Top-1 and Top-2 belong to different semantic groups that have completely different physical shapes — for example, a circular blue mandatory sign versus a triangular red warning sign — and the Top-2 confidence is above 15%, this combination is semantically impossible. The prediction is rejected.

- **Rule 4 — Out-of-Distribution:** If the Top-1 confidence is below 50%, the margin is below 20%, and both top predictions come from different semantic groups, the model is genuinely confused rather than simply unsure between two similar-looking signs. The prediction is rejected as likely out-of-distribution input.

- **Rule 5 — Overconfidence Detection:** If the Top-1 confidence is extremely high (above 95%) but the Top-2 class belongs to a different semantic group and still has more than 10% confidence, this pattern is suspicious. A truly confident model would only have alternatives within the same visual group. The prediction is rejected as potentially false confidence.

- **Rule 6 — Negligible Alternative:** If the Top-2 confidence score is below 5%, it is treated as noise and completely ignored, even if it belongs to a different semantic group. Only meaningful alternatives warrant concern.

**Step 18.** The LLM returns a structured response containing four fields:
- A **decision**: either AGREE (the prediction is semantically consistent), DISAGREE (a conflict was detected), or UNCERTAIN (insufficient evidence to decide firmly).
- A **confusion type**: a label for the specific pattern detected, such as FINE_GRAINED, STRUCTURAL_CONFLICT, or DOMINANT_PREDICTION.
- A **reason**: a plain-English sentence explaining the decision in terms of the sign categories and visual properties.
- A **risk level**: LOW, MEDIUM, HIGH, or CRITICAL, reflecting how seriously the uncertainty should be treated.

---

### Phase E: Final Decision

**Step 19.** If the LLM decision is **AGREE**, accept the Top-1 CNN prediction as the final output. Record the decision path as **CNN_LLM_AGREED**.

**Step 20.** If the LLM decision is **DISAGREE**, do not output a class prediction. Flag the sample as **UNVERIFIED** and mark it for manual human review. Record the decision path as **CNN_LLM_DISAGREED**.

**Step 21.** If the LLM decision is **UNCERTAIN**, accept the Top-1 CNN prediction as a best-effort output but attach the risk level and reason as a caution note. Record the decision path as **CNN_LLM_UNCERTAIN**.

---

### Phase F: Output

**Step 22.** Return the final predicted class name and its original confidence score.

**Step 23.** Return the decision path taken: CNN_DIRECT, CNN_LLM_AGREED, CNN_LLM_DISAGREED, or CNN_LLM_UNCERTAIN.

**Step 24.** Return the explanation text — either the default message for direct outputs or the LLM-generated reason for reviewed outputs.

**Step 25.** Return the risk level (LOW, MEDIUM, HIGH, or CRITICAL) for any downstream system that needs to act differently based on the level of uncertainty.

---

## 5. Decision Path Summary

| Decision Path | When It Occurs | Final Action |
|---------------|---------------|-------------|
| CNN_DIRECT | Top-1 confidence is 85% or above AND Entropy is 0.50 or below | Accept CNN prediction directly — LLM not used |
| CNN_LLM_AGREED | CNN was uncertain; LLM confirmed semantic consistency | Accept CNN prediction |
| CNN_LLM_DISAGREED | CNN was uncertain; LLM detected a semantic conflict or structural impossibility | Reject prediction, flag for manual review |
| CNN_LLM_UNCERTAIN | CNN was uncertain; LLM could not reach a confident verdict | Accept CNN prediction with a caution note |

---

## 6. Semantic Groups Used by the LLM

The LLM's entire reasoning is anchored to five visual categories of traffic signs. These descriptions are embedded directly in the prompt so the LLM does not need prior traffic sign training to reason correctly:

**Group 1 — Speed Limit Signs (Classes 0 to 8 and Class 32)**
All have a circular shape with a red border and white background. They are distinguished from each other only by the number printed in the centre. Confusion between two speed limit signs is visually expected and semantically acceptable.

**Group 2 — Prohibitory Signs (Classes 9, 10, 15, 16, 17, 41, 42)**
All have a circular shape with a red border, containing a crossed-out symbol or vehicle type. Confusion within this group is acceptable. Confusion with the blue mandatory signs is not, because the background colours and meanings are fundamentally opposite.

**Group 3 — Priority, Stop, and Yield Signs (Classes 11 to 14)**
Each sign in this group has a unique and distinctive shape: a yellow diamond (Priority Road), a red octagon (Stop), an inverted triangle (Yield), and a white triangle with a red border (Right-of-Way at intersection). Confusion between any of these and any other group is highly suspicious because their geometric shapes are unlike anything else in the dataset.

**Group 4 — Warning and Danger Signs (Classes 18 to 31)**
All have a triangular shape with a red border and a pictogram inside describing the hazard. Confusion between two warning signs, such as one curve versus another, is visually reasonable. Confusion with circular or blue signs is a structural conflict.

**Group 5 — Mandatory Signs (Classes 33 to 40)**
All have a circular shape with a blue background and a white directional arrow. Confusion within this group, such as between "turn right" and "go straight," is acceptable because the signs look similar. Confusion with red-border prohibitory signs is a structural conflict because shape, colour, and meaning all differ.

---

## 7. Fallback Behaviour

If the LLM API is unavailable or returns an error, the system automatically falls back to a rule-based reviewer that implements the same six semantic reasoning rules (Step 17, Rules 1–6) locally without any API call. This ensures the system continues to produce risk-annotated decisions even in offline or rate-limited conditions.

If all configured LLM models — the primary model plus two fallback models — fail to respond, the system defaults to returning an UNCERTAIN decision with MEDIUM risk, and the Top-1 CNN prediction is accepted as a best-effort output.

---

## 8. Complete Algorithm Summary

1. Resize and normalise the input traffic sign image to 32 × 32 pixels.
2. Run the image through ConvNeXt-Tiny independently and record the 43-class probability scores.
3. Run the same image through ResNet18 independently and record the 43-class probability scores.
4. Run the same image through EfficientNet-B0 independently and record the 43-class probability scores.
5. Average the three sets of probability scores to produce the final ensemble probability distribution.
6. Identify the Top-3 classes and their confidence scores from the ensemble output.
7. Compute Shannon Entropy to measure how spread the prediction is across all 43 classes.
8. Compute the Confidence Margin as the gap between the Top-1 and Top-2 confidence scores.
9. Check if Top-1 confidence is at or above 85% and Entropy is at or below 0.50.
10. If yes — accept the Top-1 class directly as the final prediction (CNN_DIRECT path).
11. If no — build a text description of the Top-3 predictions, entropy, and margin along with traffic sign semantic group rules.
12. Send the text description to the LLM. No image is sent.
13. The LLM checks whether the top predictions belong to the same visual group or whether they conflict structurally, and applies the six reasoning rules.
14. If the LLM returns AGREE — accept the Top-1 class (CNN_LLM_AGREED path).
15. If the LLM returns DISAGREE — flag the image for manual human review (CNN_LLM_DISAGREED path).
16. If the LLM returns UNCERTAIN — accept the Top-1 class with a caution note (CNN_LLM_UNCERTAIN path).
17. Return the final class name, confidence score, decision path, explanation, and risk level.

---

## 9. Novelty of This Approach

1. **Text-only LLM verification.** The LLM never receives the raw image. It only reads the CNN's numerical output formatted as text. This makes the system fast, privacy-preserving, and compatible with any LLM provider.

2. **Structured semantic knowledge in the prompt.** Rather than asking the LLM to freely guess, the prompt embeds explicit descriptions of all five sign groups and six reasoning rules, turning the LLM into a constrained semantic auditor rather than a free classifier.

3. **Risk-annotated output.** Every prediction comes with a risk level and a confusion type label, enabling downstream systems such as driver-assistance software to respond proportionally to the level of uncertainty.

4. **Graceful offline fallback.** The rule-based mock reviewer mirrors the full LLM reasoning logic locally, ensuring the system works without an internet connection or API access.

5. **Three-model ensemble with uncertainty metrics.** Combining ConvNeXt-Tiny, ResNet18, and EfficientNet-B0 through soft voting provides better-calibrated confidence estimates than any single model, making the entropy and margin signals more reliable inputs for the gate and the LLM.

---

*Proposed algorithm for the TotalDataSetLLm two-stage GTSRB recognition system.*
*Related implementation: `TotalDataSetLLm/total_data_llm.ipynb`, `llm_reviewer.py`, `model_architectures.py`*
