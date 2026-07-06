"""
LLM Reviewer for Traffic Sign Recognition — Full GTSRB (43 Classes)
Uses OpenRouter API to verify IEEE Ensemble CNN predictions.
Adapted from two_stage_system/llm_reviewer.py for all 43 GTSRB classes.
"""

import json
import os
from typing import Dict
import requests


# ═══════════════════════════════════════════════════════════════════
# Full GTSRB 43-class descriptions (for LLM semantic reasoning)
# ═══════════════════════════════════════════════════════════════════
SIGN_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5t",
    11: "Right-of-way at next intersection", 12: "Priority road",
    13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "Vehicles over 3.5t prohibited", 17: "No entry",
    18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows on the right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing",
    30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead", 34: "Turn left ahead", 35: "Ahead only",
    36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End of no passing", 42: "End of no passing for vehicles over 3.5t",
}

# Semantic groups for GTSRB visual hierarchy reasoning
SEMANTIC_GROUPS = {
    "Speed Limits": [0, 1, 2, 3, 4, 5, 6, 7, 8, 32],
    "Prohibitory": [9, 10, 15, 16, 17, 41, 42],
    "Priority/Stop/Yield": [11, 12, 13, 14],
    "Warning/Danger": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    "Mandatory": [33, 34, 35, 36, 37, 38, 39, 40],
}


def get_semantic_group(class_id: int) -> str:
    """Get the semantic group for a GTSRB class ID."""
    for group_name, class_ids in SEMANTIC_GROUPS.items():
        if class_id in class_ids:
            return group_name
    return "Unknown"


class LLMReviewer:
    """
    LLM-based output reviewer for traffic sign classification (43 classes).
    - Text-only input (NO images)
    - Structured JSON output
    - Supports OpenRouter API (local or cloud models)
    """

    def __init__(self, api_key: str = None, model: str = "meta-llama/llama-3.1-8b-instruct:free"):
        """
        Initialize LLM Reviewer.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model to use (default: Llama 3.1 8B free tier)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def create_prompt(self, cnn_output: Dict) -> str:
        """
        Create structured text prompt for LLM semantic verification.
        NO IMAGE — Text only.
        """
        preds = cnn_output['predictions']

        prompt = f"""You are an independent verification agent for traffic sign recognition.
You must NOT classify images.
You must NOT use fixed confidence thresholds.
You must reason about semantic consistency using traffic sign visual hierarchy.

Context:
Traffic sign recognition using an IEEE Ensemble CNN (ConvNeXt-Tiny + ResNet18 + EfficientNet-B0)
with soft-voting on the full GTSRB dataset (43 classes).

GTSRB Traffic Sign Semantic Groups (43 classes):

1. SPEED LIMIT SIGNS (Circular, Red Border, White Background):
   Classes 0-8: 20/30/50/60/70/80/100/120 km/h speed limits
   Class 6: End of speed limit 80 km/h (crossed out)
   Class 32: End of all speed and passing limits

2. PROHIBITORY SIGNS (Circular, Red Border):
   Class 9: No passing
   Class 10: No passing for vehicles over 3.5t
   Class 15: No vehicles
   Class 16: Vehicles over 3.5t prohibited
   Class 17: No entry
   Class 41: End of no passing
   Class 42: End of no passing for vehicles over 3.5t

3. PRIORITY/STOP/YIELD (Distinctive Shapes):
   Class 11: Right-of-way (triangular, white with red border)
   Class 12: Priority road (diamond, yellow)
   Class 13: Yield (inverted triangle)
   Class 14: Stop (octagonal, red)

4. WARNING/DANGER SIGNS (Triangular, Red Border):
   Classes 18-31: General caution, curves, bumpy road, slippery road,
   road narrows, road work, traffic signals, pedestrians,
   children crossing, bicycles, ice/snow, wild animals

5. MANDATORY SIGNS (Circular, Blue Background):
   Classes 33-40: Turn right/left, ahead only, go straight or right/left,
   keep right/left, roundabout

Visual Similarity Rules:
- Speed limit signs differ ONLY by numerals (high confusion is EXPECTED)
- Prohibitory vs Mandatory have DIFFERENT shapes/colors (confusion is SUSPICIOUS)
- Warning signs share triangular shape (moderate confusion is REASONABLE)
- Priority/Stop/Yield each have unique shapes (confusion between them is SUSPICIOUS)

Model Output:
Top-1: {preds[0]['class_name']} (confidence = {preds[0]['confidence']:.3f})
Top-2: {preds[1]['class_name']} (confidence = {preds[1]['confidence']:.3f})
Top-3: {preds[2]['class_name']} (confidence = {preds[2]['confidence']:.3f})

Uncertainty Indicators:
Entropy level: {"High" if cnn_output['entropy'] > 0.5 else "Medium" if cnn_output['entropy'] > 0.3 else "Low"} ({cnn_output['entropy']:.3f})
Margin (Top-1 minus Top-2): {cnn_output['margin']:.3f}

Your Task:
Categorize the type of confusion using the semantic groups above.

Analyze:
1. FINE-GRAINED CONFUSION: Top-K classes from same group
   → Expected, visually similar → AGREE (Low Risk)

2. STRUCTURAL CONFLICT: Top-K classes from different shape categories
   → Unlikely geometric error → DISAGREE (High Risk)

3. OUT-OF-DISTRIBUTION: Near-equal prob + unrelated classes
   → Model is hallucinating → DISAGREE (Critical)

4. SEMANTIC VALIDATION: Low confidence BUT classes from same group
   → Confusion is plausible → AGREE (Validated Low Confidence)

5. OVERCONFIDENCE DETECTION: High confidence BUT unrelated alternatives
   → Suspicious certainty → DISAGREE (False Confidence)

Decide ONE based on semantic category:
AGREE → confusion pattern matches expected visual similarity
DISAGREE → semantic inconsistency detected
UNCERTAIN → ambiguous pattern, insufficient evidence

Output STRICT JSON only:
{{
  "decision": "AGREE | DISAGREE | UNCERTAIN",
  "confusion_type": "FINE_GRAINED | STRUCTURAL_CONFLICT | OUT_OF_DISTRIBUTION | SEMANTIC_VALIDATED | OVERCONFIDENCE | AMBIGUOUS",
  "reason": "Explain using semantic groups from knowledge base",
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL"
}}"""

        return prompt

    def call_api(self, prompt: str) -> Dict:
        """Call OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a traffic sign recognition verifier. "
                        "Analyze CNN predictions for logical consistency. "
                        "Output ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }

        try:
            response = requests.post(
                self.base_url, headers=headers, json=data, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            llm_output = result["choices"][0]["message"]["content"]

            # Parse JSON from LLM response
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()

            parsed_response = json.loads(llm_output)

            # Validate required fields
            required_fields = ["decision", "reason", "risk_level"]
            if not all(field in parsed_response for field in required_fields):
                raise ValueError("LLM response missing required fields")

            if parsed_response["decision"] not in ["AGREE", "DISAGREE", "UNCERTAIN"]:
                raise ValueError(f"Invalid decision: {parsed_response['decision']}")

            if parsed_response["risk_level"] not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                raise ValueError(f"Invalid risk_level: {parsed_response['risk_level']}")

            return parsed_response

        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return {
                "decision": "UNCERTAIN",
                "confusion_type": "AMBIGUOUS",
                "reason": f"API call failed: {str(e)}",
                "risk_level": "MEDIUM",
            }
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            return {
                "decision": "UNCERTAIN",
                "confusion_type": "AMBIGUOUS",
                "reason": "LLM returned invalid JSON format",
                "risk_level": "MEDIUM",
            }
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return {
                "decision": "UNCERTAIN",
                "confusion_type": "AMBIGUOUS",
                "reason": f"Unexpected error: {str(e)}",
                "risk_level": "HIGH",
            }

    def review(self, cnn_output: Dict) -> tuple:
        """
        Review CNN output.

        Args:
            cnn_output: Dictionary containing predictions and uncertainty metrics

        Returns:
            (llm_response, prompt): LLM decision and the prompt used
        """
        prompt = self.create_prompt(cnn_output)
        llm_response = self.call_api(prompt)
        return llm_response, prompt


def llm_reviewer_mock(cnn_output: Dict) -> Dict:
    """
    Mock LLM reviewer simulating SEMANTIC reasoning with 43-class visual hierarchy.

    Performs structured confusion categorization:
    - FINE_GRAINED: Same semantic group (low risk)
    - STRUCTURAL_CONFLICT: Different shapes (high risk)
    - OUT_OF_DISTRIBUTION: Hallucination (critical risk)
    - SEMANTIC_VALIDATED: Low confidence but plausible
    - OVERCONFIDENCE: High confidence but suspicious

    Replace with actual LLM API call for production use.
    """
    preds = cnn_output['predictions']
    confidence = preds[0]['confidence']
    margin = cnn_output['margin']
    entropy = cnn_output['entropy']

    top1_class = preds[0].get('class_id', -1)
    top2_class = preds[1].get('class_id', -1)

    # Convert to int if string
    if isinstance(top1_class, str):
        top1_class = int(top1_class)
    if isinstance(top2_class, str):
        top2_class = int(top2_class)

    top1_group = get_semantic_group(top1_class)
    top2_group = get_semantic_group(top2_class)

    # Pattern 1: FINE-GRAINED CONFUSION (Same semantic group)
    if top1_group == top2_group and top1_group != "Unknown":
        if top1_group == "Speed Limits":
            return {
                "decision": "AGREE",
                "confusion_type": "FINE_GRAINED",
                "reason": (
                    "Both predictions are speed limit signs (same circular shape, "
                    "red border). Confusion between numeric values is expected and "
                    "plausible. Low-risk fine-grained distinction."
                ),
                "risk_level": "LOW",
            }
        return {
            "decision": "AGREE",
            "confusion_type": "FINE_GRAINED",
            "reason": (
                f"Top predictions belong to {top1_group} category (same visual "
                f"structure). Within-group confusion is semantically reasonable."
            ),
            "risk_level": "LOW",
        }

    # Pattern 2: STRUCTURAL CONFLICT (Different shapes + tight margin)
    if (
        top1_group != top2_group
        and top1_group != "Unknown"
        and top2_group != "Unknown"
        and confidence > 0.85
        and margin < 0.15
    ):
        return {
            "decision": "DISAGREE",
            "confusion_type": "STRUCTURAL_CONFLICT",
            "reason": (
                f"Top-1 is {top1_group} but Top-2 is {top2_group}. These have "
                f"different geometric shapes. High confidence with cross-category "
                f"alternatives suggests overconfidence."
            ),
            "risk_level": "HIGH",
        }

    # Pattern 3: OUT-OF-DISTRIBUTION (Low margin + unrelated classes)
    if (
        margin < 0.15
        and confidence < 0.55
        and top1_group != top2_group
        and top1_group != "Unknown"
        and top2_group != "Unknown"
    ):
        return {
            "decision": "DISAGREE",
            "confusion_type": "OUT_OF_DISTRIBUTION",
            "reason": (
                f"Near-equal probabilities ({confidence:.2f} vs "
                f"{confidence - margin:.2f}) between unrelated categories "
                f"({top1_group} vs {top2_group}). Model appears lost — possible "
                f"out-of-distribution input."
            ),
            "risk_level": "CRITICAL",
        }

    # Pattern 4: SEMANTIC VALIDATION (Low confidence + same group)
    if confidence < 0.70 and top1_group == top2_group and top1_group != "Unknown":
        return {
            "decision": "AGREE",
            "confusion_type": "SEMANTIC_VALIDATED",
            "reason": (
                f"Although confidence is moderate ({confidence:.2f}), confusion "
                f"within {top1_group} category is semantically plausible. "
                f"Prediction is trustworthy despite lower confidence."
            ),
            "risk_level": "MEDIUM",
        }

    # Pattern 5: OVERCONFIDENCE (Very high conf + unrelated top-2)
    if confidence > 0.97 and top1_group != top2_group and top2_group != "Unknown":
        return {
            "decision": "DISAGREE",
            "confusion_type": "OVERCONFIDENCE",
            "reason": (
                f"Very high confidence ({confidence:.2f}) but Top-2 is from a "
                f"different category ({top2_group}). If model were truly certain, "
                f"alternatives should be from same group. Possible false confidence."
            ),
            "risk_level": "HIGH",
        }

    # Pattern 6: High entropy + same group = reasonable uncertainty
    if entropy > 0.5 and top1_group == top2_group:
        return {
            "decision": "AGREE",
            "confusion_type": "SEMANTIC_VALIDATED",
            "reason": (
                f"High entropy appropriately reflects genuine ambiguity within "
                f"{top1_group} category. Model correctly expresses uncertainty."
            ),
            "risk_level": "MEDIUM",
        }

    # Pattern 7: HIGH CONFIDENCE + LARGE MARGIN (dominant prediction)
    if confidence > 0.80 and margin > 0.5:
        return {
            "decision": "AGREE",
            "confusion_type": "DOMINANT_PREDICTION",
            "reason": (
                f"High confidence ({confidence:.2f}) with large margin "
                f"({margin:.2f}) to alternatives. CNN prediction is dominant "
                f"and trustworthy."
            ),
            "risk_level": "LOW",
        }

    # Pattern 8: MODERATE CONFIDENCE + REASONABLE MARGIN
    if confidence > 0.70 and margin > 0.3:
        return {
            "decision": "AGREE",
            "confusion_type": "CLEAR_WINNER",
            "reason": (
                f"Confidence ({confidence:.2f}) with meaningful margin "
                f"({margin:.2f}) indicates the model is discriminating well. "
                f"Cross-group alternatives at very low probability do not warrant concern."
            ),
            "risk_level": "LOW",
        }

    # Default: Truly ambiguous
    return {
        "decision": "UNCERTAIN",
        "confusion_type": "AMBIGUOUS",
        "reason": (
            f"Low confidence ({confidence:.2f}) with small margin ({margin:.2f}) "
            f"between {top1_group} and {top2_group}. Insufficient evidence for "
            f"a strong decision."
        ),
        "risk_level": "MEDIUM",
    }


# Convenience function
def call_llm_reviewer(
    cnn_output: Dict,
    api_key: str = None,
    model: str = "meta-llama/llama-3.1-8b-instruct:free",
) -> tuple:
    """
    Convenience function to call LLM reviewer.

    Args:
        cnn_output: CNN prediction output
        api_key: OpenRouter API key
        model: Model name

    Returns:
        (llm_response, prompt)
    """
    reviewer = LLMReviewer(api_key=api_key, model=model)
    return reviewer.review(cnn_output)


if __name__ == "__main__":
    print("Testing LLM Reviewer (Mock)...")

    test_output = {
        "predictions": [
            {"class_id": 2, "class_name": "Speed limit (50km/h)", "confidence": 0.85},
            {"class_id": 1, "class_name": "Speed limit (30km/h)", "confidence": 0.10},
            {"class_id": 3, "class_name": "Speed limit (60km/h)", "confidence": 0.03},
        ],
        "entropy": 0.45,
        "margin": 0.75,
    }

    response = llm_reviewer_mock(test_output)
    print(f"Decision: {response['decision']}")
    print(f"Type: {response['confusion_type']}")
    print(f"Reason: {response['reason']}")
    print(f"Risk: {response['risk_level']}")
