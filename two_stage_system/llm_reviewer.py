"""
LLM Reviewer for Traffic Sign Recognition
Uses OpenRouter API to verify CNN predictions
"""

import json
import os
from typing import Dict, List
import requests


class LLMReviewer:
    """
    LLM-based output reviewer for traffic sign classification
    - Text-only input (NO images)
    - Structured JSON output
    - Supports OpenRouter API (local or cloud models)
    """
    
    def __init__(self, api_key: str = None, model: str = "meta-llama/llama-3.1-8b-instruct"):
        """
        Initialize LLM Reviewer
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model to use (default: Llama 3.1 8B)
                   Other options:
                   - "meta-llama/llama-3.1-70b-instruct" (more powerful)
                   - "anthropic/claude-3.5-sonnet" (best reasoning)
                   - "openai/gpt-4-turbo" (OpenAI via OpenRouter)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter.")
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def create_prompt(self, cnn_output: Dict) -> str:
        """
        Create structured text prompt for LLM
        NO IMAGE - Text only
        """
        preds = cnn_output['predictions']
        
        prompt = f"""Primary Model: CNN with Attention Ensemble (99.72% accuracy on GTSRB)

Prediction Summary:
Top-1: {preds[0]['class_name']} (confidence = {preds[0]['confidence']:.3f})
Top-2: {preds[1]['class_name']} (confidence = {preds[1]['confidence']:.3f})
Top-3: {preds[2]['class_name']} (confidence = {preds[2]['confidence']:.3f})

Uncertainty Metrics:
Entropy: {"High" if cnn_output['entropy'] > 0.5 else "Medium" if cnn_output['entropy'] > 0.3 else "Low"} ({cnn_output['entropy']:.3f})
Margin (Top1 - Top2): {cnn_output['margin']:.3f}

Task:
Verify whether the Top-1 prediction is logically consistent with the confidence scores and class semantics.
Consider:
1. Is the Top-1 confidence significantly higher than alternatives?
2. Are the Top-3 classes semantically similar (could indicate confusion)?
3. Does the entropy level match the confidence?

Choose ONLY one decision: AGREE, DISAGREE, or UNCERTAIN.
Provide a brief logical explanation (1-2 sentences).

Output STRICTLY in JSON format:
{{
  "decision": "AGREE | DISAGREE | UNCERTAIN",
  "reason": "Brief logical explanation",
  "risk_level": "LOW | MEDIUM | HIGH"
}}"""
        
        return prompt
    
    def call_api(self, prompt: str) -> Dict:
        """
        Call OpenRouter API
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/traffic-sign-recognition",  # Optional
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a traffic sign recognition verifier. Analyze CNN predictions for logical consistency. Output ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent output
            "max_tokens": 500,
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            llm_output = result['choices'][0]['message']['content']
            
            # Parse JSON from LLM response
            # Sometimes LLMs add markdown code blocks, so clean it
            if '```json' in llm_output:
                llm_output = llm_output.split('```json')[1].split('```')[0].strip()
            elif '```' in llm_output:
                llm_output = llm_output.split('```')[1].split('```')[0].strip()
            
            parsed_response = json.loads(llm_output)
            
            # Validate required fields
            required_fields = ['decision', 'reason', 'risk_level']
            if not all(field in parsed_response for field in required_fields):
                raise ValueError("LLM response missing required fields")
            
            # Validate decision value
            if parsed_response['decision'] not in ['AGREE', 'DISAGREE', 'UNCERTAIN']:
                raise ValueError(f"Invalid decision: {parsed_response['decision']}")
            
            # Validate risk_level value
            if parsed_response['risk_level'] not in ['LOW', 'MEDIUM', 'HIGH']:
                raise ValueError(f"Invalid risk_level: {parsed_response['risk_level']}")
            
            return parsed_response
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return {
                "decision": "UNCERTAIN",
                "reason": f"API call failed: {str(e)}",
                "risk_level": "MEDIUM"
            }
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"LLM Output: {llm_output}")
            return {
                "decision": "UNCERTAIN",
                "reason": "LLM returned invalid JSON format",
                "risk_level": "MEDIUM"
            }
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return {
                "decision": "UNCERTAIN",
                "reason": f"Unexpected error: {str(e)}",
                "risk_level": "HIGH"
            }
    
    def review(self, cnn_output: Dict) -> tuple[Dict, str]:
        """
        Review CNN output
        
        Args:
            cnn_output: Dictionary containing predictions and uncertainty metrics
            
        Returns:
            (llm_response, prompt): LLM decision and the prompt used
        """
        prompt = self.create_prompt(cnn_output)
        llm_response = self.call_api(prompt)
        return llm_response, prompt


# Convenience function for backward compatibility
def call_llm_reviewer(cnn_output: Dict, api_key: str = None, model: str = "meta-llama/llama-3.1-8b-instruct") -> tuple[Dict, str]:
    """
    Convenience function to call LLM reviewer
    
    Args:
        cnn_output: CNN prediction output
        api_key: OpenRouter API key
        model: Model name (default: Llama 3.1 8B)
    
    Returns:
        (llm_response, prompt)
    """
    reviewer = LLMReviewer(api_key=api_key, model=model)
    return reviewer.review(cnn_output)


if __name__ == "__main__":
    # Test the reviewer
    print("🧪 Testing LLM Reviewer...")
    
    # Mock CNN output for testing
    test_output = {
        'predictions': [
            {'class_name': 'Speed limit 50 km/h', 'confidence': 0.85},
            {'class_name': 'Speed limit 30 km/h', 'confidence': 0.10},
            {'class_name': 'Speed limit 70 km/h', 'confidence': 0.03}
        ],
        'entropy': 0.45,
        'margin': 0.75
    }
    
    try:
        reviewer = LLMReviewer()  # Will use OPENROUTER_API_KEY env var
        response, prompt = reviewer.review(test_output)
        print("\n✅ Test successful!")
        print(f"Decision: {response['decision']}")
        print(f"Reason: {response['reason']}")
        print(f"Risk: {response['risk_level']}")
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print("Set OPENROUTER_API_KEY environment variable to test")
