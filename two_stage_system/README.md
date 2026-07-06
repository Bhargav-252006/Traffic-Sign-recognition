# Two-Stage Traffic Sign Recognition System
## CNN (99.72%) + LLM Output Reviewer

Clean implementation with modular architecture.

## 📁 Files

- **`cnn_llm_system.ipynb`** - Main notebook (simplified, clean)
- **`model_architectures.py`** - CNN model definitions
- **`llm_reviewer.py`** - LLM reviewer with OpenRouter API
- **`.env.example`** - Configuration template

## 🚀 Setup

### 1. Get OpenRouter API Key

```bash
# Sign up at https://openrouter.ai/
# Get your API key from https://openrouter.ai/keys
```

### 2. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your_key_here"
```

**Windows (CMD):**
```cmd
set OPENROUTER_API_KEY=your_key_here
```

**Linux/Mac:**
```bash
export OPENROUTER_API_KEY="your_key_here"
```

### 3. Install Dependencies

```bash
pip install requests torch torchvision pillow
```

## 💡 Usage

### In Notebook:

```python
from llm_reviewer import LLMReviewer

# Create reviewer
reviewer = LLMReviewer(
    api_key="your_key",  # or use env var
    model="meta-llama/llama-3.1-8b-instruct"  # or other model
)

# Use in two-stage prediction
result = two_stage_predict(image_path, use_llm=True)
```

### Standalone Script:

```python
from llm_reviewer import call_llm_reviewer

# Review CNN output
llm_response, prompt = call_llm_reviewer(
    cnn_output=cnn_prediction,
    api_key="your_key",
    model="meta-llama/llama-3.1-8b-instruct"
)

print(f"Decision: {llm_response['decision']}")
print(f"Reason: {llm_response['reason']}")
```

## 🎯 Available Models

| Model | Speed | Cost | Reasoning |
|-------|-------|------|-----------|
| `meta-llama/llama-3.1-8b-instruct` | ⚡⚡⚡ | $ | Good |
| `meta-llama/llama-3.1-70b-instruct` | ⚡⚡ | $$ | Better |
| `anthropic/claude-3.5-sonnet` | ⚡ | $$$ | Best |
| `openai/gpt-4-turbo` | ⚡ | $$$$ | Excellent |

## 🔧 Architecture

```
Image → CNN Ensemble (99.72%) → Confidence Gate → LLM Reviewer → Final Decision
                                      ↓
                               High Confidence?
                                      ↓
                                  Yes → Accept
                                  No  → LLM Review
```

## 📊 Key Features

- ✅ Loads saved ensemble model (no retraining)
- ✅ LLM sees TEXT ONLY (no image processing)
- ✅ Structured JSON output from LLM
- ✅ Multiple LLM providers via OpenRouter
- ✅ Local model support (Llama 3.1)
- ✅ Error handling and fallbacks
- ✅ Modular and maintainable code

## 🧪 Testing

Test the LLM reviewer:

```bash
cd two_stage_system
python llm_reviewer.py
```

## 📝 Notes

- LLM is only called for uncertain predictions (saves cost)
- Default confidence threshold: 95%
- LLM cannot invent labels (closed decision space)
- All decisions are logged and traceable
