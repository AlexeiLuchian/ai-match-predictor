import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def generate_match_analysis(team_a, team_b, result, shap_vals, feature_cols):
    """
    Generates a natural language match analysis using a HuggingFace
    language model, informed by SHAP feature importance values.
    """
    shap_importance = sorted(
        zip(feature_cols, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    winner = team_a if result["sets_a"] > result["sets_b"] else team_b
    top_factors = ", ".join([f[0] for f in shap_importance])

    messages = [
        {
            "role": "system",
            "content": "You are a volleyball analyst. Write brief match analyses."
        },
        {
            "role": "user",
            "content": f"Analyze this volleyball match: {team_a} vs {team_b}, sets {result['sets_a']}-{result['sets_b']}, winner {winner}, key factors: {top_factors}. Write a brief analysis in 3-4 sentences."
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    return response