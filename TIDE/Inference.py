import os
import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# =========================
# CONFIGURATION
# =========================
ADAPTER_CHECKPOINT = "Adapter_best_checkpoint-285"
TEST_DATA_PATH = "data/test.jsonl"
OUTPUT_FOLDER = "Tide_outputs"
PARTIAL_SAVE_EVERY = 10

checkpoint_num = os.path.basename(ADAPTER_CHECKPOINT).split("-")[-1]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_JSON_PATH = os.path.join(
    OUTPUT_FOLDER, f"predictions_ckpt_{checkpoint_num}.json"
)
PARTIAL_JSON_PATH = os.path.join(
    OUTPUT_FOLDER, f"partial_predictions_ckpt_{checkpoint_num}.json"
)

# =========================
# MODEL LOADING
# =========================
def load_model_and_tokenizer(adapter_checkpoint):
    peft_config = PeftConfig.from_pretrained(adapter_checkpoint)
    base_model_id = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.padding_side = "left"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16,
        device_map="auto"
    )
    model = base_model
    #model = PeftModel.from_pretrained(base_model, adapter_checkpoint)
    return model.eval(), tokenizer


# =========================
# PARSING HELPERS
# =========================
def extract_paper_id(user_content):
    try:
        return user_content.split("Paper ID: ")[1].split("\n")[0].strip()
    except:
        return "UNKNOWN"


def parse_model_output(raw_output):
    """JSON-first parsing, regex fallback"""
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            return parsed
    except:
        pass

    contradictions = []
    pattern = r'\{[^{}]*"contradiction"[^{}]*"aspect"[^{}]*"intensity"[^{}]*\}'
    matches = re.findall(pattern, raw_output, re.DOTALL)

    for m in matches:
        try:
            contradictions.append(json.loads(m))
        except:
            continue

    return contradictions


# =========================
# INFERENCE
# =========================
def run_inference(model, tokenizer, test_data_path, output_path):

    with open(test_data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    all_predictions = {}

    for idx, item in tqdm(enumerate(data), total=len(data), desc="Inference"):
        try:
            messages = item["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")["content"]

            paper_id = extract_paper_id(user_msg)

            input_messages = [m for m in messages if m["role"] != "assistant"]

            tokenized = tokenizer.apply_chat_template(
                input_messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )

            input_ids = tokenized.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=768,
                    num_beams=2,
                    temperature=0.01,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            decoded_output = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )
            raw_output = decoded_output.strip()

            if not raw_output:
                contradictions = []
            else:
                contradictions = parse_model_output(raw_output)


            all_predictions[paper_id] = {
                "analysis": contradictions,
                "raw_output": raw_output
            }

            if idx % PARTIAL_SAVE_EVERY == 0:
                with open(PARTIAL_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, indent=2, ensure_ascii=False)

                print(f"\nSample {idx} | Paper ID: {paper_id}")
                print(f"Predicted contradictions: {len(contradictions)}")

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"\nFinal results saved to {output_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(ADAPTER_CHECKPOINT)
    run_inference(model, tokenizer, TEST_DATA_PATH, OUTPUT_JSON_PATH)