import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU non disponibile, verrà usata la CPU.")

!pip install transformers sentencepiece accelerate -q
!pip install bitsandbytes -q

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model_name_new = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer_new = AutoTokenizer.from_pretrained(model_name_new)

if tokenizer_new.pad_token is None:
    tokenizer_new.pad_token = tokenizer_new.eos_token

model_new = AutoModelForCausalLM.from_pretrained(
    model_name_new,
    quantization_config=bnb_config,
    device_map={"": 0}
)

model_new.eval()

messages = [
    {"role": "user", "content": "Qual è il contrario di 'grande'?"}
]

formatted_prompt = tokenizer_new.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids_new = tokenizer_new.encode(formatted_prompt, return_tensors="pt").to(model_new.device)

with torch.no_grad():
    output_new = model_new.generate(
        input_ids_new,
        max_new_tokens=100,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer_new.pad_token_id
    )

generated_text_new = tokenizer_new.decode(output_new[0][len(input_ids_new[0]):], skip_special_tokens=True)

print(generated_text_new)
