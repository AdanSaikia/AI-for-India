from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define the model name and path
model_name = "summarisation_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarise(text:str, max_length:int=1024):

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)

    # Set generation parameters
    generation_params = {
        "max_length": 104,
        "min_length": 64,
        "early_stopping": True,
        "num_beams": 4,
        "length_penalty": 2.0,
        "no_repeat_ngram_size": 3,
        "forced_bos_token_id": 0
    }

    # Generate the summary
    summary_ids = model.generate(**inputs, **generation_params)

    # Decode and print the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
  
