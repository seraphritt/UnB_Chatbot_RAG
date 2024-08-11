from transformers import pipeline

generator = pipeline("text-generation", model="nicholasKluge/TeenyTinyLlama-460m")

completions  = generator("Então temos que 2 mais 2 são", num_return_sequences=2, max_new_tokens=100)

#   for comp in completions:
  #  print(f"🤖 {comp['generated_text']}")
for comp in completions:
    print(f"{comp['generated_text']}")
    break
