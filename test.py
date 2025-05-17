from pyllms import LLM

# Initialize the model
model = LLM(provider="openai")  # or your preferred provider

# Test the model
response = model.complete("What is 5 + 5?")
print(response.text)