import ollama

model = 'gemma3:270m'
question = 'What is the capital of the United States of America?'

def generate_response(model_name, prompt_question):
    response = ollama.generate(model=model_name, 
                               prompt=prompt_question)
    return response['response']

answer = generate_response(model, question)
print(question)
print(answer)