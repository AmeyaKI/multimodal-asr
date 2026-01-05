import ollama

def query(question, model_name='gemma3:270m'):
    response = ollama.generate(model=model_name, 
                               prompt=question)
    return response['response']


def main():
    model = 'gemma3:270m'
    question = 'What is the capital of the United States of America?'
    answer = query(question)
    print(question)
    print(answer)
    
if __name__ == '__main__':
    main()
    

    
# This answers the question, but  need for LLM to detail assistant response
# https://ai.plainenglish.io/building-a-simple-computer-use-ai-agent-with-gpt-4-and-applescript-1b0eec60dfd1