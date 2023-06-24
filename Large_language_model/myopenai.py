import openai

class MyOpenAI:
    def __init__(self, prompt, model, n, maxtokens, temperature, key):
        self.prompt = prompt
        self.model = model
        self.key = key
        self.n = n
        self.maxtokens = maxtokens
        self.temperature = temperature
        openai.api_key = self.key
    
    def callgpt(self):
        response = openai.Completion.create(
            model = self.model,
            prompt = self.prompt,
            temperature = self.temperature,
            max_tokens = self.maxtokens,
            n = self.n,
        )
        resp = []
        for a in range(len(response.choices)):
            resp.append(response.choices[a].text.replace('\n',''))
        return resp
    
