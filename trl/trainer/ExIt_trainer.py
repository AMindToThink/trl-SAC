def short_condition(x):
    return len(x) < 20
class RejectionSamplingExpert:
    def __init__(self, model, condition):
        self.model = model
        self.condition = condition
    
    def generate_surplus(self, **kwargs):
        # May generate a few more completions than requested. 
        num_return_sequences = kwargs.pop("num_return_sequences", 1)
        legal_generations = []
        while len(legal_generations) < num_return_sequences:
            generated_ids = self.model.generate(**kwargs)
            for completion in generated_ids:
                legal_generations.append(completion)
        return legal_generations
    
    def generate(self, **kwargs):
        # Generates exactly the amount requested by discarding some completions.
        return self.generate_surplus(**kwargs)[:kwargs['num_return_sequences']]
