import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel

class LocalIA:
    def __init__(self, index_path="local_ia/index", model_name="HuggingFaceTB/SmolLM-135M-Instruct"):
        self.index_path = index_path
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embed_tokenizer = None
        self.embed_model = None
        self.workflows = []
        self.embeddings = []
        self._load_index()

    def _load_index(self):
        if os.path.exists(os.path.join(self.index_path, "workflows.json")):
            with open(os.path.join(self.index_path, "workflows.json"), 'r', encoding='utf-8') as f:
                self.workflows = json.load(f)
            self.embeddings = np.load(os.path.join(self.index_path, "embeddings.npy"))
            
            # Cargar modelo de embeddings
            emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embed_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
            self.embed_model = AutoModel.from_pretrained(emb_model_name)

    def _get_embedding(self, text):
        inputs = self.embed_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _load_llm(self):
        if self.model is None:
            print(f"Cargando modelo LLM local {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print("Modelo cargado.")

    def generate_workflow(self, body_dict):
        instruction = body_dict.get("userInstruction", "")
        current_workflow = body_dict.get("currentWorkflow", {})
        
        # 1. Buscar ejemplos similares en el indice local
        query = f"Instruccion: {instruction}"
        query_emb = self._get_embedding(query)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        best_idx = np.argmax(similarities)
        example_wf = self.workflows[best_idx]

        # 2. Construir prompt para el LLM local
        prompt = f"""
        Eres un asistente de workflows. 
        Contexto local (Ejemplo de workflow valido):
        {json.dumps(example_wf, indent=2)}

        Workflow actual:
        {json.dumps(current_workflow, indent=2)}

        Instruccion: {instruction}

        Devuelve SOLO el JSON del workflow actualizado.
        """
        
        self._load_llm()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Intentar extraer el JSON de la respuesta
        try:
            # Buscar el primer '{' y el ultimo '}'
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Error parseando JSON local: {e}")
        
        return None

if __name__ == "__main__":
    # Test
    ia = LocalIA()
    # test_body = ...
    # print(ia.generate_workflow(test_body))
