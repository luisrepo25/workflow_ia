import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class WorkflowIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.workflows = []
        self.embeddings = []

    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def train(self, json_path):
        print(f"Entrenando indice local con {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.workflows = data
        texts = []
        for wf in data:
            # Crear una representacion textual del workflow para buscar
            desc = f"Nombre: {wf.get('nombre', '')}. Descripcion: {wf.get('descripcion', '')}. Nodos: {', '.join([n.get('nombre', '') for n in wf.get('nodes', [])])}"
            texts.append(desc)
        
        for text in texts:
            self.embeddings.append(self._get_embedding(text))
        
        self.embeddings = np.vstack(self.embeddings)
        print("Entrenamiento completado.")

    def search(self, query, k=1):
        query_emb = self._get_embedding(query)
        similarities = cosine_similarity(query_emb, self.embeddings)[0]
        indices = np.argsort(similarities)[-k:][::-1]
        return [self.workflows[i] for i in indices]

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "workflows.json"), 'w', encoding='utf-8') as f:
            json.dump(self.workflows, f)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)

if __name__ == "__main__":
    index = WorkflowIndex()
    index.train("datos.json")
    index.save("local_ia/index")
