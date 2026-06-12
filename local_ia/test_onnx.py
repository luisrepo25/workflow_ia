"""Prueba rápida del modelo ONNX exportado."""
import json
import numpy as np
import onnxruntime as ort
from transformers import DistilBertTokenizerFast

# Cargar sesión ONNX
session = ort.InferenceSession("local_ia/models/recommender.onnx")
print("Inputs:", [(i.name, i.shape, i.type) for i in session.get_inputs()])
print("Outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])

# Cargar tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("local_ia/models/recommender")

# Cargar class mapping
with open("local_ia/models/class_mapping.json", encoding="utf-8") as f:
    mapping = json.load(f)

# Probar consultas
queries = [
    "Necesito registrarme en el seguro social",
    "Quiero hacer un tramite de prueba",
    "Como registro una empresa",
    "Ayuda con el proceso de finanzas",
    "Necesito aprobar un documento",
]

for text in queries:
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    result = session.run(["logits"], {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    })
    logits = result[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    best = int(np.argmax(probs))

    print(f"\n🔍 Consulta: '{text}'")
    print(f"   → {mapping[str(best)]['nombre']} ({mapping[str(best)]['codigo']})")
    print(f"   Confianza: {probs[best]*100:.1f}%")
    print(f"   Top-3:")
    for i in np.argsort(probs)[-3:][::-1]:
        print(f"     {i}: {mapping[str(i)]['nombre']} ({probs[i]*100:.1f}%)")
