"""
Módulo de inferencia para el recomendador de trámites.
Carga el modelo ONNX y predice el workflow más adecuado para una consulta.
Funciona tanto en servidor (Python) como es exportable a Flutter.
"""
import json
import os
import numpy as np

MODELS_DIR = "local_ia/models"
ONNX_PATH = os.path.join(MODELS_DIR, "recommender_int8.onnx")
# Fallback si no existe el int8
if not os.path.exists(ONNX_PATH):
    ONNX_PATH = os.path.join(MODELS_DIR, "recommender.onnx")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
SERVING_INFO_PATH = os.path.join(MODELS_DIR, "serving_info.json")


class WorkflowRecommender:
    """
    Recomendador de trámites usando modelo DistilBERT fine-tuneado.
    Soporta dos backends:
    - 'onnx': ONNX Runtime (rápido, portable)
    - 'torch': PyTorch (para desarrollo/debug)
    """

    def __init__(
        self,
        model_path: str = ONNX_PATH,
        class_mapping_path: str = CLASS_MAPPING_PATH,
        backend: str = "onnx",
    ):
        self.model_path = model_path
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.num_classes = len(self.class_mapping)
        self.backend = backend
        self.model = None
        self.tokenizer = None
        self.session = None

    def _load_class_mapping(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def load(self):
        """Carga el modelo según el backend configurado."""
        if self.backend == "onnx":
            self._load_onnx()
        elif self.backend == "torch":
            self._load_torch()
        else:
            raise ValueError(f"Backend no soportado: {self.backend}")

    def _load_onnx(self):
        """Carga modelo ONNX Runtime."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime no está instalado. "
                "Ejecuta: pip install onnxruntime"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado en {self.model_path}. "
                "Ejecuta primero local_ia/train_recommender.py"
            )

        # Cargar tokenizer desde HuggingFace
        from transformers import DistilBertTokenizerFast
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-multilingual-cased"
        )

        # Crear sesión ONNX
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"✅ Modelo ONNX cargado: {self.model_path}")
        print(f"   Providers: {providers}")
        print(f"   Inputs: {self.input_names}")
        print(f"   Outputs: {self.output_names}")

    def _load_torch(self):
        """Carga modelo PyTorch (para desarrollo)."""
        import torch
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

        pt_path = self.model_path.replace(".onnx", ".pt")
        model_dir = self.model_path.replace(".onnx", "")

        if os.path.exists(model_dir):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        elif os.path.exists(pt_path):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-multilingual-cased"
            )
            self.model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-multilingual-cased",
                num_labels=self.num_classes,
            )
            self.model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        else:
            raise FileNotFoundError(
                f"No se encontró modelo en {model_dir} ni {pt_path}"
            )

        self.model.eval()
        print(f"✅ Modelo PyTorch cargado")

    def predict(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Predice el/los workflow(s) más relevantes para una consulta.

        Args:
            query: Consulta del usuario en lenguaje natural
            top_k: Número de resultados a devolver

        Returns:
            Lista de dicts con workflow recomendado y score de confianza
        """
        if self.session is None and self.model is None:
            self.load()

        # Tokenizar
        inputs = self.tokenizer(
            query,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np",  # numpy para ONNX
        )

        if self.backend == "onnx":
            return self._predict_onnx(inputs, top_k)
        else:
            return self._predict_torch(inputs, top_k)

    def _predict_onnx(self, inputs, top_k: int) -> list[dict]:
        """Inferencia con ONNX Runtime."""
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        logits = ort_outputs[0][0]

        return self._process_logits(logits, top_k)

    def _predict_torch(self, inputs, top_k: int) -> list[dict]:
        """Inferencia con PyTorch."""
        import torch
        with torch.no_grad():
            outputs = self.model(
                input_ids=torch.tensor(inputs["input_ids"]),
                attention_mask=torch.tensor(inputs["attention_mask"]),
            )
        logits = outputs.logits[0].cpu().numpy()
        return self._process_logits(logits, top_k)

    def _process_logits(self, logits: np.ndarray, top_k: int) -> list[dict]:
        """Convierte logits en resultados con class mapping."""
        import torch.nn.functional as F
        import torch

        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            class_key = str(idx)
            workflow_info = self.class_mapping.get(class_key, {})
            results.append({
                "clase": int(idx),
                "confianza": float(probs[idx]),
                "confianza_pct": round(float(probs[idx]) * 100, 2),
                "workflow_id": workflow_info.get("workflow_id", ""),
                "nombre": workflow_info.get("nombre", ""),
                "codigo": workflow_info.get("codigo", ""),
                "descripcion": workflow_info.get("descripcion", ""),
            })

        return results

    def predict_best(self, query: str) -> dict | None:
        """Devuelve solo la mejor predicción."""
        results = self.predict(query, top_k=1)
        return results[0] if results else None


# Singleton para usar en FastAPI
_recommender_instance: WorkflowRecommender | None = None


def get_recommender() -> WorkflowRecommender:
    """Obtiene o crea la instancia singleton del recomendador."""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = WorkflowRecommender()
        try:
            _recommender_instance.load()
        except (FileNotFoundError, ImportError) as e:
            print(f"⚠️ No se pudo cargar el recomendador: {e}")
            print("   El endpoint /recomendar usará el modo embedding-based como fallback.")
            _recommender_instance = None
    return _recommender_instance


if __name__ == "__main__":
    # Test rápido
    recommender = WorkflowRecommender()
    recommender.load()

    queries = [
        "Necesito registrarme en el seguro social",
        "Quiero hacer un trámite de prueba",
        "Cómo registro una empresa",
        "Ayuda con decisión de personal",
    ]

    for q in queries:
        print(f"\n🔍 Consulta: '{q}'")
        results = recommender.predict(q)
        for r in results:
            print(f"   → {r['nombre']} ({r['codigo']}) "
                  f"confianza: {r['confianza_pct']}%")
