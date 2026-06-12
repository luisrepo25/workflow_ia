"""
Generador de dataset sintético para el recomendador de trámites.
Lee datos.json y genera consultas de usuario etiquetadas con el workflow correspondiente.
"""
import json
import os
import random
from typing import Any

random.seed(42)

TEMPLATES_PREGUNTA = [
    "Necesito {nombre}",
    "Quiero {descripcion}",
    "Cómo hago para {descripcion}",
    "Trámite de {nombre}",
    "Me gustaría {descripcion}",
    "Hola, necesito {nombre}",
    "Quisiera saber sobre {nombre}",
    "Estoy interesado en {descripcion}",
    "Puedo {descripcion}",
    "Ayuda con {nombre}",
    "Cómo tramito {codigo}",
    "Información sobre {nombre}",
    "Procedimiento para {descripcion}",
    "Requiero {nombre}",
    "Solicito {descripcion}",
    "{nombre}",
    "Consulta: {descripcion}",
    "Qué necesito para {descripcion}",
    "Dónde puedo {descripcion}",
    "{descripcion}",
]

TEMPLATES_VARIACION = [
    "necesito hacer un {nombre}",
    "quiero realizar {descripcion}",
    "cómo solicito {nombre}",
    "pasos para {descripcion}",
    "ayuda con el trámite de {nombre}",
    "info sobre {codigo}",
    "quisiera información acerca de {nombre}",
    "hola buenos días necesito {descripcion}",
    "me pueden ayudar con {nombre}",
    "ando buscando cómo {descripcion}",
]

PALABRAS_ALEATORIAS = [
    "por favor", "gracias", "es urgente", "cuanto tiempo toma",
    "qué documentos necesito", "es para un familiar", "tengo dudas",
    "podría indicarme", "quiero saber más", "",
]


def _extract_workflows(datos_path: str) -> list[dict]:
    """Lee datos.json y extrae los workflows con información relevante."""
    with open(datos_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workflows = []
    for wf in data:
        nombre = wf.get("nombre", "")
        descripcion = wf.get("descripcion", "")
        codigo = wf.get("codigo", "")
        wf_id = wf.get("_id", {}).get("$oid", "")

        # Extraer descripción de nodos para enriquecer
        nodos = wf.get("nodes", [])
        actividades = [
            n.get("nombre", "")
            for n in nodos
            if n.get("tipo") in ("actividad", "inicio", "fin")
        ]

        # Extraer campos de formularios
        campos = []
        for n in nodos:
            form = n.get("form")
            if form:
                for c in form.get("campos", []):
                    campos.append(c.get("label", ""))

        workflows.append({
            "id": wf_id,
            "nombre": nombre,
            "descripcion": descripcion,
            "codigo": codigo,
            "actividades": actividades,
            "campos": campos,
        })

    return workflows


def _normalize_text(text: str) -> str:
    """Limpia y normaliza texto."""
    return text.strip().lower()


def generate_dataset(datos_path: str = "datos.json", samples_per_workflow: int = 30) -> list[dict]:
    """
    Genera dataset sintético de consultas de usuario etiquetadas.
    Cada ejemplo: {"text": str, "label": int, "workflow_id": str, "nombre": str}
    """
    workflows = _extract_workflows(datos_path)

    if not workflows:
        raise ValueError(f"No se encontraron workflows en {datos_path}")

    print(f"Generando dataset desde {len(workflows)} workflows...")

    dataset = []
    for idx, wf in enumerate(workflows):
        nombre = wf["nombre"] or wf["codigo"] or "trámite"
        descripcion = wf["descripcion"] or f"realizar {nombre}"
        codigo = wf["codigo"] or ""

        # Generar consultas usando plantillas
        all_templates = TEMPLATES_PREGUNTA + TEMPLATES_VARIACION
        for i in range(samples_per_workflow):
            template = random.choice(all_templates)
            query = template.format(
                nombre=nombre,
                descripcion=descripcion,
                codigo=codigo,
            )
            # Añadir palabras aleatorias al final
            sufijo = random.choice(PALABRAS_ALEATORIAS)
            if sufijo:
                query = f"{query} {sufijo}"

            dataset.append({
                "text": query,
                "label": idx,
                "workflow_id": wf["id"],
                "workflow_nombre": nombre,
            })

    # Mezclar
    random.shuffle(dataset)

    print(f"Dataset generado: {len(dataset)} ejemplos, {len(workflows)} clases.")
    for i, wf in enumerate(workflows):
        count = sum(1 for d in dataset if d["label"] == i)
        print(f"  Clase {i}: '{wf['nombre']}' ({wf['codigo']}) -> {count} ejemplos")

    return dataset


def save_dataset(dataset: list[dict], output_path: str = "local_ia/models/dataset.json"):
    """Guarda el dataset en JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset guardado en {output_path}")


def save_class_mapping(workflows: list[dict], output_path: str = "local_ia/models/class_mapping.json"):
    """Guarda el mapeo clase -> workflow_id para usarlo en inferencia."""
    mapping = {}
    for idx, wf in enumerate(workflows):
        mapping[str(idx)] = {
            "workflow_id": wf["id"],
            "nombre": wf["nombre"],
            "codigo": wf["codigo"],
            "descripcion": wf["descripcion"],
        }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Class mapping guardado en {output_path}")
    return mapping


if __name__ == "__main__":
    ds = generate_dataset("datos.json", samples_per_workflow=30)
    save_dataset(ds)
    workflows = _extract_workflows("datos.json")
    save_class_mapping(workflows)
