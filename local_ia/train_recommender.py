"""
Entrenamiento del modelo de recomendación de trámites usando DistilBERT Multilingual.
Exporta a ONNX para usar en Flutter via onnxruntime.
"""
import json
import os
import sys
import argparse
import torch
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


MODEL_NAME = "distilbert-base-multilingual-cased"
MODELS_DIR = "local_ia/models"
DATASET_PATH = os.path.join(MODELS_DIR, "dataset.json")
CLASS_MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
OUTPUT_DIR = os.path.join(MODELS_DIR, "checkpoint")
ONNX_PATH = os.path.join(MODELS_DIR, "recommender.onnx")
PT_PATH = os.path.join(MODELS_DIR, "recommender.pt")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")


def load_dataset(json_path: str) -> tuple[list[str], list[int], dict]:
    """Carga el dataset desde JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]

    # Cargar class mapping
    mapping_path = os.path.join(os.path.dirname(json_path), "class_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            class_mapping = json.load(f)
    else:
        class_mapping = {}

    num_classes = len(set(labels))
    print(f"Dataset cargado: {len(texts)} ejemplos, {num_classes} clases")
    print(f"Clases: {class_mapping}")

    return texts, labels, class_mapping, num_classes


def tokenize_function(examples, tokenizer, max_length=128):
    """Tokenización para DistilBERT."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def compute_metrics(eval_pred):
    """Métricas de evaluación."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


def train_model(
    dataset_path: str = DATASET_PATH,
    model_name: str = MODEL_NAME,
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    val_split: float = 0.2,
):
    """Entrena el clasificador DistilBERT para recomendación de trámites."""

    # ── Cargar dataset ──
    texts, labels, class_mapping, num_classes = load_dataset(dataset_path)

    # ── Cargar tokenizer y modelo ──
    print(f"\nCargando modelo: {model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    )

    # ── Split train/val ──
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_split, random_state=42, stratify=labels
    )

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

    # ── Crear datasets HuggingFace ──
    train_data = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_data = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # Tokenizar
    train_data = train_data.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )
    val_data = val_data.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=["text"],
    )

    # Renombrar label
    train_data = train_data.rename_column("label", "labels")
    val_data = val_data.rename_column("label", "labels")

    # Formato torch
    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Entrenar ──
    print("\n🚀 Iniciando entrenamiento...")
    trainer.train()

    # ── Evaluación final ──
    print("\n📊 Evaluación final:")
    eval_result = trainer.evaluate()
    print(json.dumps(eval_result, indent=2))

    # ── Guardar modelo PyTorch ──
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_pretrained(PT_PATH.replace(".pt", ""))
    tokenizer.save_pretrained(PT_PATH.replace(".pt", ""))
    torch.save(model.state_dict(), PT_PATH)
    print(f"\n💾 Modelo PyTorch guardado en: {PT_PATH}")
    print(f"💾 Tokenizer guardado en: {PT_PATH.replace('.pt', '')}")

    # ── Reporte ──
    report = {
        "model": model_name,
        "num_classes": num_classes,
        "num_train_samples": len(train_texts),
        "num_val_samples": len(val_texts),
        "eval_results": eval_result,
        "class_mapping": class_mapping,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📄 Reporte guardado en: {REPORT_PATH}")

    return model, tokenizer, class_mapping


def export_to_onnx(model, tokenizer, num_classes: int, output_path: str = ONNX_PATH):
    """Exporta el modelo PyTorch a ONNX para usarlo en Flutter."""
    print(f"\n🔄 Exportando a ONNX: {output_path}")

    model.eval()
    model.cpu()

    # Crear input dummy (batch=1, seq=128)
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128))
    dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)

    # Exportar a ONNX
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    except Exception as e:
        print(f"⚠️ Error con opset 14, intentando con opset 18: {e}")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=18,
            do_constant_folding=True,
        )

    # Verificar modelo ONNX
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX exportado correctamente: {output_path}")
    print(f"   Tamaño: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"   Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"   Outputs: {[out.name for out in onnx_model.graph.output]}")

    return output_path


def save_serving_info(output_path: str, onnx_path: str, pt_path: str, class_mapping: dict):
    """Guarda info de serving para el endpoint."""
    info = {
        "model_name": MODEL_NAME,
        "num_classes": len(class_mapping),
        "onnx_path": onnx_path,
        "pt_path": pt_path,
        "max_seq_length": 128,
        "class_mapping": class_mapping,
    }
    info_path = os.path.join(MODELS_DIR, "serving_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"📄 Serving info guardada en: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Entrenar recomendador de trámites")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Ruta al dataset JSON")
    parser.add_argument("--model", default=MODEL_NAME, help="Modelo base de HuggingFace")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fracción de validación")
    parser.add_argument("--skip-onnx", action="store_true", help="Saltar exportación ONNX")
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 ENTRENAMIENTO DEL RECOMENDADOR DE TRÁMITES")
    print("=" * 60)
    print(f"Modelo: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Val split: {args.val_split}")
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 1. Entrenar
    model, tokenizer, class_mapping = train_model(
        dataset_path=args.dataset,
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
    )

    # 2. Exportar a ONNX
    if not args.skip_onnx:
        export_to_onnx(model, tokenizer, len(class_mapping))

    # 3. Guardar info de serving
    save_serving_info(
        OUTPUT_DIR,
        ONNX_PATH,
        PT_PATH,
        class_mapping,
    )

    print("\n✅ Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
