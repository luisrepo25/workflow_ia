"""
Cuantización del modelo a int8 usando optimum (HuggingFace).
Reduce el tamaño ~4x (de 517MB a ~130MB) con mínima pérdida de precisión.
"""
import os
import json
import shutil
import numpy as np


MODELS_DIR = "local_ia/models"
ONNX_FP32_DIR = os.path.join(MODELS_DIR, "onnx_optimum")
ONNX_INT8_DIR = os.path.join(MODELS_DIR, "onnx_optimum_int8")
ONNX_INT8_PATH = os.path.join(MODELS_DIR, "recommender_int8.onnx")
ONNX_FP16_PATH = os.path.join(MODELS_DIR, "recommender_fp16.onnx")


def quantize_int8():
    """Exporta a ONNX con optimum y cuantiza a int8."""
    print("=" * 60)
    print("🧮 CUANTIZACIÓN INT8 DEL MODELO (optimum)")
    print("=" * 60)

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    model_id = os.path.join(MODELS_DIR, "recommender")

    if not os.path.exists(model_id):
        print(f"❌ No se encuentra {model_id}. Ejecuta primero train_recommender.py")
        return False

    # Limpiar directorios previos
    for d in [ONNX_FP32_DIR, ONNX_INT8_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # 1. Exportar a ONNX FP32
    print("\n📦 Exportando a ONNX FP32 con optimum...")
    model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(ONNX_FP32_DIR)
    tokenizer.save_pretrained(ONNX_FP32_DIR)

    size_fp32 = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fn in os.walk(ONNX_FP32_DIR) for f in fn)
    print(f"   Tamaño FP32: {size_fp32 / 1024 / 1024:.0f} MB")

    # 2. Cuantizar a int8
    print("\n🔧 Cuantizando a int8 (ARM64 compatible)...")
    quantizer = ORTQuantizer.from_pretrained(ONNX_FP32_DIR)
    qconfig = AutoQuantizationConfig.arm64(is_static=False)
    quantizer.quantize(save_dir=ONNX_INT8_DIR, quantization_config=qconfig)

    size_int8 = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, _, fn in os.walk(ONNX_INT8_DIR) for f in fn)
    reduction = 100 * (1 - size_int8 / size_fp32)

    print(f"\n✅ Cuantización completada!")
    print(f"   Tamaño INT8: {size_int8 / 1024 / 1024:.0f} MB")
    print(f"   Reducción: {reduction:.0f}%")

    # 3. Copiar al directorio principal
    onnx_file = os.path.join(ONNX_INT8_DIR, "model_quantized.onnx")
    if os.path.exists(onnx_file):
        shutil.copy2(onnx_file, ONNX_INT8_PATH)
        print(f"   Copiado a: {ONNX_INT8_PATH}")

    return True


def export_fp16():
    """Exporta modelo PyTorch a ONNX FP16 (mitad de tamaño)."""
    print("\n" + "=" * 60)
    print("🧮 EXPORTACIÓN A ONNX FP16")
    print("=" * 60)

    import torch
    from transformers import DistilBertForSequenceClassification

    model_dir = os.path.join(MODELS_DIR, "recommender")
    if not os.path.exists(model_dir):
        print(f"❌ No se encuentra {model_dir}")
        return False

    print("\nCargando modelo PyTorch...")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval().cpu().half()

    print("Exportando a ONNX FP16...")
    dummy_ids = torch.randint(0, 1000, (1, 128)).long()
    dummy_mask = torch.ones((1, 128), dtype=torch.long)

    torch.onnx.export(
        model, (dummy_ids, dummy_mask), ONNX_FP16_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )

    if os.path.exists(ONNX_FP16_PATH):
        total = os.path.getsize(ONNX_FP16_PATH)
        if os.path.exists(ONNX_FP16_PATH + ".data"):
            total += os.path.getsize(ONNX_FP16_PATH + ".data")
        print(f"\n✅ FP16 listo: {total / 1024 / 1024:.0f} MB")
        return True
    return False


def compare_sizes():
    """Compara tamaños de todas las versiones."""
    print("\n" + "=" * 60)
    print("📊 COMPARATIVA DE TAMAÑOS")
    print("=" * 60)

    entries = [
        ("FP32 (original)", os.path.join(MODELS_DIR, "recommender.onnx")),
        ("FP32 (optimum)", ONNX_FP32_DIR),
        ("FP16", ONNX_FP16_PATH),
        ("INT8 (optimum)", ONNX_INT8_DIR),
    ]

    for name, path in entries:
        if os.path.isdir(path):
            total = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fn in os.walk(path) for f in fn)
        elif os.path.exists(path):
            total = os.path.getsize(path)
            if os.path.exists(path + ".data"):
                total += os.path.getsize(path + ".data")
        else:
            print(f"  {name:25s}: ❌ No encontrado")
            continue

        print(f"  {name:25s}: {total / 1024 / 1024:7.0f} MB")


def test_quantized():
    """Prueba el modelo cuantizado."""
    print("\n" + "=" * 60)
    print("🧪 PROBANDO MODELO INT8")
    print("=" * 60)

    if not os.path.exists(ONNX_INT8_PATH):
        print("❌ No hay modelo int8")
        return

    import onnxruntime as ort
    from transformers import AutoTokenizer

    session = ort.InferenceSession(ONNX_INT8_PATH, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "recommender"))

    queries = [
        "Necesito registrarme en el seguro social",
        "Quiero hacer un tramite de prueba",
        "Ayuda con el proceso de finanzas",
    ]

    with open(os.path.join(MODELS_DIR, "class_mapping.json"), encoding="utf-8") as f:
        mapping = json.load(f)

    for text in queries:
        inputs = tokenizer(text, return_tensors="np", padding="max_length",
                          truncation=True, max_length=128)
        outputs = session.run(["logits"], {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        })
        logits = outputs[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        best = int(np.argmax(probs))

        print(f"\n🔍 '{text}'")
        print(f"   → {mapping[str(best)]['nombre']} ({probs[best]*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 OPTIMIZACIÓN DEL MODELO DE RECOMENDACIÓN")
    print("=" * 60)

    hacer_int8 = input("\n¿Cuantizar a INT8? (s/N): ").lower() == "s"
    hacer_fp16 = input("¿Exportar a FP16? (s/N): ").lower() == "s"

    if hacer_int8:
        quantize_int8()

    if hacer_fp16:
        export_fp16()

    compare_sizes()

    if hacer_int8:
        test_quantized()

    print("\n" + "=" * 60)
    print("💡 RECOMENDACIÓN PARA FLUTTER")
    print("=" * 60)
    print("""
  📱 Usa el modelo INT8: recommender_int8.onnx (~130 MB)
     - Cárgalo con onnxruntime en Flutter
     - Descarga: GET /ai/modelo/descargar
  
  ⚡ Alternativa FP16: recommender_fp16.onnx (~259 MB)
     - Sin pérdida de precisión
     - Compatible con GPU móvil

  Para descargar desde Flutter:
    var response = await http.get('$baseUrl/ai/modelo/descargar');
    await File('recommender_int8.onnx').writeAsBytes(response.bodyBytes);
    """)
