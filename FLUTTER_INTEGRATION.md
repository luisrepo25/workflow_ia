# Integración del Modelo de Recomendación en Flutter

## 📱 Resumen

El modelo entrenado (`recommender_int8.onnx`) es un clasificador DistilBERT Multilingual fine-tuneado que predice cuál de los 14 trámites existentes es el más adecuado para la consulta del usuario.

## 🧠 Ejecutar Inferencia en Flutter

Usa el paquete [`onnxruntime`](https://pub.dev/packages/onnxruntime):

### 1. Agregar dependencias en `pubspec.yaml`

```yaml
dependencies:
  onnxruntime: ^1.15.0
  http: ^1.2.0
  path_provider: ^2.1.0
```

### 2. Clase de inferencia

```dart
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';

class TramiteRecommender {
  late OrtSession _session;
  late OrtEnvironment _env;
  late List<String> _vocab;
  Map<String, Map<String, dynamic>>? _classMapping;
  
  static const int maxSeqLength = 128;
  
  Future<void> load(String modelPath, {String? classMappingJson}) async {
    // Cargar modelo ONNX
    _env = OrtEnvironment();
    await _env.init();
    
    final sessionOptions = OrtSessionOptions();
    _session = await OrtSession.fromFile(_env, modelPath, sessionOptions);
    
    // Cargar class mapping (opcional, para obtener nombre del trámite)
    if (classMappingJson != null) {
      _classMapping = json.decode(classMappingJson);
    }
    
    print('✅ Modelo cargado: ${_session.inputNames} -> ${_session.outputNames}');
  }
  
  /// Tokenización simple para DistilBERT Multilingual
  Uint8List _tokenize(String text) {
    // Para producción, usa un tokenizer real.
    // Aquí se muestra una versión simplificada.
    // Lo ideal es usar flutter_tokenizers o pre-tokenizar en el servidor.
    
    // El servidor ya devuelve resultados, esta implementación
    // es para inference offline completa.
    // ...
    
    // Para simplificar, recomiendo hacer la inferencia via API
    // cuando hay internet, y usar una versión reducida offline.
    throw UnimplementedError('Implementar con flutter_tokenizers');
  }
  
  Future<List<Map<String, dynamic>>> predict(String query, {int topK = 3}) async {
    // NOTA: La tokenización requiere un tokenizer BERT completo.
    // Opción A: Usar el endpoint /ai/recomendar (con internet)
    // Opción B: Incluir tokenizer en la app (recomendado)
    //
    // Para Opción B, usa el paquete `tokenizers` para Flutter:
    // https://pub.dev/packages/tokenizers
    
    throw UnimplementedError(
      'Usa el endpoint /ai/recomendar para inferencia remota '
      'o implementa tokenizer offline con flutter_tokenizers'
    );
  }
}
```

## 🎯 Estrategia Recomendada

### Opción 1: Inferencia Remota (Endpoint API) ✅ RECOMENDADA

```dart
Future<String> recomendarTramite(String consulta) async {
  final response = await http.post(
    Uri.parse('$baseUrl/ai/recomendar'),
    headers: {'Content-Type': 'application/json'},
    body: json.encode({'consulta': consulta, 'top_k': 3}),
  );
  
  if (response.statusCode == 200) {
    final data = json.decode(response.body);
    final recomendaciones = data['recomendaciones'];
    return recomendaciones[0]['nombre']; // Mejor trámite
  }
  throw Exception('Error: ${response.statusCode}');
}
```

### Opción 2: Inferencia Local (Offline Completa) ⚡

Para tener el modelo 100% offline en Flutter, necesitas:

1. **Descargar `recommender.onnx`** desde `/ai/modelo/descargar`
2. **Incluir el tokenizer** (archivos `vocab.txt` y `tokenizer.json`) desde HuggingFace
3. Usar el paquete [`tokenizers`](https://pub.dev/packages/tokenizers) para Flutter
4. Ejecutar inferencia con `ort_session.run()`

```
┌─────────────────────────────────────────┐
│           Flutter App                    │
│  ┌───────────────────────────────────┐   │
│  │   onnxruntime session             │   │
│  │   + recommender.onnx (25 MB)      │   │
│  │   + tokenizer.json + vocab.txt    │   │
│  └───────────────────────────────────┘   │
│         ↕                                 │
│  Consulta: "registro seguro social"       │
│  Resultado: "Ndeah" (confianza: 95%)     │
└─────────────────────────────────────────┘
```

## 📊 Comparativa de Tamaños

| Formato | Tamaño | Precisión | Uso en Flutter |
|---|---|---|---|
| **INT8 (recomendado)** | **~130 MB** | 89.7% F1 | ✅ `onnxruntime` |
| FP16 | ~259 MB | 90.5% F1 | ✅ `onnxruntime` |
| FP32 original | ~517 MB | 90.5% F1 | ❌ Muy grande |

> **¿Por qué 130 MB y no 67 MB?** El modelo `distilbert-base-multilingual-cased`
> tiene **134M parámetros** (no 67M) porque su vocabulario cubre 104 idiomas con
> 119K tokens. Incluso en INT8 (1 byte/parámetro) son ~134 MB.

## 📥 Descarga del Modelo

```dart
// 1. Obtener información del modelo
final info = await http.get(Uri.parse('$baseUrl/ai/modelo/info'));
print('Tamaño: ${info["tamano_mb"]} MB');
print('Precisión: ${info["precision"]}');

// 2. Descargar el modelo ONNX (INT8 ~130 MB)
final response = await http.get(Uri.parse('$baseUrl/ai/modelo/descargar'));
final bytes = response.bodyBytes;

// 3. Guardar en almacenamiento local
final dir = await getApplicationDocumentsDirectory();
final modelFile = File('${dir.path}/recommender_int8.onnx');
await modelFile.writeAsBytes(bytes);
```

## 🔄 Flujo Offline Completo (para producción)

```dart
// 1. Tokenizar el texto usando tokenizers package
final tokenizer = await BertTokenizer.fromPretrained('distilbert-base-multilingual-cased');
final encoding = tokenizer.encode(consulta);
final inputIds = Uint64List.fromList(encoding.ids.take(128).toList());
final attentionMask = Uint64List.fromList(encoding.attentionMask.take(128).toList());

// 2. Ejecutar inferencia
final inputs = {
  'input_ids': OrtTensor(inputIds, [1, 128]),
  'attention_mask': OrtTensor(attentionMask, [1, 128]),
};
final outputs = _session.run(inputs);
final logits = outputs['logits'] as List<double>;

// 3. Obtener clase con mayor probabilidad
final probs = softmax(logits);
final bestClass = probs.indexOf(probs.reduce(max));
final tramite = classMapping[bestClass.toString()];
```

## 📦 Archivos a incluir en la App Flutter

| Archivo | Tamaño | Fuente |
|---|---|---|
| `recommender.onnx` | ~170 MB | `/ai/modelo/descargar` |
| `class_mapping.json` | ~2 KB | `local_ia/models/class_mapping.json` |
| `tokenizer.json` | ~2 MB | HuggingFace Hub |
| `vocab.txt` | ~1 MB | HuggingFace Hub |
