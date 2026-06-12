@echo off
REM ==========================================
REM  Pipeline completo: Entrenar recomendador
REM ==========================================
echo.
echo ========================================
echo   🧠 RECOMENDADOR DE TRAMITES - DL
echo   DistilBERT Multilingual + ONNX
echo ========================================
echo.

REM Paso 1: Generar dataset sintetico
echo [1/4] Generando dataset sintetico...
python local_ia/dataset_generator.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en dataset_generator
    exit /b 1
)
echo.

REM Paso 2: Entrenar modelo (fine-tuning)
echo [2/4] Entrenando DistilBERT (fine-tuning profundo)...
python local_ia/train_recommender.py --epochs 10 --batch-size 8
if %ERRORLEVEL% NEQ 0 (
    echo ERROR en train_recommender
    exit /b 1
)
echo.

REM Paso 3: Mostrar resultados
echo [3/4] Resultados del entrenamiento:
if exist local_ia\models\training_report.json (
    type local_ia\models\training_report.json | python -m json.tool
) else (
    echo   Reporte no encontrado
)
echo.

REM Paso 4: Probar inferencia
echo [4/4] Probando inferencia...
python -c "
from local_ia.recommender import WorkflowRecommender
r = WorkflowRecommender()
r.load()
tests = [
    'Necesito registrarme en el seguro social',
    'Quiero hacer un tramite de prueba',
    'Como registro una empresa',
]
for q in tests:
    res = r.predict(q)
    print(f\"\nConsulta: '{q}'\")
    for item in res:
        print(f\"  -> {item['nombre']} ({item['codigo']}) confianza: {item['confianza_pct']}%\")
"

echo.
echo ========================================
echo   ✅ ENTRENAMIENTO COMPLETADO
echo   Modelo ONNX: local_ia/models/recommender.onnx
echo   Testing: uvicorn core.asgi:app --reload
echo ========================================
pause
