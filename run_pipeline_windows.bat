
@echo off
REM === Pipeline para Detecção de Violência Física ===
REM === Ajuste os nomes dos vídeos se necessário ===

set MODEL=lstm
set FPS=10
set MAX_PEOPLE=5
set TRAIN_VIOLENCE=Fight1
set TRAIN_NONVIOLENCE=Normal1
set TEST_VIOLENCE=Fight2
set TEST_NONVIOLENCE=Normal2

echo Iniciando pipeline completo...
python src/run_pipeline.py ^
  --model %MODEL% ^
  --fps %FPS% ^
  --max_people %MAX_PEOPLE% ^
  --train_violence %TRAIN_VIOLENCE% ^
  --train_nonviolence %TRAIN_NONVIOLENCE% ^
  --test_violence %TEST_VIOLENCE% ^
  --test_nonviolence %TEST_NONVIOLENCE%

pause
