import subprocess
import pathlib

def test_quality_thresholds():
    # запускаем основной скрипт как «чёрный ящик»

    proc = subprocess.run(
        ["python", "run_ragas_demo.py"],
        capture_output=True, text=True
    )
    # если скрипт завершился кодом 1 — значит пороги не пройдены и тест провален 

    assert proc.returncode == 0, f"Quality gates failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    
    # проверим, что артефакт с подробностями создан
    assert pathlib.Path("ragas_result_detailed.csv").exists()
