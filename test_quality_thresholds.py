import subprocess
import pathlib

def test_quality_thresholds():
    # run the main script as a "black box"

    proc = subprocess.run(
        ["python", "run_ragas_demo.py"],
        capture_output=True, text=True
    )
    # if the script exits with code 1, thresholds are not met and the test fails

    assert proc.returncode == 0, f"Quality gates failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    
    # Check that the artifact with details is created
    assert pathlib.Path("ragas_result_detailed.csv").exists()


if __name__ == "__main__":
    test_quality_thresholds()

