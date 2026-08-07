#!/usr/bin/env python
"""Script to run MLflow UI."""

import subprocess

if __name__ == "__main__":
    # Start MLflow UI
    cmd = ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
    print(f"Running: {' '.join(cmd)}")
    print("MLflow UI will be available at http://localhost:5000")
    subprocess.run(cmd)
