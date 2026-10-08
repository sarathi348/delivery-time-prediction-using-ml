"""One-click launcher: installs libraries, trains the model, makes figures, opens the app.

Usage:  python run_all.py            (trains only the first time)
        python run_all.py --retrain  (force a fresh run)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable


def run(*args):
    subprocess.check_call([PY, *args])


def main():
    print("== Step 1/4: installing libraries ==")
    run("-m", "pip", "install", "-q", "-r", "requirements.txt")

    need_train = "--retrain" in sys.argv or not os.path.exists("delivery_model.joblib")
    if need_train:
        print("== Step 2/4: generating dataset ==")
        run("simulate.py")
        print("== Step 3/4: training and comparing models (takes a few minutes) ==")
        run("experiment.py")
        run("figs.py")
        print("Figures saved as fig*.png, metrics saved in results.json")
    else:
        print("== Steps 2-3 skipped: trained model already exists (use --retrain to redo) ==")

    print("== Step 4/4: opening the prediction app ==")
    run("app.py")


if __name__ == "__main__":
    main()
