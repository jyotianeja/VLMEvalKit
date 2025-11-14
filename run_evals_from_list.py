evals_to_run = [
    "AI2D_TEST",
    "atomic_dataset",
    "BLINK",
    "CCBench",
    "CCOCR",
    "ChartMimic_v1_direct",
    "ChartMuseum_test",
    "ChartQA_TEST",
    "ChartQAPro",
    "CharXiv_descriptive_val",
    "CMMU_MCQ",
    "COCO_VAL",
    "Detailed_Difference",
    "DocVQA_VAL",
    "DUDE",
    "EMMA",
    "GOBench",
    "GSM8K-V",
    "HallusionBench",
    "LEGO",
    "LiveMMBench_Infographic",
    "LiveMMBench_Perception",
    "LiveMMBench_Reasoning",
    "LogicVista",
    "MathVerse_MINI",
    "MathVision_MINI",
    "MathVista_MINI"
]

import subprocess, signal, sys, threading, os
from pathlib import Path
from datetime import datetime
from queue import Queue


logs_dir = Path("/home/mharrison/repos/VLMEvalKit/logs")
logs_dir.mkdir(exist_ok=True)

def make_command(deployed_model_name, port, eval_name):
    return [
        "python",
        "run.py",
        "--model",
        deployed_model_name + f"-{port}",
        "--api-nproc",
        "1",
        "--judge",
        "gpt-4o-impact",
        "--data",
        eval_name
    ]

deployed_model_name = "bunny-phi3"
PORTS = ["8080", "8081"]

STOP = False  # flag to tell workers to stop early on Ctrl-C


def run_eval_job(port: int, eval_name: str) -> int:
    """
    Run one eval for a given port. Returns the subprocess return code.
    """

    cmd = make_command(deployed_model_name, port, eval_name)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{eval_name}_port{port}_{timestamp}.log"
    with log_path.open("a", buffering=1) as log_file:
        log_file.write(f"=== START {eval_name} on port {port} ===\n")
        log_file.write(f"CMD: {' '.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = proc.wait()

        log_file.write(f"\n=== END {eval_name} (rc={return_code}) ===\n")

    return return_code


def worker(port: int, job_queue: Queue):
    """
    Worker loop: always grab the next job and run it on this port.
    """
    while not STOP and not job_queue.empty():
        try:
            eval_name = job_queue.get_nowait()
        except:
            break

        try:
            rc = run_eval_job(port, eval_name)
            print(f"[port {port}] job {eval_name} finished with rc={rc}")
        except Exception as e:
            print(f"[port {port}] job {eval_name} failed: {e}", file=sys.stderr)
        finally:
            job_queue.task_done()


def main():
    global STOP

    job_queue = Queue()
    for job in evals_to_run:
        job_queue.put(job)

    threads = []
    for port in PORTS:
        t = threading.Thread(target=worker, args=(port, job_queue), daemon=True)
        t.start()
        threads.append(t)

    try:
        # Wait for all jobs to be processed
        job_queue.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping workers...", file=sys.stderr)
        STOP = True

    # Ensure threads exit
    for t in threads:
        t.join(timeout=1.0)

    print("All jobs done (or interrupted).")

if __name__ == "__main__":
    main()