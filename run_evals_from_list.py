evals_to_run = [
    # "MathVista_MINI",
    # "MathVerse_MINI",
    # "MathVision_MINI",
    # "AI2D_TEST",
    # "BLINK",
    # # "ChartMuseum_test",
    # "ChartQA_TEST",
    # # "CharXiv_descriptive_val",
    # "DocVQA_VAL",
    # "HallusionBench",
    # # "LogicVista",
    # "MMMU_DEV_VAL",
    # "MMStar",
    # # "MUIRBench",
    # "OCRBench",
    # # "OlympiadBench",
    # "ScreenSpot_v2_Desktop",
    # "ScreenSpot_v2_Mobile",
    # "ScreenSpot_v2_Web",
    # # "ScreenSpot_Pro",
    # "WeMath",
    # "WildVision",
    "ZEROBench_sub",
    # # "VStarBench"
]

import subprocess, signal, sys, threading, os
from pathlib import Path
from datetime import datetime
from queue import Queue


logs_dir = Path("/home/jyotianeja/VLMEvalKit/logs")
logs_dir.mkdir(exist_ok=True)

def make_command(deployed_model_name, port, eval_name, api_nproc):
    return [
        "python",
        "run.py",
        "--model",
        deployed_model_name + f"-{port}",
        "--api-nproc",
        str(api_nproc),
        "--judge",
        "gpt-4o",
        "--data",
        eval_name
    ]

def parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run evals from a predefined list.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base name of the deployed model to evaluate.",
    )
    parser.add_argument(
        "--ports",
        type=str,
        nargs="+",
        required=True,
        help="List of ports where the model is deployed.",
    )
    parser.add_argument(
        "--par",
        type=int,
        default=1,
        help="Number of API processes to use.",
    )
    args = parser.parse_args()
    return args

STOP = False  # flag to tell workers to stop early on Ctrl-C


def run_eval_job(port: int, deployed_model_name: str, eval_name: str, api_nproc: int) -> int:
    """
    Run one eval for a given port. Returns the subprocess return code.
    """

    cmd = make_command(deployed_model_name, port, eval_name, api_nproc)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{eval_name}_model{deployed_model_name}_port{port}_{timestamp}.log"
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


def worker(port: int, deployed_model_name: str, job_queue: Queue, api_nproc):
    """
    Worker loop: always grab the next job and run it on this port.
    """
    while not STOP and not job_queue.empty():
        try:
            eval_name = job_queue.get_nowait()
        except:
            break

        try:
            rc = run_eval_job(port, deployed_model_name, eval_name, api_nproc)
            print(f"[port {port}] job {eval_name} finished with rc={rc}")
        except Exception as e:
            print(f"[port {port}] job {eval_name} failed: {e}", file=sys.stderr)
        finally:
            job_queue.task_done()


def main(model_name, ports, api_nproc):
    global STOP

    job_queue = Queue()
    for job in evals_to_run:
        job_queue.put(job)

    threads = []
    for port in ports:
        t = threading.Thread(target=worker, args=(port, model_name, job_queue, api_nproc), daemon=True)
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
    try:
        args = parse_cli_args()
        deployed_model_name = args.model
        PORTS = args.ports
        api_nproc = args.par
    except:
        raise("Failed to parse CLI arguments.")
    main(deployed_model_name, PORTS, api_nproc)