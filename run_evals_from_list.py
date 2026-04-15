evals_to_run = [
    "AI2D_TEST",
     # "BLINK",
     # "ChartMuseum_test",
     # "ChartQA_TEST",
     # "CharXiv_descriptive_val",
     # "DocVQA_VAL",
    "HallusionBench",
    "MathVerse_MINI",
    "MathVision_MINI",
    "MathVista_MINI",        
     # "LogicVista",
    "MMMU_DEV_VAL",
    "MMStar",
    # "MUIRBench",
    "OCRBench",
    # "OlympiadBench",
    "ScreenSpot_v2_Desktop",
    "ScreenSpot_v2_Mobile",
    "ScreenSpot_v2_Web",
    # "ScreenSpot_Pro",
    # "WeMath",
    # "WildVision",
    # "ZEROBench_sub",
    # "VStarBench"
]

import subprocess, signal, sys, threading, os
from pathlib import Path
from datetime import datetime
from queue import Queue
from tqdm import tqdm


logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

def make_command(deployed_model_name, port, eval_name, api_nproc):
    return [
        "python",
        "-u",
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
    Stdout goes to the log file; stderr (where tqdm writes) goes to the terminal.
    """

    cmd = make_command(deployed_model_name, port, eval_name, api_nproc)

    print(f"Starting with command: {' '.join(cmd)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{eval_name}_model{deployed_model_name}_port{port}_{timestamp}.log"
    with log_path.open("a", buffering=1) as log_file:
        log_file.write(f"=== START {eval_name} on port {port} ===\n")
        log_file.write(f"CMD: {' '.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = proc.wait()

        log_file.write(f"\n=== END {eval_name} (rc={return_code}) ===\n")

    return return_code


def worker(port: int, deployed_model_name: str, job_queue: Queue, api_nproc, pbar: tqdm):
    """
    Worker loop: always grab the next job and run it on this port.
    """
    while not STOP and not job_queue.empty():
        try:
            eval_name = job_queue.get_nowait()
        except:
            break

        pbar.set_postfix_str(f'port {port}: {eval_name}')
        try:
            rc = run_eval_job(port, deployed_model_name, eval_name, api_nproc)
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            tqdm.write(f"[port {port}] {eval_name}: {status}")
        except Exception as e:
            tqdm.write(f"[port {port}] {eval_name} failed: {e}")
        finally:
            pbar.update(1)
            job_queue.task_done()


def main(model_name, ports, api_nproc):
    global STOP

    job_queue = Queue()
    for job in evals_to_run:
        job_queue.put(job)

    pbar = tqdm(total=len(evals_to_run), desc='Evals', unit='eval')

    threads = []
    for port in ports:
        t = threading.Thread(target=worker, args=(port, model_name, job_queue, api_nproc, pbar), daemon=True)
        t.start()
        threads.append(t)

    try:
        # Wait for all jobs to be processed
        job_queue.join()
    except KeyboardInterrupt:
        tqdm.write("KeyboardInterrupt: stopping workers...")
        STOP = True

    pbar.close()

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