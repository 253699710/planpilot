#! /usr/bin/env python3
from __future__ import annotations

import logging
import os
import pty
import select
import shutil
import sys
import termios
import utils

from subprocess import Popen, PIPE


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s ::: %(message)s",
)
logger = logging.getLogger(__name__)

dirname = os.path.dirname(__file__)


def fix_path(path):
    return os.path.join(dirname, path)


def run_fd_translator(domain, instance):
    binary_path = fix_path("translate/translate.py")
    python_exec = sys.executable  # Uses the currently running Python interpreter
    command = [python_exec, binary_path]
    # Local patch: translator pruning is the default; PLANPILOT_TRANSLATE_ORIGINAL=1
    # restores the keep-flags to match the unpruned SAT export.
    if os.environ.get("PLANPILOT_TRANSLATE_ORIGINAL", "0") == "1":
        command.append("--keep-unimportant-variables")
    command += [domain, instance]
    time = utils.get_elapsed_time()
    process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
    stdout, stderr = process.communicate()
    # if stdout:
    #     print(stdout, file=sys.stdout, end='')

    if stderr:
        print(stderr, file=sys.stderr, end="")
    _, error = process.communicate()
    logging.info(f"translator time: {utils.get_elapsed_time() - time:.2f}s")

    logger.info(f"translator return code: {process.returncode}")
    if process.returncode != 0:
        print(error)
        exit(process.returncode)


def run_plasp(domain, instance, lp, encoding, dump_output, pddl_instance=True):
    binary_path = fix_path("bin/plasp")
    command = [binary_path, "translate", instance]  # SAS+ instance
    if pddl_instance:
        command = [binary_path, "translate", domain, instance]

    with open(lp, "w") as lp_file:
        # First, we add the corresponding sequential encoding to it
        encoding = fix_path("encodings/exact-sequential-horizon.lp")
        if args.encoding == "bounded":
            encoding = fix_path("encodings/bounded-sequential-horizon.lp")
        with open(encoding) as seq_encoding:
            lp_file.write(seq_encoding.read())

        time_steps_encoding = fix_path("encodings/action-per-time-step.lp")
        if args.abstract_time_steps:
            time_steps_encoding = fix_path("encodings/abstract-time-steps.lp")

        with open(time_steps_encoding) as time_encoding:
            lp_file.write(time_encoding.read())

        if args.partial_plan:
            logger.info(f"Using partial plan '{args.partial_plan}'...")
            constraints: str | None = translate_partial_plan_into_constraints(
                args.partial_plan
            )
            lp_file.write("\n%%%%%%% partial plan encoding\n" + constraints + "\n")

        if args.add_constraints_file:
            logger.info(
                f"Using additional constraints of file '{args.add_constraints_file}'..."
            )
            try:
                with open(args.add_constraints_file, "r") as f:
                    constraints = f.readlines()
                    constraints = "".join(constraints)
                    lp_file.write(
                        "\n%%%%%%% partial plan encoding\n" + constraints + "\n"
                    )
            except IOError:
                print(
                    f"Error: File {args.add_constraints_file} does not appear to exist."
                )
                exit(1)

        if args.add_constraints:
            constraints = args.add_constraints.encode().decode("unicode_escape")
            lp_file.write("\n%%%%%%% partial plan encoding\n" + constraints + "\n")

        # Now, we run plasp to produce the instance-specific info
        process = Popen(command, stdout=lp_file, stdin=PIPE, stderr=PIPE, text=True)

    time = utils.get_elapsed_time()
    _, error = process.communicate()
    logging.info(f"plasp time: {utils.get_elapsed_time() - time:.2f}s")

    logger.info(f"plasp return code: {process.returncode}")
    if process.returncode != 0:
        print(error)
        exit(process.returncode)


def run_fasb(lp, horizon, script=None):
    binary_path = fix_path(
        "bin/fasb-x86_64-unknown-linux-gnu/fasb"
    )  # TODO don't keep multiple versions

    if script:
        binary_path = fix_path("bin/fasb_interpreter")
        logger.info(f"Executing fasb script {script}...")

    command = [binary_path, lp, "-c", f"horizon={horizon}", "0"]

    if script:
        full_script_path = fix_path(script)
        command.append(full_script_path)

    if script:
        # bin/fasb_interpreter is built locally (needs the interpreter Cargo
        # feature) against a recent glibc/libstdc++; on systems with an older
        # system glibc (e.g. HPC login nodes), run it inside a container that
        # has a matching glibc instead. Opt-in: only if both apptainer and the
        # container image are present; otherwise falls back to the direct call
        # above (which is what already works on machines with a recent glibc).
        sif_path = fix_path("bin/ubuntu2404.sif")
        apptainer = shutil.which("apptainer") or shutil.which("singularity")
        if apptainer and os.path.exists(sif_path):
            # realpath, not abspath: on systems where the home directory is a
            # symlink into another filesystem (e.g. /home/... -> /pfs/data6/...
            # on bwUniCluster), binding the symlinked path leaves the real one
            # unmounted, and apptainer aborts with
            #   FATAL: stat .../bin/fasb_interpreter: no such file or directory
            # The binary's own directory has to be bound for the same reason.
            binds = {os.path.dirname(os.path.realpath(lp)),
                     os.path.dirname(os.path.realpath(full_script_path)),
                     os.path.dirname(os.path.realpath(binary_path)),
                     os.path.realpath(os.path.expanduser("~"))}
            bind_args = []
            for b in binds:
                bind_args += ["--bind", b]
            command = [apptainer, "exec", *bind_args, sif_path] + command

    if args.dry:
        logging.info("Dry startup...")
        logging.info(
            "To compute facets matching some regular expression 're', use the command '!? re'..."
        )
        command.append("--f")

    time = utils.get_elapsed_time()
    if script:
        logging.info("Starting fasb...")
        process = Popen(command, stdout=PIPE, stderr=PIPE, text=True)
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout, file=sys.stdout, end="")

        if stderr:
            print(stderr, file=sys.stderr, end="")
    else:
        # Give fasb a terminal, so it flushes each interactive response and
        # its ready prompt. TextIO.readline combined with select on a pipe can
        # strand already-buffered lines and hang after a large answer.
        master_fd, slave_fd = pty.openpty()
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        process = Popen(
            command,
            stdout=slave_fd,
            stdin=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)

        logging.info(
            "Starting fasb... To quit the interactive mode, use the command ':q'."
        )

        # Set up the file descriptors for select
        prompts = 0
        try:
            inputs = [sys.stdin.fileno(), master_fd]
            while master_fd in inputs:
                reads, _, _ = select.select(inputs, [], [])
                if master_fd in reads:
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    os.write(sys.stdout.fileno(), data)
                if sys.stdin.fileno() in reads:
                    data = os.read(sys.stdin.fileno(), 65536)
                    if not data:
                        inputs.remove(sys.stdin.fileno())
                        os.write(master_fd, b":q\n")
                    else:
                        os.write(master_fd, data)
                        prompts += data.count(b"\n")

        except KeyboardInterrupt:
            # Handle Ctrl+C to exit cleanly
            print("Exiting...")
            os.write(master_fd, b":q\n")
        finally:
            os.close(master_fd)
            process.wait()

    logging.info(f"fasb time: {utils.get_elapsed_time() - time:.2f}s")

    logger.info(f"fasb return code: {process.returncode}")


def remove_lp_files():
    current_directory = os.getcwd()

    for file_name in os.listdir(current_directory):
        file_path = os.path.join(current_directory, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".lp"):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")


def translate_partial_plan_into_constraints(file_path: str) -> str | None:
    # TODO: give the name of specific format
    """
    Expects actions in specific format.

    Example:
    `a b c (1) d e (1)`
    will be translated into
    `
    :- not occurs(action(("a","b","c")),1).
    :- not occurs(action(("d","e")),2).
    `
    """
    global PARTIAL_PLAN_DELIMITER
    PARTIAL_PLAN_DELIMITER = "(1)"

    try:
        with open(file_path, "r") as f:
            actions = filter(
                lambda x: x,
                map(lambda x: x.strip(), f.read().split(PARTIAL_PLAN_DELIMITER)),
            )
            return "\n".join(
                map(
                    lambda t: translate_action_to_constraint(*t[::-1]),
                    enumerate(actions),
                )
            )
    except Exception as e:
        logger.error(f"could not read partial plan '{file_path}': {e}")


def translate_action_to_constraint(action: str, at: int) -> str:
    """
    Translates whitespace-separated action into integrity constraint.

    Example:
    `a b c` as first action will be translated into
    `:- not occurs(action(("a","b","c")),1).`
    """
    return (
        ":- not occurs(action(("
        + ",".join(map(lambda x: f'"{x}"', action.split(" ")))
        + f")),{at+1})."
    )


if __name__ == "__main__":
    args = utils.parse_arguments()

    domain_file = args.domain
    instance_file = args.instance
    if args.is_pddl_instance and not os.path.isfile(domain_file):
        sys.stderr.write("Error: Domain file does not exist.\n")
        sys.exit()
    if not os.path.isfile(instance_file):
        sys.stderr.write("Error: Instance file does not exist.\n")
        sys.exit()

    if args.is_pddl_instance:
        logger.info("Running translator...")
        run_fd_translator(args.domain, args.instance)
        args.instance = "output.sas"
        # run_plasp(
        #     args.domain,
        #     args.instance,
        #     args.lp_name,
        #     args.encoding,
        #     args.dump_output)

    logger.info("Running plasp...")
    run_plasp("", args.instance, args.lp_name, args.encoding, args.dump_output, False)

    logger.info("Running fasb...")
    run_fasb(args.lp_name, args.horizon, args.script)

    if args.cleanup:
        remove_lp_files()

    logging.info(f"Total time: {utils.get_elapsed_time():.2f}s")
    logger.info("Done!")
