#!/usr/bin/env python3
"""Run a segmented task (scripted moves + learned segments) in a cycle.

    python ./bin/RolloutSegments.py RealFairino5Demo \\
        --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \\
        --task ./envs/configs/segments/CupPickPlace.yaml [--mode exhibit]

See common/base/SegmentRunnerBase.py and doc/segment_data_collection.md.
"""

import argparse
import importlib
import importlib.util
import os
import sys

import yaml


class RolloutSegmentsMain:
    operation_parent_module_str = "robo_manip_baselines.envs.operation"

    def __init__(self):
        self.setup_args()

    def setup_args(self):
        env_utils_spec = importlib.util.spec_from_file_location(
            "EnvUtils",
            os.path.join(os.path.dirname(__file__), "..", "common/utils/EnvUtils.py"),
        )
        env_utils_module = importlib.util.module_from_spec(env_utils_spec)
        env_utils_spec.loader.exec_module(env_utils_module)

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="This is a meta argument parser for the segment runner switching between different environments. The actual arguments are handled by another internal argument parser.",
            fromfile_prefix_chars="@",
            add_help=False,
        )
        parser.add_argument(
            "env",
            type=str,
            help="environment",
            nargs="?",
            default=None,
            choices=env_utils_module.get_env_names(
                operation_parent_module_str=self.operation_parent_module_str
            ),
        )
        parser.add_argument("--config", type=str, help="configuration file")
        parser.add_argument(
            "-h",
            "--help",
            action="store_true",
            help="Show this help message and continue",
        )

        self.args, remaining_argv = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv
        if self.args.env is None:
            parser.print_help()
            sys.exit(1)
        elif self.args.help:
            parser.print_help()
            print("\n================================\n")
            sys.argv += ["--help"]

    def run(self):
        from robo_manip_baselines.common.base.SegmentRunnerBase import (
            SegmentRunnerBase,
        )

        operation_module = importlib.import_module(
            f"{self.operation_parent_module_str}.Operation{self.args.env}"
        )
        OperationEnvClass = getattr(operation_module, f"Operation{self.args.env}")

        # The order of parent classes must not be changed in order to maintain the method resolution order (MRO)
        class SegmentRunner(OperationEnvClass, SegmentRunnerBase):
            pass

        if self.args.config is None:
            config = {}
        else:
            with open(self.args.config, "r") as f:
                config = yaml.safe_load(f)

        runner = SegmentRunner(**config)
        runner.run()


if __name__ == "__main__":
    main = RolloutSegmentsMain()
    main.run()
