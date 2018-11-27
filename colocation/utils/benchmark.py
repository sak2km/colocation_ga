"""Benchmark Helper functions
"""

import logging
import time
import pathlib


LOG = logging.getLogger("M.benchmark")


class BenchmarkManager(object):
    """Manage benchmark states
    """

    def start(self):
        """Start benchmark
        """
        pass

    def end(self):
        """End benchmark
        """
        pass

    @staticmethod
    def new(option, output_path="./"):
        """Get a new benchmark manager composed
        """
        bm_list = (
            []
            if option == "None"
            else [_BenchmarkTimeIt()]
            if option == "TimeIt"
            else [_BenchmarkCProfile(output_path), _BenchmarkTimeIt()]
        )
        return _BenchmarkNested(bm_list)


class _BenchmarkTimeIt(BenchmarkManager):
    def __init__(self):
        self._start_time = 0
        self._logger = logging.getLogger("M.benchmark.timeit")

    def start(self):
        if not self._start_time:
            self._start_time = time.time()
            self._logger.info("Start Time %d", self._start_time)

    def end(self):
        if self._start_time:
            tok = time.time()
            time_passed = tok - self._start_time
            self._logger.info("End Time %d, Time Passed = %d", tok, time_passed)


class _BenchmarkCProfile(BenchmarkManager):
    def __init__(self, output_file):
        import cProfile

        self._profiler = cProfile.Profile = cProfile.Profile()
        self._output_file = pathlib.Path(output_file)

    def start(self):
        LOG.info("profiler starts")
        self._profiler.enable()

    def end(self):
        self._profiler.disable()
        LOG.info("profiler ended")

        dump_path = str(self._output_file.joinpath("profile.dump"))
        dot_path = str(self._output_file.joinpath("profile.dot"))
        png_path = str(self._output_file.joinpath("profile.png"))

        self._profiler.dump_stats(dump_path)

        import subprocess as sp

        LOG.debug("Run gprof2dot")
        ret = sp.run(
            ["gprof2dot", "-f", "pstats", "-o", dot_path, dump_path],
            timeout=1,
            shell=True,
        )

        if ret.returncode != 0:
            LOG.error("gprof2dot failed with code %d", ret.returncode)
            return

        LOG.debug("Run dot")
        ret = sp.run(["dot", "-Tpng", "-o", png_path, dot_path], timeout=1, shell=True)

        if ret.returncode != 0:
            LOG.error("dot failed with code %d", ret.returncode)
            return

        LOG.info("Successfully created profile files in dir %s", str(self._output_file))


class _BenchmarkNested(BenchmarkManager):
    def __init__(self, bm_list):
        self._bm_list = bm_list

    def append(self, new_bm):
        """Append new benchmark manager
        """
        self._bm_list.append(new_bm)

    def start(self):
        for bm in self._bm_list:
            bm.start()

    def end(self):
        for bm in reversed(self._bm_list):
            bm.end()
