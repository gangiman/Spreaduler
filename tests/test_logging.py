import spreaduler


class MockExperimentParams:
    def __init__(self):
        self.last_call = None
        self.last_args = None

    def _h(self, call, args):
        self.last_call = call
        self.last_args = args

    def log_server(self):
        self._h('server', "")

    def log_time_to_column(self, column_name):
        self._h('time', [column_name])

    def log_status(self, status):
        self._h('status', [status])

    def log_comment(self, text):
        self._h('comment', [text])

    def log_progress(self, i, max_i):
        self._h('progress', [i, max_i])

    def log_experiment_id(self, value):
        self._h('experiment', [value])

    def log_metric(self, metric, value):
        self._h('metric', [metric, value])


def test_log_functions():
    to_test = [
        (spreaduler.log_metric, "metric", ["accuracy", 0.7]),
        (spreaduler.log_progress, "progress", [1, 10]),
        (spreaduler.log_status, "status", ["DONE"]),
        (spreaduler.log_experiment_id, "experiment", [42]),
        (spreaduler.log_comment, "comment", ["text"])
    ]
    # It should not throw exceptions when experiment is not set up
    spreaduler._current_experiment = None
    for fun, _, args in to_test:
        fun(*args)
    # Calls and args will be registered by MockExperimentParams
    mock = MockExperimentParams()
    spreaduler._current_experiment = mock
    for fun, expected_call, args in to_test:
        fun(*args)
        assert mock.last_call == expected_call
        assert mock.last_args == args
