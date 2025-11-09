import os


class Calculator():
    def __init__(
        self,
        root_dir: str,
        task: str,
    ) -> None:
        self.root_dir = root_dir
        self.task = task
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def calc(self):
        NotImplementedError
