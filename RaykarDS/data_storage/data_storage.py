class DataStorage:
    """
    Interface for data storages for experiments.
    """
    def bootstrap(self, size: int = None, marks_percentage:int = None, at_least: int = 1, seed: int = 0):
        """
        Choose uniformly <size> tasks and <cnt_marks> so that each task would have at least <at_least> marks. If for some
        task there are less marks then <at_least> all marks will be taken.

        :param size:
        :param cnt_marks:
        :param at_least:
        :param seed: Seed of random.
        :return: Numpy array of tasks' features, workers marks and real answer.
        """
        raise NotImplementedError("Interface method")
