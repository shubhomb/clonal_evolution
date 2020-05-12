class Environment():
    """
        This is used as an input to contruct the graph.
    """

    def __init__(self, names, matrix, alpha, prop):
        self.names = names  # Names of Subclone Colony (List)
        self.relations = matrix  # Adj Matrix representing relations
        self.alpha = alpha
        self.prop = prop

    def log(self):
        print('Logging Environment:')
        items = [f' -->  {d}' for d in zip(self.names, self.alpha, self.prop)]
        for i in items:
            print(i)
        print(20 * '-*')

    def get_env_data(self):
        return [d for d in zip(self.names, self.alpha, self.prop)]

