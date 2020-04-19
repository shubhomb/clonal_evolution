class Colony:
    """
        Initializes a Subclone Population.
        :attr label:    Either A, B or S
        :attr fitness:  Current fitness
        :attr prop:     Current Proportion
    """
    def __init__(self, lbl, data, prop=0.333):
        self.label = lbl
        self.fitness = 1
        self.prop = prop

class Evolve:
    """
        Evolve Class contains relevant methods to simulate the evolution
        of the subclone colonies.  In this example, it is hardcoded to three
        colonies.
    """


    def get_full_data(self):
        colonies = [self.colonyA, self.colonyB, self.colonyS]
        return self.data, colonies

    def adjust_proportion(self):
        """
            Adjusts using p_t+1 = p_t * W(i)/W
            This assumes fitness are already updated / calculuated
        """
        avg_fit = self.calc_avg_fitness()
        self.colonyA.prop *= self.colonyA.fitness/avg_fit
        self.colonyB.prop *= self.colonyB.fitness/avg_fit
        self.colonyS.prop *= self.colonyS.fitness/avg_fit
        

    def get_fitness(self, type):
        """
            Returns the fitness with the given environment for subclone [type]
        """
        if type == 'A':
            fitness = 1 - self.data['cA'] - self.data['alpha'] * self.data['dB']
        elif type == 'B':
            fitness = 1 - self.data['cB'] - self.data['beta'] * self.data['dA']
        elif type == 'S':
            fitness = 1 - self.data['dA'] - self.data['dB']
        else:
            raise ValueError('Invalid Type')
        return fitness


    def calc_avg_fitness(self):
        """
            Given fitness environment, returns average fitness as per :
            
        """
        avgfit = 0
        avgfit += self.colonyA.prop * self.colonyA.fitness
        avgfit += self.colonyB.prop * self.colonyB.fitness
        avgfit += (1 - self.colonyA.prop - self.colonyB.prop) * self.colonyS.fitness
        return avgfit


    def log(self):
        header = f'At time step t = {self.time}:'
        astr = f'A:\t prop{round(self.colonyA.prop,5)} \t {round(self.colonyA.fitness, 5)}'
        bstr = f'B:\t prop{round(self.colonyB.prop,5)} \t {round(self.colonyB.fitness, 5)}'
        sstr = f'S:\t prop{round(self.colonyS.prop, 5)} \t {round(self.colonyS.fitness, 5)}'
        print(header + '\n\t' + astr + '\n\t' + bstr + '\n\t' + sstr + '\n')
        

    def print_attr(self):
        cA = self.data['cA']
        cB = self.data['cB']
        alpha = self.data['alpha']
        beta = self.data['beta']
        dB = self.data['dB']
        dA  = self.data['dA']

        header = f'At time step t={self.time}:'
        data = f'\t cA = { cA } \n \t cB = {cB} \n'
        data2 = f'\t alpha = {alpha} \n\t beta= {beta} \n'
        data3 = f'\t dB={dB} \n\t dA = {dA} \n'
        print(header + '\n' + data + data2 + data3)

    def inc_time(self):
        self.time += 1

    def __init__(self, dic=None, num_med=2):        
        """
            :attr time: Time stamp starting from t=0
            :attr data: dictionary containing envirnoment parameters
            :attr colonyA: colonyA information
            :attr colonyB: colonyB information
            :attr colonyS: colonyS information

        """
        self.time = 0
        self.data = {}

        if dic == None:
            assert ['cA', 'cB', 'alpha', 'beta', 'dB', 'dA'] in data.keys()
            assert len(dic) == 6
            self.data = {'cA': 0,
                         'cB': 0,
                         'alpha': 0,
                         'beta': 0,
                         'dB':   0,
                         'dA': 0
                         }
        else:
            self.data = dic
    
        self.colonyA = Colony('A', self.data)     # RA
        self.colonyB = Colony('B', self.data)     # RB
        self.colonyS = Colony('S', self.data)     # S
        

        