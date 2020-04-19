from evolve import *




def extract_fit(model):
    data_dic, colonies = model.get_full_data()
    fitA = colonies[0].fitness
    fitB = colonies[1].fitness
    fitS = colonies[2].fitness
    return fitA, fitB, fitS

def extract_prop(model):
    data_dic, colonies = model.get_full_data()    
    propA = colonies[0].prop
    propB = colonies[1].prop
    propS = colonies[2].prop
    return propA, propB, propS


if __name__ == "__main__":
    """
        Start Simulation
    """
    model_data = {  'cA':       0.01,
                    'cB':       0.01,
                    'alpha':    0.03,
                    'beta':     0.03,
                    'dB':       0,
                    'dA':       1
                    }
    model = Evolve(model_data)

    MAX_TIME = 100

    logpropA = []
    logpropB = []
    logpropS = []
    logfitA = []
    logfitB = []
    logfitS = []
    
    model.log()

    for _ in range(MAX_TIME):

        # Calculuate Fitness
        model.colonyA.fitness = model.get_fitness('A')
        model.colonyB.fitness = model.get_fitness('B')
        model.colonyS.fitness = model.get_fitness('S')

        # Adjust Proportion
        model.adjust_proportion()
        
        propA, propB, propS = extract_prop(model)
        fitA, fitB, fitS    = extract_fit(model)

        logpropA.append(propA)
        logpropB.append(propB)
        logpropS.append(propS)

        logfitA.append(fitA)
        logfitB.append(fitB)
        logfitS.append(fitS)
        model.log()
        model.inc_time()
    
    x_axis = [k for k in range(MAX_TIME)]
    import matplotlib.pyplot as plt
    plt.plot(x_axis, logpropA, x_axis, logpropB, x_axis, logpropS)
    title = 'Proportions over Time'
    plt.title(title)
    plt.savefig(title + ".png")
    plt.close()
    









    


