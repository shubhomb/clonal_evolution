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

def run_sim(cs, alphas, num_iters, doc_actions)
    assert len(cs) == len(alphas)

    model_data = {  'cA':       0.03,
                    'cB':       0.01,
                    'alpha':    0.05,
                    'beta':     0.03,
                    'dB':       0,
                    'dA':       0
                    }
if __name__ == "__main__":
    """
        Start Simulation
    """
    model_data = {  'cA':       0.03,
                    'cB':       0.01,
                    'alpha':    0.05,
                    'beta':     0.03,
                    'dB':       0,
                    'dA':       0
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

    for t in range(MAX_TIME):
        
        # Treatment
        if( t < MAX_TIME//2):
            model.modify_drug('dA', 0.9)
            model.modify_drug('dB', 0)
        else:
            model.modify_drug('dA', 0)
            model.modify_drug('dB', 0.5)


        # Calculuate Fitness
        model.colonyA.fitness = model.get_fitness('A')
        model.colonyB.fitness = model.get_fitness('B')
        model.colonyS.fitness = model.get_fitness('S')

        # Adjust Proportion
        model.adjust_proportion()
        

        # Log Data
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

    # Plot Resulting Curves
    
    x_axis = [k for k in range(MAX_TIME)]
    import matplotlib.pyplot as plt
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("proprtion of tumor population")
    plt.plot(x_axis, logpropA, label="A")
    plt.plot(x_axis, logpropB, label="B")
    plt.plot(x_axis, logpropS, label="S")
    plt.legend()
    title = str(model_data)
    plt.title(title)
    plt.savefig("viz/evolution" + ".png")
    plt.close()
    









    


