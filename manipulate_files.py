import pandas as pd
import numpy as np
# open files and create lists (x, y)
import Descriptor
import IA


def load_dados(file):
    dataset = pd.read_csv(file)

    df = pd.DataFrame({'seqs': dataset['sequence'],
                       'docking': dataset['Energies']})

    df = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
    seqs = df['seqs']
    docking = df['docking']

    return seqs, docking

def preidct(file):
    list_seqs = []
    with open(file, 'r') as arq:
        lines = arq.readlines()
        for i in lines:
            list_seqs.append(i.rstrip('\n'))
    return list_seqs

x, y = load_dados('results_dock.csv')

if __name__ == '__main__':
    descriptor = Descriptor.Descriptors(x, y)

    print("SOCNUMBER (Schneider Wrede matrix)\n")
    x, z = descriptor.socnumber('Schneider_Wrede_matrix')
    training = IA.Traning(x, z)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nSOCNUMBER (Grantham matrix)\n")
    x, z = descriptor.socnumber('Grantham_matrix')
    training = IA.Traning(x, z)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nBalabanJ\n")
    x, y = descriptor.CalcDescRDKit('BalabanJ')
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nHallKierAlpha\n")
    x, y = descriptor.CalcDescRDKit('HallKierAlpha')
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nBertzCT\n")
    x, y = descriptor.CalcDescRDKit('BertzCT')
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nKappa1\n")
    x, y = descriptor.CalcDescRDKit('Kappa1')
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nKappa2\n")
    x, y = descriptor.CalcDescRDKit('Kappa2')
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nKappa3\n")
    x, y = descriptor.CalcDescRDKit('Kappa3')
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()
