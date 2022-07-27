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


x, y = load_dados('results_dock.csv')

if __name__ == '__main__':
    descriptor = Descriptor.Descriptor(x, y)

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

    print("\nHibrid (Schneider Wrede matrix)\n")
    x, z = descriptor.hibrid('Schneider_Wrede_matrix')
    training = IA.Traning(x, z)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nHibrid (Grantham matrix)\n")
    x, z = descriptor.hibrid('Grantham_matrix')
    training = IA.Traning(x, z)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nBalabanJ\n")
    x, y = descriptor.balabanj()
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nHallKierAlpha\n")
    x, y = descriptor.hallkieralpha()
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nBertzCT\n")
    x, y = descriptor.BertzCT()
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nKappa1\n")
    x, y = descriptor.Kappa1()
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nKappa2\n")
    x, y = descriptor.Kappa2()
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()

    #################################

    print("\nKappa3\n")
    x, y = descriptor.Kappa3()
    training = IA.Traning(x, y)
    training.radomflorest()
    training.svr()
    training.linear_svr()
    training.nu_svr()
    training.linear_regression()
