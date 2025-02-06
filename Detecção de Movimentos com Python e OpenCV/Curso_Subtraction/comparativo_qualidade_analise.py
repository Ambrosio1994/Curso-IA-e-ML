import pandas as pd
import matplotlib.pyplot as plt

avaliacao = pd.read_csv('report.csv')
#print(avaliacao)

resultado_avaliacao = avaliacao.groupby(['Frame']).sum()
print(resultado_avaliacao)

resultado_avaliacao.plot.bar()
plt.show()