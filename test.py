import matplotlib.pyplot as plt


def plot():
    data=[[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5],[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5],[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5],[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5]]
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    rows=['AB','BC','SK','MB']
    cols=["Gas", "Wk/Wk", "Yr/Yr","Oil", "Wk/Wk", "Yr/Yr","Bit", "Wk/Wk", "Yr/Yr","Total", "Wk/Wk", "Yr/Yr","Hor", "Wk/Wk", "Yr/Yr","Vert/Dir", "Wk/Wk", "Yr/Yr"]
    table = ax.table(cellText=data, colLabels=cols,rowLabels=rows, loc='upper center',cellLoc='center')
    table.auto_set_font_size(True)
    table.scale(1.5,1.5)
    #plt.savefig(filenameTemplate2, format='png',bbox_inches='tight')

def main():
    plot()