import torch
import matplotlib.pyplot as plt
import pickle
import scipy.io
import matplotlib.transforms as mtransforms


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Helvetica",
    "font.size": 12,
    "axes.grid" : True
})

files = ["data_paper/data/num_laser2_dist0_lengthscale010.obj",
         "data_paper/data/num_laser2_dist1_lengthscale010.obj",
         "data_paper/data/num_laser2_dist3_lengthscale010.obj",
         "data_paper/data/num_laser2_dist5_lengthscale010.obj",
         "data_paper/data/num_laser2_dist7_lengthscale010.obj",
         ]


## uncomment to plot newly generated data
# files = ["data/num_laser2_dist0_lengthscale010.obj",
#          "data/num_laser2_dist1_lengthscale010.obj",
#          "data/num_laser2_dist3_lengthscale010.obj",
#          "data/num_laser2_dist5_lengthscale010.obj",
#          "data/num_laser2_dist7_lengthscale010.obj",
#          ]

def plot1(ax,str,c):
    with open(str,'rb') as file:
        data = pickle.load(file)
    muy = -torch.tensor([y[-1] for y in data['bests']]).mean(dim=0)
    id1 = str.find('dist') + len('dist')
    id2 = str.find('_lengthscale')
    ax.plot(muy,f'--C{c}',label=rf"$\pm {int(str[id1:id2])*10}\%$")

fig,ax = plt.subplots(1,2)
fig.set_size_inches(9,2)

labels = ['(a)','(b)']

for label,ax_ in zip(labels,ax):
        trans = mtransforms.ScaledTranslation(0/72, 3/72, fig.dpi_scale_trans)
        ax_.text(0.0, 1.0, label, transform=ax_.transAxes + trans,
                fontsize=14, va='bottom', ha='center')

for i in range(len(files)):
    plot1(ax[0],files[i],i)
ax[0].legend(loc = 'upper right', ncol = 2)
ax[0].set_xlabel(r"Evaluations of main task $n$")
ax[0].set_ylabel(r"$J_{opt}(n)$")
ax[0].set_xlim(0,50)
ax[0].set_yticks([5,10,15,20])
ax[0].set_ylim(5,25)
###### Second Part

files = ["data_paper/data/num_laser5_pure_bo.obj",
         "data_paper/data/matlabSim_num_laser5.mat",
         "data_paper/data/num_laser5_dist1_lengthscale010_2.obj"]

## uncomment to plot newly generated data
# files = ["data/num_laser5_pure_bo.obj",
#          "data_paper/data/matlabSim_num_laser5.mat",
#          "data_paper/data/num_laser5_dist1_lengthscale010_2.obj"]

def plot2(ax,str,i):
    if str.find('.obj') != -1:
        with open(str,'rb') as file:
            data = pickle.load(file)
            stdy,muy = torch.std_mean(-torch.tensor([y[-1] for y in data['bests']]),dim=0)
    else:
        data = scipy.io.loadmat(str)
        stdy = data['std_Y'].squeeze(); muy = data['Y'].squeeze()
    x = torch.arange(len(muy))
    fill = ax.fill_between(x,muy+stdy,muy-stdy,color=f"C{i}",alpha=.3)
    line, = ax.plot(x,muy,f"C{i}")
    return (fill,line)

h = []
for i in range(len(files)):
    h.append(plot2(ax[1],files[i],i))
   
ax[1].legend(h,[r"\texttt{SafeBO} + EI",r"\texttt{MoSaOpt}",r"\texttt{SaMSBO} (our)"])
ax[1].set_xlabel(r"Evaluations of main task $n$")
ax[1].set_xlim(0,200)
ax[1].set_yticks([14,16,18,20])
ax[1].set_ylim(14,22)
plt.show()
fig.savefig("figures/comparison.pdf", bbox_inches='tight', pad_inches = 0.01, format = 'pdf')