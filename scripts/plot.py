import seaborn as sns
import pandas as pd
import matplotlib
import os
import math
from statistics import mean
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

parser = ArgumentParser()
parser.add_argument("-m", "--method", help='chol or lu')
parser.add_argument("-s", "--size", help='size of matrix (12000, 24000, etc.)')
parser.add_argument('-l', '--legend', action='store_true', 
                    help='add legend to the graph')
parser.add_argument('-t', '--title', action='store_true', 
                    help='add title to the graph')
parser.add_argument('-st', '--subtitle', action='store_true', 
                    help='add title to the graph')
# parser.add_argument("-q", "--quiet",
#                     action="store_false", dest="verbose", default=True,
#                     help="don't print status messages to stdout")

args = parser.parse_args()

sizes = [12000, 24000, 36000, 48000, 60000]
if args.size == None:
    print("generating for all sizes")
else:
    size = int(args.size)
    if size in sizes:
        sizes = [size]

gpu_count = 8
metric = "flops"
tflops = True
addlegend = args.legend
addtitle = args.title
addsubtitle = args.subtitle

method_name = 'LU'
if args.method == 'chol':
    method_name = 'Cholesky'
elif args.method != 'lu':
    print("wrong method, provide '-m lu' or '-m chol'")
    exit(0)

methods = {0: "cusolver", 1: "cudaGraph", 2: "Mustard", 3: "cusolverMg", 4: "StarPU", 5: "Slate", 6: "cuMgFine"}

# Cholesky flop calculation
def getFLOPs(n, time):
    flop = (2.0*float(n*n*n))/3.0
    flops = flop/time
    if tflops:
        return flops/1000.0/1000.0/1000.0/1000.0
    else:
        return flops

def readStarPUDirty(file, params):
    global data
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data_point = [method,gpu_count,0.0,0.0]
    for line in file.readlines():
        if (line.startswith(str(size))):
            #print(line)
            runtime = float(line.split("	")[1])/1000.0
            data_point[2] = runtime
            data_point[3] = getFLOPs(size, runtime)
            data = pd.concat([pd.DataFrame([data_point], columns=data.columns), data], ignore_index=True)
    # print(data[method][gpu_count])
    # data[method][gpu_count] = mean(data[method][gpu_count])

def readStarPU(file, params):
    global data
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data_point = [method,gpu_count,0.0,0.0]
    for line in file.readlines():
        if (line.startswith("[starpu][_starpu_update_perfmodel_history]")):
            # print("flushing uncalibrated starpu data")
            # print(data[data["method"] == method] )
            data = data[data["method"] != method] 
        if (line.startswith(str(size))):
            #print(line)
            runtime = float(line.split("	")[1])/1000.0
            data_point[2] = runtime
            data_point[3] = getFLOPs(size, runtime)
            data = pd.concat([pd.DataFrame([data_point], columns=data.columns), data], ignore_index=True)
    # print(data[method][gpu_count])
    # data[method][gpu_count] = mean(data[method][gpu_count])

def readMG(file, params):
    global data
    method = methods[int(params[0][3:])]
    size = int(params[1])
    tiles = int(params[2])
    gpu_count = int(params[3][:-3])
    if (tiles > 20):
        method = methods[6]
    data_point = [method,gpu_count,0.0,0.0]
    for line in file.readlines():
        if (line.startswith("Run")):
            #print(line)
            runtime = float(line.split(" ")[-1])
            data_point[2] = runtime
            data_point[3] = getFLOPs(size, runtime)
            data = pd.concat([pd.DataFrame([data_point], columns=data.columns), data], ignore_index=True)


def readSlate(file, params):
    global data
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data_point = [method,gpu_count,0.0,0.0]
    for line in file.readlines():
        if (line.startswith("device")):
            runtime = float(line.split(" ")[-3])
            data_point[2] = runtime
            data_point[3] = getFLOPs(size, runtime)
            data = pd.concat([pd.DataFrame([data_point], columns=data.columns), data], ignore_index=True)

def readMustard(file, params):
    global data
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data_point = [method,gpu_count,0.0,0.0]
    gpu_idx = 0
    max_runtime = 0.0
    for line in file.readlines():
        if (line.startswith("device")):
            #print(line)
            runtime = float(line.split(" ")[-1])
            max_runtime = max(runtime, max_runtime) 
            gpu_idx += 1
            if (gpu_idx == gpu_count):
                data_point[2] = runtime
                data_point[3] = getFLOPs(size, runtime)
                data = pd.concat([pd.DataFrame([data_point], columns=data.columns), data], ignore_index=True)
                max_runtime = 0.0
                gpu_idx = 0

def readData(size):
    log_folder = "./results/" + str(args.method) + "/" + str(size)
    for f in os.listdir(log_folder):
        filename = os.fsdecode(f)
        print(filename)
        if filename.endswith(".log"): 
            file = open(os.path.join(log_folder, filename), "r")
            # print(os.path.join(directory, filename))
            spl_name = filename[:-4].split("_")
            method = int(spl_name[0][3:])
            if (method < 3):  
                if (method == 0):
                    spl_name.append("1")
                if (method <= 1):
                    spl_name.append("1GPU")
                readMustard(file, spl_name)
            if (method == 3):
                readMG(file, spl_name)
            if (method == 4):
                readStarPU(file, spl_name)
            if (method == 5):
                readSlate(file, spl_name)
            continue
        else:
            continue 

def plotForSize(size):
    global data
    print(size)
    data = pd.DataFrame(columns=['method','gpu_count','time','flops'])
    readData(size)
    skip_gpu = [3,5,7]

    custom = {"axes.edgecolor": ".2", "grid.linestyle": "dashed", "grid.color": ".2"}
    sns.set_style("darkgrid", rc = custom)

    # g = sns.catplot(
    #     data=df, kind="bar",
    #     x="species", y="body_mass_g", hue="sex",
    #     errorbar="sd", palette="dark", alpha=.6, height=6
    # )
    plot_count = gpu_count-len(skip_gpu)
    width_ratios=[4]*plot_count
    width_ratios[0] = 6
    figsize=(12, 2.2)
    if addtitle and addsubtitle:
        figsize=(12, 2.2*1.17)
    elif addsubtitle:
        figsize=(12, 2.2*0.97)
    # elif not addtitle and not addsubtitle:
    #     figsize=(12, 2.0*0.95)
    # elif not addtitle and not addsubtitle:
    #     figsize=(12, 2.0*0.95)
    fig, axes = plt.subplots(1, plot_count, figsize=figsize, gridspec_kw={'width_ratios': width_ratios}, sharey=True)

    # hatches = ['/', '+', '-', 'x', '\\', '*', 'o', 'O', '.']
    hatches = ['++', '||', '--', '//', '///', '\\\\', 'xx']
    legend_hatches = ['+++', '|||', '----', '///', '////', '\\\\\\', 'xxx']
    baseline_method=data[data["method"] == methods[0]]
    # print(baseline_method)
    # baseline=baseline_method[data["gpu_count"] == 1]
    baseline=baseline_method[metric].mean()
    # print(baseline)
    colors=["#16a55c", "#AFD12C", "#F2C020", "#7868ce", "#5771dc", "#37A1cc", "#31Ab92"]
    baselinecolor="#06452c"

    index = 0
    matplotlib.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth
    methods_to_plot = methods.copy()

    ymax = 0
    if (method_name == "LU"):
        del legend_hatches[3]
        del hatches[3]
        del methods_to_plot[3]
        del colors[3]
    for gpu in range(gpu_count):
        if gpu+1 in skip_gpu:
            # print("skip")
            continue
        order = list(methods_to_plot.values())
        order[-1], order[-3] = order[-3], order[-1]
        palette = colors
        hatch_palette = hatches
        title=str(gpu+1)+"GPU"
        if (index != 0):
            order = order[2:]
            palette = colors[2:]
            hatch_palette = hatches[2:]
            title+="s"

        sns.barplot(ax=axes[index], data=data[data["gpu_count"] == gpu+1][1:], x="method", 
                    y=metric, linewidth=1, edgecolor=".2", hue="method", palette=palette, 
                    order=order, hue_order=order, capsize=.2, dodge=False, err_kws={'linewidth': 1.5})
                    # errwidth=1.5
        
        patches = axes[index].patches
        lines_per_err = 3

        for i, line in enumerate(axes[index].get_lines()):
            newcolor = patches[i // lines_per_err].get_facecolor()
            line.set_color(".2")

        for i, bar in enumerate(patches):
            # if (index == 0 and i == 1):
            #     continue
            bar.set_hatch(hatch_palette[i%len(hatch_palette)])

        axes[index].axhline(baseline.mean(), color=baselinecolor)
        if (addsubtitle):
            axes[index].title.set_text(title)
        
        if (index == 0):
            ylabel = metric.upper()
            if tflops:
                ylabel = "T" + ylabel
            axes[index].set(ylabel=ylabel)
            
            if addlegend:
                legend_elements = []
                method_names = list(methods_to_plot.values())
                method_names[-1], method_names[-3] = method_names[-3], method_names[-1]
                for i in range(len(method_names)):
                    legend_elements.append(Patch(facecolor=palette[i%len(palette)], 
                                                edgecolor='0.2', label=method_names[i], 
                                                hatch=legend_hatches[i%len(legend_hatches)]))
                legend = axes[index].legend(handles=legend_elements, loc='upper right', 
                                ncol=2, columnspacing=2, prop={'size': 9.4}, handleheight=1.6)
                labels = legend.get_texts()
                labels[2].set_fontweight('bold')
        else:
            axes[index].set(ylabel=None)
        axes[index].set(xticklabels=[])  # remove the tick labels
        axes[index].set(xlabel=None)
        # axes[index].set(xlabel=title)
        _, ymax_current = axes[index].get_ylim()
        # print(ymax_current)
        ymax = max((math.ceil(ymax_current/20.1))*20, ymax)
        # print(ymax)
        if (gpu+1 == gpu_count):
            axes[index].set_ylabel(str(size)+'x'+str(size), rotation=270, labelpad=15, ha='right')
            axes[index].yaxis.set_label_position("right")
            axes[0].set(ylim=(0, ymax))
        axes[index].yaxis.set_major_locator(MultipleLocator(20))
        axes[index].set_xlim(-0.6, len(order)-0.4)

        # Adding "X" symbol for zero values
        for i in range(len(order)):
            method = order[i]
            if len(data[metric][data["method"] == method][data["gpu_count"] == gpu+1][1:]) == 0:
                # print(method)
                axes[index].annotate('X', (i, 0), ha='center', va='bottom', color='0.2', fontsize=12)

        index+=1
    plt.tight_layout()
    
    if (addtitle):
        fig.suptitle(method_name, fontsize=14, x=0.47)
        fig.subplots_adjust(top=0.85)
        if (addsubtitle):
            fig.subplots_adjust(top=0.8)
    figname = method_name + "_" + str(size) + "_"  + str(gpu_count) + "GPU_" + metric
    if (addlegend):
        figname += "_legend"
    plt.savefig('figures/'+figname + ".pdf")
    plt.show()

for size in sizes:
    plotForSize(size)
