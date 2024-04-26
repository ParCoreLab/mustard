import seaborn as sns
import pandas as pd
import matplotlib
import os
from statistics import mean 
from matplotlib import pyplot as plt

title = "LU"
metric = "flops"
log_folder = "./logs/24000_2"
gpu_count = 8
#log_folder = "./chol_logs/24000"
data = pd.DataFrame(columns=['method','gpu_count','time','flops'])
methods = {0: "getrf", 1: "cudaGraph", 2: "ours", 3: "MgGetrf", 4: "StarPU"}
#methods = {0: "potrf", 1: "cudaGraph", 2: "ours", 3: "MgPotrf"}

# LU flop calculation
def getFLOPs(n, time):
    flop = (2.0*float(n*n*n))/3.0
    return flop/time #/1000.0

def readStarPU(file, params):
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

def readMG(file, params):
    global data
    method = methods[int(params[0][3:])]
    size = int(params[1])
    gpu_count = int(params[3][:-3])
    data_point = [method,gpu_count,0.0,0.0]
    for line in file.readlines():
        if (line.startswith("Run")):
            #print(line)
            runtime = float(line.split(" ")[-1])
            data_point[2] = runtime
            data_point[3] = getFLOPs(size, runtime)
            data = pd.concat([pd.DataFrame([data_point], columns=data.columns), data], ignore_index=True)

def readOurs(file, params):
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

def readData():
    for f in os.listdir(log_folder):
        filename = os.fsdecode(f)
        print(filename)
        if filename.endswith(".log"): 
            file = open(os.path.join(log_folder, filename), "r")
            # print(os.path.join(directory, filename))
            spl_name = filename[:-4].split("_")
            method = int(spl_name[0][3:])
            # size = int(spl_name[1])
            # print(method)
            # print(size)
            # if len(spl_name) > 2:
            #     tiles = int(spl_name[2])
            #     print(tiles)
            # if len(spl_name) > 3:
            #     gpu_count = int(spl_name[3][:-3])
            #     print(gpu_count)
            # else:
            if (method < 3):  
                if (method == 0):
                    spl_name.append("1")
                if (method <= 1):
                    spl_name.append("1GPU")
                readOurs(file, spl_name)
            if (method == 3):
                readMG(file, spl_name)
            if (method == 4):
                readStarPU(file, spl_name)
            continue
        else:
            continue 

readData()

sns.set_theme(style="whitegrid")
# g = sns.catplot(
#     data=df, kind="bar",
#     x="species", y="body_mass_g", hue="sex",
#     errorbar="sd", palette="dark", alpha=.6, height=6
# )
fig, axes = plt.subplots(1, gpu_count, figsize=(15, 5), sharey=True)

baseline_method=data[data["method"] == methods[1]]
print(baseline_method)
# baseline=baseline_method[data["gpu_count"] == 1]
baseline=baseline_method[metric].mean()
print(baseline)

for gpu in range(gpu_count):
    order = list(methods.values())
    title=str(gpu+1)+"GPU"
    if (gpu != 0):
        order = order[2:]
        title+="s"
    sns.barplot(ax=axes[gpu], data=data[data["gpu_count"] == gpu+1], x="method", y=metric,
                order=order, linewidth=2, edgecolor=".5", facecolor=(0, 0, 0, 0))
    axes[gpu].axhline(baseline.mean(), color='r')
    axes[gpu].title.set_text(title)
    axes[gpu].set(xlabel=None)
    axes[gpu].set(ylabel=metric.upper())

# ax2 = fig.add_subplot(121)
# sns.catplot(ax=ax2, 
#     data=data[data["gpu_count"] > 1], kind="bar", 
#     x="method", y="time", col="gpu_count"
# )

# plt.close(2)
# plt.close(3)
plt.tight_layout()
# g.despine(left=True)
#g.set_axis_labels("", "Body mass (g)")
#g.legend.set_title("")
# matplotlib.pyplot.show()
plt.savefig(title + "_" + metric + ".pdf")
#plt.show()
